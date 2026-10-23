# 4. Training objectives, optimization and persistence

## 4.1 What learning changes

At inference, weights are fixed and a forward pass produces logits. During training, weights change to reduce a target-defined scalar loss. Training does not attach a verbal explanation to each neuron or store each correction in a searchable dictionary. Information is distributed across numerical parameters, and the resulting behavior must be evaluated empirically.

The same transformer can be trained with different objectives. This project first tried supervised buggy-to-fixed generation alone, then added plain-Python next-byte pretraining followed by debugging fine-tuning. Pretraining supplies more syntax/idiom exposure; fine-tuning teaches the conditional prompt format and repair behavior. Neither stage guarantees the other capability.

## 4.2 Plain-Python pretraining

pipeline_train.py reads a training byte stream and a separate validation stream. For context T, it chooses start offsets uniformly with replacement and takes T+1 bytes. Inputs are the first T bytes; targets are the following T bytes:

```text
stream:   d e f   f ( x ) ...
input:    d e f   f ( x ...
target:   e f   f ( x ) ...
```

Every target position is scored. The causal mask ensures a prediction cannot read its target from a future input position. Random windows may begin inside a line or function; the model learns local continuations without being told every window begins a complete module. The corpus concatenates files with newlines, without a distinct document-boundary token. Windows can cross those boundaries. That is a known simplification.

EOS ID 256 is not present in ordinary byte streams. Pretraining therefore does not teach it as an end-of-document marker; debugging fine-tuning introduces EOS at the end of reference answers. This stage mismatch is small and intentional, but worth remembering when interpreting a pretrained model's output behavior.

The default pipeline B=2, T=384, steps=5000 predicts 3,840,000 sampled training token positions. The corpus holds 8,846,516 training bytes. Sampling with replacement means overlap and repeats are expected; "5000 steps" does not mean a complete pass over all source bytes.

## 4.3 Supervised debugging and teacher forcing

train.encode serializes prompt bytes and fixed-answer bytes plus EOS. Let prompt length be P and answer-plus-EOS length be A. The full sequence length is P+A, while x and y each have length P+A-1. The first answer byte is predicted from the final prompt byte at input position P-1.

```text
x = full_sequence[:-1]
y = full_sequence[1:]
mask[P-1 : P+A-1] = 1
```

The correct answer prefix is visible during training. This is teacher forcing: at the position predicting answer byte j, the network sees earlier *reference* bytes, not earlier generated mistakes. At inference it sees its own chosen bytes. That train/inference difference, sometimes called exposure bias, can cause a small error to lead into an unfamiliar continuation and compound.

The loss mask excludes prompt predictions but does not remove the prompt from attention. It also excludes padding. The trainer trims trailing unused positions from each batch up to the last scored target. Earlier scored positions cannot read later padding because attention is causal.

If prompt plus answer exceeds the checkpoint's context, encode raises an error. It does not silently truncate the reference. The app checks only whether the prompt itself fits; during generation it keeps the last context positions once full. Thus training has a stricter combined example-size requirement than the app's initial-input check.

## 4.4 Batch loss, sampling and noisy curves

Training batches are sampled with replacement. A batch of two long loop examples may be harder than the previous batch of two one-line arithmetic examples. Therefore training loss can rise from one evaluation point to another while general training progress continues.

A reported training loss is for the current batch, not a smoothed full-dataset loss. Validation in the pipeline is more comparable over time: pretraining uses eight fixed sampled batches from a fixed RNG seed; debugging uses all validation examples packed into batches. Their losses are weighted by scored token counts, not a simple average that gives a short answer the same weight as a much longer one.

The legacy train.py instead averages per-example validation losses equally. This difference is another reason not to compare every old logged number as though it used the same measurement procedure.

## 4.5 Global gradient clipping

Collect all parameter gradients into a conceptual vector g:

```text
norm = sqrt(sum_over_arrays sum_elements gradient^2)
scale = min(1, 1/max(norm,1e-12))
clipped_gradient = scale * gradient
```

The threshold is one. Scaling all arrays by the same factor preserves gradient direction while bounding its Euclidean magnitude. Clipping can limit extreme updates from outlier batches, but it does not guarantee convergence, repair incorrect gradients, or replace an appropriate learning rate.

The pipeline explicitly rejects nonfinite loss or gradient norm. If numerical failure occurs, the finally block attempts to save last state and export best weights. A filesystem error during saving is still possible; there is no transactional checkpoint system.

## 4.6 Adam

The optimizer tracks two exponentially smoothed arrays for each parameter: first moment m and second raw moment v. At step t with clipped gradient g_t:

```text
m_t = beta1*m_(t-1) + (1-beta1)*g_t
v_t = beta2*v_(t-1) + (1-beta2)*g_t^2
mhat_t = m_t / (1-beta1^t)
vhat_t = v_t / (1-beta2^t)
theta_t = theta_(t-1) - eta_t*mhat_t/(sqrt(vhat_t)+epsilon_adam)
beta1=0.9, beta2=0.999, epsilon_adam=1e-8
```

Initialization at zero biases early moments toward zero; the division by 1-beta^t corrects that under the standard moment interpretation. The square is element-wise. Different parameters receive different normalization from their gradient history. Adam is described by [Kingma and Ba](https://arxiv.org/abs/1412.6980); the constants and code paths here are our implementation.

Adam's momentum is not a memory of which answer was correct. It stores numerical summaries of gradient history. Restoring weights without m/v changes subsequent optimization even if the next batch is the same.

## 4.7 AdamW and weight decay

The new pipeline adds decoupled decay before the adaptive update:

```text
theta <- (1-eta_t*lambda)*theta
theta <- theta - eta_t*mhat/(sqrt(vhat)+epsilon_adam)
```

Default lambda is 0.01. In this implementation only matrix parameters with at least two dimensions are decayed, excluding `token` and `position`. Biases and normalization gains are not decayed. The exact exclusion policy is an implementation choice, not an intrinsic law of AdamW.

Adding lambda*theta to the gradient before Adam is not generally equivalent, because Adam's adaptive scaling would also act on that term. This distinction motivates decoupled decay; background is [Loshchilov and Hutter](https://arxiv.org/abs/1711.05101).

Weight decay can discourage unconstrained parameter growth and sometimes improve generalization. With 80 examples it cannot create missing information about unseen bug families. Dropout was discussed as another regularizer but is absent from all captured numerical code.

## 4.8 Warmup and cosine decay

For a run with R requested steps, pipeline warmup factor and decay are:

```text
warmup(t) = min(1, t/min(100,R))
decay(t) = 0.1 + 0.45*[1+cos(pi*t/R)]
eta_t = base_lr * warmup(t) * decay(t)
```

The decay factor begins near one and ends at 0.1. It is multiplied by warmup rather than beginning only after warmup completes. Thus early learning rates reflect both factors. The legacy trainer uses the same-shaped decay but warmup min(50,R), and its default base rate is 0.001 instead of the pipeline's 0.0003.

Fine-tuning instructions lower the base rate to 0.0001. That is a conservative choice to adapt pretrained weights, not a measured optimum. The run metadata records actual arguments so future comparisons can distinguish changed data from changed optimization settings.

## 4.9 Best-checkpoint saving and early stopping

Before any update, the pipeline evaluates validation loss and saves step zero as the initial best. At each evaluation it considers a score improved only if it is at least 1e-5 lower than the previous best. On improvement, best state is saved and the non-improvement counter resets. Otherwise the counter increments. At the default patience eight, eight consecutive non-improving evaluations stop training.

Validation is measured every 50 steps by default and on the final requested step. This means patience is measured in **evaluations**, not examples or raw steps. An improvement at step 250 followed by worse performance at step 300 need not stop training; later recovery is allowed. The actual captured pretraining history demonstrates that: validation rose slightly at 300 then improved at 350 and 400.

Early stopping minimizes a particular validation metric. It does not guarantee best behavioral repair accuracy, and repeatedly tuning on the same validation data can overfit the evaluation procedure. A separate final test set remains necessary.

## 4.10 File formats and exact state

GPT.save produces an NPZ with parameter arrays and a config vector ordered as width, context, heads, layers, seed. GPT.load reads these without pickle. The pipeline optimizer checkpoints additionally contain step and keys `adam_m.<parameter>` / `adam_v.<parameter>`. Config order relies on the dictionary insertion order in GPT.__init__; changing it would require a format migration.

The new pipeline writes:

| Suffix | Meaning |
|---|---|
| requested `.npz` | Exported best weights only |
| `.best.npz` | Best weights with Adam arrays and its step |
| `.last.npz` | Most recent weights with Adam arrays and reported step |
| `.history.jsonl` | Initial validation plus evaluation rows |
| `.run.json` | Stage, arguments, config, data hashes and optional init hash |

The final requested .npz lacks a step key, even though its values came from the best checkpoint. It is not evidence that the run had zero steps. Use the associated best/history files to find the selected step.

The archive found `python-base.best.npz` at step 400 and `python-base.last.npz` at step 422. Best validation loss at the last recorded evaluation is 2.4919140802352753. The final weights-only export matches the best checkpoint's arrays. This is partially completed pretraining, not evidence of a new debugging fine-tune or a general debugging ability.

## 4.11 Pause, resume and interruption limits

Ctrl+C raises KeyboardInterrupt. The pipeline catches it, saves last state in finally, then loads best weights and exports the requested output. It must finish saving before the terminal is closed. Killing the process or powering off skips that Python cleanup.

**Legacy train.py --resume:** restores stored weights, m/v and step if present. It resets the run's random generator and local learning-rate schedule, so it is not bit-for-bit continuation. A weight-only checkpoint starts fresh optimizer state.

**pipeline_train.py --init:** restores weights only. It starts new m/v, step count, RNG and schedule. This is appropriate when changing objectives between pretraining and fine-tuning. It is also a way to continue learning from paused weights, but is not exact optimizer resume. Use a new output name because the pipeline refuses existing outputs.

There is a further subtlety: Ctrl+C may interrupt inside a parameter update. The saved `step` variable is set before the entire iteration completes, so last state could contain a partially applied step. Neither trainer saves an atomic before/after transaction for every update. Best checkpoints written after evaluation are a safer consistently evaluated point, though writes themselves are not atomic. A stronger continuation design would save RNG state, absolute schedule state, best score, patience, dataset fingerprints, optimizer arrays and a completed-step counter, writing to a temporary file then atomically replacing a checkpoint.

## 4.12 Reproducibility boundaries

Seeds fix initialization and sampling logic within a given implementation. Different NumPy versions, CPU BLAS libraries or threading may still lead to floating-point differences. The archive stores source/data hashes and records the local verification environment, but the original requirements file allows a range of NumPy versions. Historical runs without a metadata file cannot be reconstructed exactly from final weights alone.

The pretraining validator samples only eight fixed batches, about 6144 token positions at defaults, from a million-byte validation corpus. That makes comparisons stable but only approximately representative. More validation windows and file-family grouping would strengthen the measurement at extra compute cost.
