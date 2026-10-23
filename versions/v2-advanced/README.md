# NumPy GPT for debugging Python

See UPGRADE.md for the advanced training workflow. On this PC use .venv-win\Scripts\python.exe; the bare python command points to MSYS2.

This is a real, small decoder-only transformer with random initial weights, causal multi-head attention, learned token and position embeddings, layer normalization, residual connections and a feed-forward network. Its numerical engine, automatic differentiation, softmax, loss and Adam optimizer use **only NumPy**. The interface and file handling use Python's standard library. No PyTorch, TensorFlow, pretrained weights, API, account or subscription is involved.

This is an educational starting point, **not a reliable debugger**. The 16 included examples demonstrate the pipeline; they cannot teach general Python debugging. Generated output may initially be gibberish, and memorizing examples does not establish debugging ability. Building useful ability from random weights needs a much larger, varied, carefully checked dataset and substantial training.

## Run on Windows

Open PowerShell in this folder. Python 3.10 or newer is required, with Tkinter (included in the usual Windows Python installation).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest -v test_gpt.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe app.py
```

Training writes `model.npz`; the app loads it. Training saves every 50 steps and on Ctrl+C. Use --resume model.npz to restore weights and optimizer state. See UPGRADE.md for the new workflow. The app never executes submitted or generated code. All training and inference run locally after installing NumPy. Standard NumPy uses the CPU, so the RTX 3050 Ti does not accelerate this implementation. The defaults fit comfortably within 16 GB RAM; runtime depends on your CPU and NumPy's BLAS configuration.

For a quicker pipeline check:

```powershell
.\.venv\Scripts\python.exe train.py --steps 10 --width 16 --layers 1 --out smoke.npz
.\.venv\Scripts\python.exe app.py --model smoke.npz
```

A ten-step checkpoint is only a mechanical demonstration. It is not trained enough to fix code. Start with short functions: the default training context is 384 **bytes**, not words. Long examples raise an error rather than being silently cut off. You can increase `--context`, but attention memory and work grow roughly with its square. Model generation uses a sliding window once it fills the context.

## The feedback loop

Training uses **teacher forcing**: it sees the buggy function, the `FIX:` marker, and the correct answer so far. At each answer position, it predicts the next byte. The future answer is hidden by the causal attention mask.

1. **Forward pass:** buggy code → embeddings → transformer → logits (scores for possible next bytes).
2. **Softmax and loss:** turn scores into probabilities, then measure how little probability the model gave the correct next byte.
3. **Backpropagation:** work backwards through every operation to calculate how each weight affected that error.
4. **Update:** clip the overall gradient norm to 1, then use Adam to make a small adjustment to the weights.
5. **Repeat:** take another batch and predict again with the updated weights.

The loss is averaged over answer bytes only. The buggy input provides context; we do not optimize the model to reproduce the question. A held-out split is evaluated without updating weights. Falling training loss with worsening held-out loss suggests overfitting. The included split is much too small to prove real-world performance.

This loop is supervised learning, not reinforcement learning. The app does **not** learn automatically when you type, accept a suggestion or correct its answer. To supply feedback, save a verified buggy/fixed pair to your dataset, then train again. Never add unchecked model suggestions as correct answers. For meaningful evaluation, keep related variants and duplicate examples together in one split, and test proposed fixes against independent tests on unseen code.

## Softmax, step by step

The model produces 257 scores: one for each of the 256 possible byte values, plus an end-of-answer token. Scores are called **logits**. They can be negative and need not sum to anything.

Softmax converts them into probabilities:

`p[i] = exp(z[i] - max(z)) / sum(exp(z[j] - max(z)))`

For three candidate scores `[2, 1, 0]`, the probabilities are approximately `[0.665, 0.245, 0.090]`. The largest score gets the highest probability and all probabilities sum to 1. Subtracting the maximum gives the same result while avoiding overflow from large exponentials.

If the correct candidate is the first one, its cross-entropy loss is `-log(0.665)`, about `0.408`. If the correct candidate is the third, the loss is `-log(0.090)`, about `2.408`: being confidently wrong costs more.

For softmax combined with cross-entropy, the gradient with respect to scores is `p - target`, where `target` is 1 for the correct candidate and 0 for the others (then scaled by the number of scored tokens). This is the error signal that starts backpropagation. The code computes log probabilities directly for numerical stability.

Softmax also appears **inside attention**. There it turns a query's scores for earlier positions into weights that sum to 1. Future positions receive an enormous negative score before softmax, giving them zero weight. The output softmax chooses a next byte; attention softmax chooses which earlier positions to use.

The app picks the highest-scoring next byte by default. The generation API also supports sampling with `temperature > 0`: divide logits by temperature before softmax. Lower temperatures concentrate probability on the strongest candidates; higher temperatures spread it out. This changes randomness, not learned knowledge.

## Add training data

Each line of `examples.jsonl` is a JSON object:

```json
{"buggy": "def add(a, b):\n    return a - b", "fixed": "def add(a, b):\n    return a + b"}
```

Use code you have permission to train on. Check that each fixed version addresses the intended bug. This first version has no natural-language bug description, so ambiguous code can have several plausible fixes. It targets small, obvious errors. A stronger dataset should include task intent, tests, and examples of already-correct code, with a corresponding prompt-format change in the trainer and app.

## Files and checks

- `gpt.py`: NumPy autodiff and transformer, checkpoint loading and generation.
- `train.py`: data preparation, supervised loss, gradient clipping and Adam.
- `app.py`: local Tkinter desktop interface.
- `test_gpt.py`: finite-difference gradients for every parameter family, causal masking, save/load round trip and a learning check.
- `examples.jsonl`: tiny demonstration dataset.

The tests validate implementation mechanics. They do not measure debugging quality. There is no sandboxed execution, repository ingestion, retrieval, model downloading or automatic code modification.

