# 9. Worked examples and reading the implementation

## 9.1 A full repair example

Buggy input:

```python
def factorial(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Python's range excludes its upper bound. For n=5, the loop multiplies 1*2*3*4 and returns 24. The intended factorial of five is 120. A reference repair changes range(1,n) to range(1,n+1), yielding 1*2*3*4*5.

The domain is nonnegative integers. For n=0 the loop is empty and r stays one, matching 0!=1. For negative n the reference does not enforce an error; that input is outside this exercise's stated domain. A robust API might explicitly reject it, but our short training target does not include that behavior.

Training serializes the entire buggy function after BUG and the entire corrected function after FIX. It is not trained to emit a diff or just the line `range(1,n+1)`. Returning only a line can therefore fail the expected function-shape check even if that line describes the correct edit.

## 9.2 Byte-level target alignment

Consider a deliberately tiny prompt of bytes `[10,20,30]` and answer `[40,50]`, followed by EOS 256:

```text
full = [10,20,30,40,50,256]
x    = [10,20,30,40,50]
y    = [20,30,40,50,256]
mask = [ 0, 0, 1, 1,  1]
```

At input 30 the model predicts the first answer byte 40. At input 40 it predicts 50. At input 50 it predicts EOS. Masking the first two prediction positions prevents spending the debugging objective on reproducing prompt bytes. It does not remove the prompt from the network.

During generation, after choosing 40 we append it and recompute the last-position prediction. If the model mistakenly chooses 99 instead, the next prediction conditions on 99 rather than the reference 40. There is no automatic rollback or verifier in that loop.

## 9.3 A miniature attention calculation

Suppose one query has D=2 components and three visible keys:

```text
q = [1,0]
k1=[1,0], k2=[0,1], k3=[1,1]
raw dot scores = [1,0,1]
scaled scores = [0.707107,0,0.707107]
attention probabilities ≈ [0.401112,0.197776,0.401112]
```

If the values are v1=[1,0], v2=[0,2], v3=[3,1], the mixture is approximately [1.604448,0.796664]. That vector is a weighted combination of value features, not a vote among complete programs. If position 3 were future context, its score would be masked before softmax and it would contribute zero weight.

This illustration uses a single query/head. The actual model computes all queries in a batch/head at once with matrix multiplication, then recombines four heads.

## 9.4 One simple gradient update

Assume three candidate next bytes with logits [2,1,0] and correct byte index 2. Softmax yields [0.665241,0.244728,0.090031]. The gradient of one-token loss with respect to logits is [0.665241,0.244728,-0.909969]. With a hypothetical direct gradient step eta=0.1:

```text
new logits ≈ [1.933476,0.975527,0.090997]
```

The correct candidate's score rises; wrong candidates fall. Our optimizer does not store logits as free parameters, so it updates matrices and embeddings through the chain rule instead. Adam also changes the exact step scaling. The simple direct update demonstrates the sign logic, not a faithful numerical simulation of the entire optimizer.

## 9.5 Why a second-largest example needs intent

```python
def second_largest(xs):
    return sorted(xs)[-2]
```

For `[2,7,7,4]`, sorting gives `[2,4,7,7]`, so the function returns 7. If duplicates count as separate positions, this is not necessarily a bug. If the task says "second-largest distinct number" and there are at least two distinct values, a reference is:

```python
def second_largest(xs):
    return sorted(set(xs))[-2]
```

The new exercise uses a name and tests to communicate that contract to the dataset author, but the current serialized prompt still supplies only code. A better specification-conditioned task would include the requirement explicitly. Model architecture alone cannot recover information absent from the problem statement.

## 9.6 Binary search and termination

The larger example proposed in conversation used low=mid and high=mid. If mid already equals a bound and the target is absent from that interval boundary, the interval may stop shrinking. Standard integer-index binary search uses low=mid+1 or high=mid-1 after excluding mid.

That example was too long for the original input limit and was not included as a new trained template. It illustrates why a behavioral checker needs timeouts: valid Python can run forever. An AST parse cannot determine general termination, and a free-form model explanation is not a proof of progress.

## 9.7 Source file walkthrough

### gpt.py

`unbroadcast` reduces shared-gradient dimensions. `Tensor` wraps arrays and backward closures. Addition/multiplication/matmul implement differentiable primitives. Transpose/reshape/take handle shape transformations and lookups. ReLU, normalization and softmax use custom derivatives. `backward` constructs a topological order and accumulates gradients.

`cross_entropy` computes stable log probabilities and a masked token average, then seeds the compact p-y gradient. `GPT.__init__` allocates named parameter arrays. `GPT.forward` implements the decoder stack. `zero_grad` clears persistent gradients. `save`/`load` serialize weights and config. `generate` builds the BUG/FIX prompt and repeatedly chooses a byte until EOS or the generation limit.

The implementation creates graph tensors even for inference, because there is no no-grad engine. It does not retain the entire generation history as a single differentiable graph across loop iterations, but each forward pass still allocates unnecessary gradient buffers. Optimizing this is a future performance task.

### train.py

`encode` implements teacher-forced sequence alignment and masking. The CLI reads JSONL, constructs deterministic family validation, optionally restores legacy Adam arrays, samples batches, trims unused padding, clips gradients, updates weights and saves periodically. Validation does not execute generated programs. The script predates the two-stage pipeline and remains preserved as a separate workflow.

### make_data.py and make_distinct.py

The first makes renamed variants of 16 templates. The second defines 100 tasks with explicit expected outputs and verifies reference/bug differences. Both use only known source-controlled templates for execution. Neither imports public repositories or provides a general untrusted-code runner.

### prepare_pipeline.py

This collects local stdlib source into train/validation streams and records a manifest. Separately it generates AST operator mutations of trusted fixed exercises, filters by finite tests/context and writes debugging split files. It does not execute the corpus it collected.

### pipeline_train.py

The stage flag selects all-token pretraining or answer-only debugging loss. It records hashes, validates disjoint family labels for repairs, initializes fresh optimizer state, evaluates an initial baseline and saves best/last outputs. A finally block exports best weights even after a clean Ctrl+C. Its `--init` is intentionally not exact resume.

### app.py

Tkinter owns widgets on the main thread. A worker thread performs generation and puts results in a queue. The GUI polls that queue every 100 ms so the window can remain responsive while the numerical model runs. A button is disabled during the request to avoid concurrent generation against the same loaded model.

After generation, ast.parse checks syntax and searches for a function node. The result is displayed; no generated code is executed. The app does not save user feedback into training data, update parameters, or implement a conversational history/retry protocol.

### evaluate.py

This reads rows, generates an answer for each up to a length budget, checks parse/function shape, compares stripped output text with the reference and prints counts. It is valuable for detecting gross failure but cannot replace task-level behavioral testing.

### test_gpt.py and test_pipeline.py

The first tests derivatives, causality, weight round-trip and small-task learning. The second creates temporary tiny corpora, forces early stopping with a negligible learning rate, verifies best/last steps and best restoration, checks dataset hashes/config transfer, and confirms overwrite protection. Temporary test models are not evidence of a trained debugger.

## 9.8 Reading a saved loss line

```json
{"step":400,"training_loss":2.4857510748334204,"validation_loss":2.4919140802352753,"lr":0.0002957587267523652,"best_loss":2.4919140802352753}
```

This says the step-400 batch had that training loss, the fixed sampled validation windows had that weighted loss, and it was the best observed validation value so far. It does not say that 2.49 bugs remain, that repair accuracy is 75%, or that the model has learned a particular number of Python concepts. Logs describe objective measurements in nats per scored byte.
