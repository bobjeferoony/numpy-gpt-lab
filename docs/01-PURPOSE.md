# 1. Purpose, scope and notation

## 1.1 The user's intended system

The project began as a request for a free local language-model app, then became a stricter requirement: build a GPT from random weights, with pure NumPy numerical computation, specializing in code debugging. The target machine has 16 GB RAM and an RTX 3050 Ti laptop GPU. The first supported language is Python.

Three ideas must be separated. A **model architecture** defines computations and trainable arrays. A **trained model** combines that architecture with weights obtained from data. An **application** supplies the user interface, loading, generation and any external tools. A functioning application and correct gradient code do not establish that the trained model can debug arbitrary programs.

The training task is conditional code generation. Given buggy code x, produce corrected code y. In the implemented prompt format, the byte sequence is:

```text
BUG:
<buggy Python code>
FIX:
<corrected Python code><end token>
```

The prompt contains no explicit specification or test results. A function name such as `second_distinct` hints at intent, but names do not provide a complete specification. Code that is valid for one intended behavior may be wrong for another. An advanced debugger should receive task intent and independently defined tests.

## 1.2 What "from scratch" means here

Token and position embeddings, attention projections, feed-forward matrices, normalization gains/biases and the output matrix are initialized within gpt.py. Random matrix entries use a seeded normal distribution with standard deviation 0.02. Normalization gains start at one; biases start at zero. No pretrained checkpoint is downloaded or imported.

NumPy supplies array operations, matrix multiplication, reductions, random-number generation and compressed array files. The project implements its own reverse-mode automatic differentiation over a small set of tensor operations. Thus it does not import a framework's autograd engine or optimizer.

This makes the mechanism inspectable. It does not make training fast. Standard NumPy on this setup runs on the CPU; the laptop GPU is not used by these files. A GPU backend would be an explicit change to the original constraint or require a different numerical implementation.

## 1.3 Notation

| Symbol | Meaning | Large model value |
|---|---|---:|
| B | batch size | 2 by default |
| T | actual sequence length in a forward pass | at most 384 |
| C | model width / embedding dimension | 128 |
| H | number of attention heads | 4 |
| D | width of one head, C/H | 32 |
| L | number of transformer blocks | 5 |
| V | vocabulary size | 257 |
| S | maximum learned position count / context | 384 |
| theta | all trainable parameters | 1,106,560 scalars |
| z | output logits, before softmax | B × T × V |
| p | next-token probabilities | B × T × V |
| m | answer scoring mask | B × T |
| eta | learning rate | schedule-dependent |

Vectors are treated as rows where needed to match NumPy expressions. A transpose in an attention equation affects the last two dimensions, leaving batch/head dimensions intact. A bar over an expression, such as bar(X), denotes a gradient of the scalar loss with respect to X. Natural logarithms are used for cross-entropy.

## 1.4 Bytes, characters, tokens and examples

The tokenizer maps UTF-8 bytes directly to IDs 0 through 255 and reserves ID 256 for the end of the answer. ASCII source has one byte per character, but Unicode names or comments may use multiple bytes per character. Therefore 384 tokens means 384 byte positions, not 384 Python tokens, words or Unicode characters.

One training **example** is a buggy/fixed pair. One **token position** is a next-byte prediction. One **step** updates weights from a batch. An **epoch** is approximately a pass over a dataset, but our trainers sample with replacement and do not implement a strict epoch iterator. These units cannot be substituted for one another.

For example, 80 training examples with batch 2 and 5000 steps means 10,000 sampled example draws. Uniform sampling yields roughly 125 draws per example on average, not 10,000 distinct debugging problems. Increasing draws cannot create missing task variety.

## 1.5 What a debugging model must learn

At least four capabilities are involved: generate syntactically valid Python; preserve relevant input structure; infer intended behavior; and make a change that meets that intent. The first can be helped by plain-code pretraining. The later ones need diverse tasks, specifications, tests and correction examples. A larger parameter count provides capacity, not an automatic source of those capabilities.

The original training examples are mostly short expressions with one operator changed. They do not teach package APIs, multi-file dependencies, runtime environments, asynchronous behavior or complex algorithmic repair. The app does not inspect repositories or execute suggestions.

## 1.6 Three feedback mechanisms

**Autoregressive feedback:** append a generated token to the input and predict another token. Weights remain fixed.

**Training feedback:** compare predicted probabilities with reference tokens, compute derivatives and change weights. The reference supplies a target; model confidence does not certify that target.

**External correctness feedback:** execute tests or use another independent check on a complete candidate and return concrete failures. The checker-and-retry system was discussed but is not implemented. Passing finite tests is useful evidence, not a proof for all possible inputs.
