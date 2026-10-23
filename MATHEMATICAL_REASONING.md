# Mathematical reasoning: how this NumPy GPT learns and why it can fail

This is the mathematical companion to the [complete technical book](TECHNICAL_GUIDE.md). It explains the implemented model, derives its main gradients, connects each equation to the code, and separates learning from checking an answer. The book contains the broader project history, operational details, dataset experiments and proposed checker design.

The equations below describe this repository's implementation, not an assertion that it has achieved reliable debugging. All numerical operations are implemented with NumPy. The complete model is in [gpt.py](gpt.py); training objectives and optimizers are in [train.py](train.py) and [pipeline_train.py](pipeline_train.py).

## Contents

1. [The problem the model is actually optimizing](#1-the-problem-the-model-is-actually-optimizing)
2. [Representation and dimensions](#2-representation-and-dimensions)
3. [The complete forward pass](#3-the-complete-forward-pass)
4. [Softmax, probabilities and numerical stability](#4-softmax-probabilities-and-numerical-stability)
5. [The loss and its derivative](#5-the-loss-and-its-derivative)
6. [How backpropagation reaches every weight](#6-how-backpropagation-reaches-every-weight)
7. [Optimizer, clipping and learning-rate logic](#7-optimizer-clipping-and-learning-rate-logic)
8. [Parameter count, memory and computation](#8-parameter-count-memory-and-computation)
9. [Pretraining, fine-tuning and generalization](#9-pretraining-fine-tuning-and-generalization)
10. [The three feedback loops](#10-the-three-feedback-loops)
11. [Testing the mathematics and interpreting results](#11-testing-the-mathematics-and-interpreting-results)
12. [Where every equation lives](#12-where-every-equation-lives)

## 1. The problem the model is actually optimizing

Let $x$ be buggy Python and $y=(y_1,\ldots,y_A)$ its reference correction, including an end-of-answer token. The model represents a conditional probability distribution:

$$
p_\theta(y\mid x)
=\prod_{t=1}^{A}p_\theta(y_t\mid x,y_1,\ldots,y_{t-1}).
$$

The trainable arrays are collectively $\theta$. This factorization is the chain rule of probability: predicting a whole function is broken into next-token predictions conditioned on its prefix.

Our tokens are UTF-8 bytes, not Python grammar tokens. A complete answer can have many individually plausible next-byte choices and still be wrong as a program. Even if each of 100 required byte choices had conditional probability 0.99 along a particular answer, the probability of that exact sequence would be $0.99^{100}\approx0.366$. This is an illustrative calculation, not a measured accuracy or an assumption of independent model errors: the probabilities in the product are conditioned on the preceding sequence.

We minimize the negative log probability of reference answers:

$$
-\log p_\theta(y\mid x)
=-\sum_{t=1}^{A}\log p_\theta(y_t\mid x,y_{1:t-1}).
$$

The logarithm turns a product into a sum, making a long-sequence objective manageable. It also means confidently assigning very little probability to the correct byte is penalized heavily.

This objective rewards reference prediction. It does not directly ask whether the generated program passes tests. A correct alternative implementation can differ from the reference; a fluent continuation can fail the task. Those distinctions motivate behavioral evaluation outside the model.

## 2. Representation and dimensions

| Symbol | Meaning | Large configuration |
|---|---|---:|
| $B$ | Batch size | 2 |
| $T$ | Actual input length | At most 384 bytes |
| $S$ | Maximum learned positions | 384 |
| $C$ | Hidden/embedding width | 128 |
| $H$ | Attention heads | 4 |
| $D=C/H$ | Width per head | 32 |
| $L$ | Transformer blocks | 5 |
| $V$ | Vocabulary | 257 |

IDs 0–255 represent byte values. ID 256 means end of answer. Non-ASCII source may consume several bytes per character. Therefore 384 positions do not mean 384 words or even 384 Unicode characters.

Token embedding $E\in\mathbb{R}^{V\times C}$ and position embedding $P\in\mathbb{R}^{S\times C}$ produce:

$$
X_{b,t,:}=E_{I_{b,t},:}+P_{t,:},
\qquad X\in\mathbb{R}^{B\times T\times C}.
$$

Embeddings turn discrete symbols into trainable vectors. Positions make order distinguishable. No vector component is manually assigned a meaning such as “this is a loop”; any useful representation emerges from the loss and training data.

## 3. The complete forward pass

### 3.1 Layer normalization and its purpose

For a single token vector $u\in\mathbb{R}^{C}$:

$$
\mu=\frac1C\sum_{i=1}^{C}u_i,\qquad
\sigma^2=\frac1C\sum_{i=1}^{C}(u_i-\mu)^2,
$$

$$
\widehat{u}_i=\frac{u_i-\mu}{\sqrt{\sigma^2+\epsilon}},
\qquad
\mathrm{LN}(u)_i=\gamma_i\widehat{u}_i+\beta_i.
$$

Here $\epsilon=10^{-5}$, and $\gamma,\beta$ are learned. The normalization is per token across features, not a batch mean. It regulates the scale presented to a sublayer, while retaining relative feature differences and allowing learned affine adjustment.

Our transformer uses pre-normalization: normalize before attention/feed-forward, then add the residual. The original encoder-decoder reference diagram used a different ordering. This project also has no encoder or cross-attention.

### 3.2 Queries, keys and values

For normalized hidden states $U=\mathrm{LN}_1(X)$:

$$
[Q\mid K\mid R]=UW_{\mathrm{qkv}}+b_{\mathrm{qkv}}.
$$

$W_{\mathrm{qkv}}$ has shape $C\times3C$. The combined projection is reshaped and divided into heads, giving $Q,K,R\in\mathbb{R}^{B\times H\times T\times D}$. The letter $R$ here denotes value vectors to avoid confusing them with vocabulary size $V$.

Queries and keys determine matching scores. Values are what gets combined. “What a token seeks” and “what a token offers” are useful intuitions, not literal hand-coded roles.

### 3.3 Scaled scores and the causal mask

For each batch and head:

$$
Z_{t,s}=\frac{Q_tK_s^\mathsf{T}}{\sqrt D}+M_{t,s},
\qquad
M_{t,s}=
\begin{cases}
0,&s\le t,\\
-10^9,&s>t.
\end{cases}
$$

If approximately independent query/key components have unit-scale variance, a sum of $D$ products has variance proportional to $D$. Dividing by $\sqrt D$ counteracts width-driven score growth, reducing the tendency to saturate softmax solely because heads are wider. Learned distributions need not continue to satisfy that simplifying variance assumption.

The mask prevents target leakage from future positions. Position $t$ may see its own input token because it predicts the token at $t+1$. During training, many reference-prefix predictions are computed at once, but none may inspect its future target.

### 3.4 Attention as a weighted sum

Apply softmax over source positions $s$:

$$
A_{t,s}=\frac{\exp Z_{t,s}}{\sum_j\exp Z_{t,j}},
\qquad
O_t=\sum_s A_{t,s}R_s.
$$

Each attention row sums to one. Different heads learn different projections and weighted mixtures. Their outputs are concatenated, projected and added to the residual stream:

$$
X'=X+\mathrm{Concat}(O_1,\ldots,O_H)W_o+b_o.
$$

This is numerical feature mixing. It is not a vote over complete repairs or a correctness test.

### 3.5 Feed-forward transformation and residuals

$$
F=\mathrm{ReLU}\left(\mathrm{LN}_2(X')W_{\mathrm{up}}+b_{\mathrm{up}}\right),
$$

$$
X''=X'+FW_{\mathrm{down}}+b_{\mathrm{down}}.
$$

$W_{\mathrm{up}}\in\mathbb{R}^{C\times4C}$ and $W_{\mathrm{down}}\in\mathbb{R}^{4C\times C}$. ReLU applies $\max(0,a)$ element-wise. Attention directly mixes positions; this feed-forward sublayer transforms each position's features.

Residual addition offers an identity path around each learned sublayer. Its derivative contains an identity term, which can help gradient flow. A residual arrow is not an answer-feedback checker.

After $L$ blocks, the output scores are:

$$
\ell=\mathrm{LN}_{\mathrm{final}}(X_L)W_{\mathrm{out}},
\qquad
W_{\mathrm{out}}\in\mathbb{R}^{C\times V}.
$$

The implementation has no output bias and does not tie $W_{\mathrm{out}}$ to the token embedding matrix.

## 4. Softmax, probabilities and numerical stability

For one output logit vector $\ell$:

$$
p_i=\frac{e^{\ell_i}}{\sum_j e^{\ell_j}}.
$$

Larger scores receive larger probabilities. Scores can be negative and need not sum to anything; probabilities sum to one.

For stability we subtract $a=\max_j\ell_j$:

$$
p_i=\frac{e^{\ell_i-a}}{\sum_j e^{\ell_j-a}}.
$$

The common factor cancels, leaving the same mathematical distribution. The largest exponential becomes one instead of potentially overflowing.

With scores $[2,1,0]$:

$$
e^{[0,-1,-2]}\approx[1,0.367879,0.135335],
$$

$$
p\approx[0.665241,0.244728,0.090031].
$$

The first candidate's 66.5% next-token probability is not a 66.5% chance that the entire proposed fix is correct. Softmax has no access to an independent program specification or test oracle.

There are two uses of softmax here: attention normalizes scores over earlier positions; the output distribution normalizes logits over possible next bytes. They use the same mathematical operation along different axes.

The app chooses the largest logit greedily. Softmax preserves ordering, so computing probabilities first would not change that choice. Sampling with temperature $\tau>0$ instead uses:

$$
p_i(\tau)=\frac{e^{\ell_i/\tau}}{\sum_j e^{\ell_j/\tau}}.
$$

Lower temperature concentrates on high scores; higher temperature spreads probability. It does not increase knowledge. The current app's malformed answers cannot be explained merely by high-temperature sampling because it uses greedy decoding.

## 5. The loss and its derivative

### 5.1 Stable log probabilities

Computing probabilities then taking a logarithm can produce $\log0$ after underflow. Instead:

$$
\log p_i=(\ell_i-a)-\log\sum_j e^{\ell_j-a}.
$$

For target ID $k$ the per-position loss is $-\log p_k$. With the example above, target 1 has loss about $0.407606$ nats; target 3 has loss about $2.407606$ nats.

### 5.2 Answer-only masking

Let $m_r$ select scored batch/sequence positions and $N=\sum_r m_r$:

$$
\mathcal L=-\frac1N\sum_r m_r\log p_{r,k_r}.
$$

Pretraining sets every $m_r=1$. Debugging fine-tuning scores only corrected-answer bytes and the end token. The prompt is still available to attention and receives gradients through its effect on answers.

Example alignment, with a three-byte prompt and two-byte answer:

| Sequence | Values |
|---|---|
| Full sequence | 10, 20, 30, 40, 50, 256 |
| Input | 10, 20, 30, 40, 50 |
| Next-token target | 20, 30, 40, 50, 256 |
| Scoring mask | 0, 0, 1, 1, 1 |

The last prompt byte predicts the first answer byte. The loss mask chooses what to score; the separate causal mask chooses what can be read.

### 5.3 Deriving the softmax Jacobian

Using the quotient rule:

$$
\frac{\partial p_i}{\partial\ell_j}
=p_i(\delta_{ij}-p_j).
$$

An increase to a logit raises its own probability and lowers others through the shared denominator. For upstream gradient $g_i=\partial\mathcal L/\partial p_i$:

$$
\frac{\partial\mathcal L}{\partial\ell_j}
=\sum_i g_i p_i(\delta_{ij}-p_j)
=p_j\left(g_j-\sum_i g_ip_i\right).
$$

That vector-Jacobian product is implemented without constructing a full square Jacobian, saving memory.

### 5.4 Why the training signal is probability minus target

Let one-hot target $y$ have one at the correct index. Since $\sum_i y_i=1$:

$$
\mathcal L=-\sum_i y_i\log p_i
=-\sum_i y_i\ell_i+\log\sum_j e^{\ell_j}.
$$

Differentiating gives:

$$
\boxed{\frac{\partial\mathcal L}{\partial\ell_j}=p_j-y_j.}
$$

For the masked average, multiply by $m_r/N$. If the third candidate is correct:

$$
p-y\approx[0.665241,\;0.244728,\;-0.909969].
$$

Gradient descent subtracts this direction: it pushes the correct candidate's score up and the others down. The actual parameters are matrices, not free logits; backpropagation applies the chain rule to find how those matrices should change.

## 6. How backpropagation reaches every weight

### 6.1 Reverse-mode automatic differentiation

The custom Tensor stores an array, accumulated gradient, parents and a backward function. A topological traversal ensures a child is processed before gradients are sent to its parents. The scalar loss starts with derivative one. Shared inputs accumulate contributions from every path.

This is the chain rule applied mechanically, not a separate learned critic. Parameters must have their gradients cleared between training steps. The tiny engine is designed for one backward call on each newly built graph, not repeated higher-order differentiation.

### 6.2 Addition, multiplication and broadcast sharing

For $Y=A+B$, each input receives the output gradient. If an input was broadcast, its gradient sums across the dimensions over which it was reused.

For $Y=A\odot B$:

$$
\overline A=\overline Y\odot B,
\qquad
\overline B=\overline Y\odot A,
$$

followed by broadcast reductions. A bias shared across $B\times T$ positions receives the sum of all their contributions. The bar denotes a derivative of the final loss with respect to that array.

### 6.3 Matrix multiplication

For $Y=XW$:

$$
\overline X=\overline YW^\mathsf T,
\qquad
\overline W=X^\mathsf T\overline Y.
$$

For shared weights, sum over the batch and position dimensions. For attention $Z=QK^\mathsf T/\sqrt D$:

$$
\overline Q=\overline ZK/\sqrt D,
\qquad
\overline K=\overline Z^\mathsf TQ/\sqrt D.
$$

For its weighted mixture $O=AR$:

$$
\overline A=\overline OR^\mathsf T,
\qquad
\overline R=A^\mathsf T\overline O.
$$

The softmax vector-Jacobian product connects $\overline A$ to $\overline Z$.

### 6.4 Layer normalization, including the dependence of the statistics

Define centered input $u=x-\mu$, inverse standard deviation $r=(\sigma^2+\epsilon)^{-1/2}$, and normalized input $h=ur$. Let $g$ be the upstream derivative with respect to $h$ after accounting for the learned gain. Differentiating both the direct route and the mean/variance routes yields:

$$
\boxed{\overline x
=r\left(g-\mathrm{mean}(g)
-h\cdot \mathrm{mean}(g\odot h)\right).}
$$

The first term is direct scaling. The second subtracts the shared-mean contribution. The third corrects for variance changing when the input changes. Treating the mean and variance as constants would give the wrong derivative.

For the separate affine parameters:

$$
\overline\gamma=\sum_{\text{positions}}\overline y\odot h,
\qquad
\overline\beta=\sum_{\text{positions}}\overline y.
$$

### 6.5 ReLU, lookups and residual connections

ReLU's derivative is one for positive input and zero for negative input; this code chooses zero at exactly zero. Reshape reverses by reshaping the gradient back, and transpose reverses by the inverse axis permutation.

Embedding gradients use scatter-add. If the same byte occurs three times, all three derivatives must accumulate into its one embedding row. Buffered advanced-index assignment can lose repeated-index contributions; the implementation uses <code>np.add.at</code>.

For a residual sublayer $Y=X+F(X)$:

$$
\overline X=\overline Y+J_F(X)^\mathsf T\overline Y.
$$

The identity term provides a direct derivative path, alongside the learned transformation's path. It does not mean the sublayer checks or revises a completed solution.

## 7. Optimizer, clipping and learning-rate logic

### 7.1 Global gradient clipping

Combine all parameter gradients conceptually into a vector $g$. The implementation rescales to Euclidean norm at most one:

$$
\widetilde g
=g\min\left(1,\frac1{\max(\lVert g\rVert_2,10^{-12})}\right).
$$

The direction is preserved while extreme magnitude is limited. Clipping does not repair incorrect gradients or guarantee convergence.

### 7.2 Adam and bias correction

For step $t$, the optimizer stores first and second moment arrays:

$$
m_t=0.9m_{t-1}+0.1\widetilde g_t,
\qquad
v_t=0.999v_{t-1}+0.001\widetilde g_t^2.
$$

They start at zero. Bias correction compensates for that initial underestimation:

$$
\widehat m_t=\frac{m_t}{1-0.9^t},
\qquad
\widehat v_t=\frac{v_t}{1-0.999^t}.
$$

The adaptive update is:

$$
\theta_t=\theta_{t-1}
-\eta_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+10^{-8}}.
$$

These moment arrays summarize numerical gradients, not examples or explanations. Loading weights without moments changes how later learning proceeds.

### 7.3 AdamW

The new pipeline separately decays selected matrices before the adaptive update:

$$
\theta\leftarrow(1-\eta_t\lambda)\theta,
\qquad \lambda=0.01\text{ by default}.
$$

Embeddings, normalization parameters and biases are excluded by this implementation's policy. Adding $\lambda\theta$ to the gradient before Adam is not generally equivalent because the adaptive normalization would also act on that added term.

Decay discourages parameter growth; it does not manufacture missing task diversity. Dropout was discussed but is not implemented.

### 7.4 Warmup and cosine decay

For $R$ requested steps, the new pipeline uses:

$$
\eta_t=\eta_0
\min\left(1,\frac{t}{\min(100,R)}\right)
\left[0.1+0.45\left(1+\cos\frac{\pi t}{R}\right)\right].
$$

Warmup and cosine factors are multiplied throughout the run. The decay floor is $0.1\eta_0$ at the final step after warmup. This is not a schedule that waits until warmup ends to begin its cosine phase.

Best-checkpoint selection evaluates a fixed validation procedure and retains a new best only after improvement beyond $10^{-5}$. Early stopping counts consecutive non-improving evaluations, not raw training examples. Validation used to choose weights is not an untouched final test set.

## 8. Parameter count, memory and computation

Each block has:

$$
\begin{aligned}
\text{QKV}&:\;3C^2+3C,\\
\text{attention output}&:\;C^2+C,\\
\text{feed-forward up}&:\;4C^2+4C,\\
\text{feed-forward down}&:\;4C^2+C,\\
\text{two layer norms}&:\;4C.
\end{aligned}
$$

Thus one block has $12C^2+13C$ parameters. Token embeddings, positions, untied output matrix and final norm give:

$$
\boxed{N=(2V+S+2)C+L(12C^2+13C).}
$$

For $V=257,S=384,C=128,L=5$:

$$
N=900(128)+5\left(12(128)^2+13(128)\right)
=115200+991360
=1{,}106{,}560.
$$

The four heads divide width into four groups of 32; they do not multiply the projection parameter count when $C$ is fixed.

Dense sublayers cost approximately $O(BTC^2)$ per block. Attention matrix products add $O(BT^2C)$. Attention scores require $O(BHT^2)$ storage, so doubling context quadruples that part of memory.

At the large defaults, one attention-score array contains $2\cdot4\cdot384^2=1{,}179{,}648$ float64 values, or 9 MiB. A single weight set occupies about 8.44 MiB; weights, gradients and two Adam buffers alone occupy about 33.8 MiB. Retained activations, temporary arrays, graph objects and numerical workspaces increase actual process memory well beyond that lower bound.

## 9. Pretraining, fine-tuning and generalization

### 9.1 Why plain-code pretraining is a separate stage

Learning Python syntax and learning repairs are different demands. Plain-code pretraining scores every next byte across a much larger corpus. Debugging fine-tuning then adapts those weights to the BUG/FIX format and reference repair distribution.

The prepared corpus contains 8,846,516 training bytes. A run with batch two, context 384 and 5000 steps scores:

$$
2\cdot384\cdot5000=3{,}840{,}000
$$

sampled token positions, with overlap. That run does not necessarily see every training byte. It is not equivalent to millions of distinct debugging examples.

### 9.2 Why renamed examples did not solve the problem

The earlier 640-row dataset had only 16 underlying templates. Repeatedly changing a function name teaches limited variation. The distinct dataset has 100 different exercises, split into 80 training and 20 validation tasks. Mutation checks yield 162 related pairs, not 162 independent specifications.

For 1,106,560 parameters and 80 training tasks, the ratio is 13,832 parameters per task. That highlights limited variety but is not a universal threshold for failure. Generalization depends on token count, task information, redundancy, model inductive bias and the evaluation distribution.

### 9.3 Teacher forcing and compounding mistakes

During training, answer position $t$ sees the correct reference prefix. During generation it sees its own prefix. An early invalid byte can therefore place later predictions in contexts unlike training. This is one reason lower reference loss does not ensure correct free generation.

A short context also limits task intent and test feedback. Increasing context helps representation capacity but raises attention cost. It does not supply missing specifications automatically.

## 10. The three feedback loops

| Loop | Information returned | Weight update? | Status |
|---|---|---|---|
| Autoregressive generation | Previously generated bytes | No | Implemented |
| Supervised training | Reference-token prediction error | Yes | Implemented |
| External checker and retry | Syntax errors, test failures, expected/actual outputs | Not by itself | Syntax check only; behavioral retry proposed |

The user's proposed “brain plus checker” fits the third row. A stronger future system would generate a candidate, validate its structure, run trusted tests in an actual isolation boundary, return concrete failures, and retry within a bounded budget.

The checker needs independent evidence. Asking the same small model to agree with its answer is not equivalent to checking behavior. Confidence and correctness can diverge.

Permanent learning would be a separate step: collect failed attempts and independently verified corrections, review them, assign stable train/test families, and train a new checkpoint. Automatically treating every self-generated answer as a correct training target can reinforce errors.

The current app never executes suggestions or updates weights from interaction. The [feedback design chapter](docs/06-FEEDBACK-DESIGN.md) explains the missing controller, test oracle, prompt format and isolation requirements. A timeout-only subprocess is not a secure sandbox.

## 11. Testing the mathematics and interpreting results

### 11.1 Finite-difference gradient checking

For normalized direction $d$ in parameter space:

$$
\nabla_\theta\mathcal L\cdot d
\approx
\frac{\mathcal L(\theta+\varepsilon d)-\mathcal L(\theta-\varepsilon d)}
{2\varepsilon}.
$$

The gradient test uses float64, $\varepsilon=10^{-5}$ and absolute tolerance $2\cdot10^{-5}$ for one random direction in each parameter family. It checks whether the implemented backward computations match numerical loss changes locally.

It does not prove every possible input/shape is correct. ReLU's nondifferentiability at zero also matters when designing numerical checks. Additional tests check causality, save/load equality, tiny-task learning, early stopping and stage transfer.

### 11.2 What loss can and cannot establish

Uniform probabilities over 257 tokens give:

$$
\mathcal L_{\mathrm{uniform}}=\log257\approx5.549076.
$$

The observed partial pretraining run reached about 2.4919 nats per scored validation byte by step 400. That corresponds to byte perplexity:

$$
\mathrm{PPL}=\exp(\mathcal L)\approx12.1.
$$

This is evidence of improved byte modeling on the chosen validation windows. It is not a repair-accuracy percentage and does not establish that a particular bug can be fixed.

The app's parse check establishes grammatical shape only. Exact reference matching is stronger for reproducing one answer but rejects alternative correct repairs. Passing independently written behavioral tests provides stronger task evidence, but finite tests still do not prove correctness for every possible input.

### 11.3 Why twenty validation tasks are noisy

For independent binary task outcomes with success probability $p$, an estimated success rate based on $n$ tasks has approximate standard error:

$$
\mathrm{SE}\approx\sqrt{\frac{p(1-p)}n}.
$$

At $p=0.5,n=20$, this is about 0.112. Correlated mutations make the effective sample size smaller than a naive row count suggests. For small samples or extreme rates, report an appropriate interval rather than relying only on this normal approximation.

An untouched final test set, task-family-aware splits and honest baseline comparisons are therefore part of the scientific logic, not optional presentation details.

## 12. Where every equation lives

| Concept | Implementation |
|---|---|
| Embeddings, Q/K/V, causal scores, residuals and feed-forward | [GPT.forward in gpt.py](gpt.py) |
| Layer-normalization formula and derivative | Tensor.norm in [gpt.py](gpt.py) |
| Softmax and its vector-Jacobian product | Tensor.softmax in [gpt.py](gpt.py) |
| Stable log-softmax, masked loss and probability-minus-target | cross_entropy in [gpt.py](gpt.py) |
| Matrix/broadcast/lookup derivative rules | Tensor and unbroadcast in [gpt.py](gpt.py) |
| Reverse-mode traversal | Tensor.backward in [gpt.py](gpt.py) |
| BUG/FIX serialization and answer-only mask | encode in [train.py](train.py) |
| Pretraining, AdamW, schedule and early stopping | [pipeline_train.py](pipeline_train.py) |
| Finite-difference and causal checks | [test_gpt.py](test_gpt.py) |
| Best/last behavior and stage transfer checks | [test_pipeline.py](test_pipeline.py) |
| Dataset construction, domains and leakage | [Data/evaluation chapter](docs/05-DATA-EVALUATION.md) |
| Current inference interface and syntax warning | [app.py](app.py) |

For the complete surrounding reasoning, continue with the [15,000-word technical book](TECHNICAL_GUIDE.md) or its [chapter index](docs/00-INDEX.md). Background papers are listed in the [references chapter](docs/11-REFERENCES.md); our specific shapes, constants, data counts and limitations are grounded in this repository's source and saved records.
