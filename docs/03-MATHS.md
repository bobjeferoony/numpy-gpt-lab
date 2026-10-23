# 3. Softmax, loss and backpropagation

This chapter derives the equations implemented in gpt.py. It is intended to make the training mechanism auditable, including the operations usually hidden by an automatic-differentiation framework.

## 3.1 From scores to probabilities

For a vector of logits z with V components:

```text
p_i = exp(z_i) / sum_j exp(z_j)
```

Every probability is nonnegative and their sum is one, up to floating-point rounding. Larger scores mean larger probabilities. The scores themselves need not sum to one, need not be positive and are not percentages.

For numerical stability, let a = max(z):

```text
p_i = exp(z_i-a) / sum_j exp(z_j-a)
```

Multiplying numerator and denominator by exp(-a) leaves the mathematical result unchanged. Now the largest exponent is zero, whose exponential is one, avoiding overflow from a very large positive logit. Very negative relative scores can underflow to zero, which is generally acceptable for probabilities but is why loss uses log probabilities directly.

Example:

```text
z = [2,1,0]
z-max(z) = [0,-1,-2]
exp(.) = [1,0.367879,0.135335]
sum = 1.503214
p = [0.665241,0.244728,0.090031]
```

This is a distribution over three hypothetical candidates. The actual model has 257 candidates. A candidate with probability 0.665 is not "66.5% certain to fix the whole function". It is a probability assigned to one next byte under the model. Those are different events.

## 3.2 Log-softmax and cross-entropy

Define logsumexp(z) = log(sum_j exp(z_j)). A stable form is:

```text
logsumexp(z) = a + log(sum_j exp(z_j-a))
log p_i = (z_i-a) - log(sum_j exp(z_j-a))
```

For a known correct next token k, the negative log-likelihood is:

```text
loss = -log p_k
```

Using the previous distribution, the first candidate yields approximately 0.407606 nats of loss, the second 1.407606, and the third 2.407606. Increasing the correct token's probability lowers loss. Assigning zero probability would give infinite mathematical loss; stable log-softmax avoids explicitly taking log of an underflowed zero probability.

For sequence and batch positions indexed collectively by r, answer mask m_r and N = sum_r m_r:

```text
L = -(1/N) sum_r m_r * log p[r,target_r]
```

The code uses max(N,1) as a defensive denominator. Data preparation normally ensures scored targets exist. Positions with mask zero contribute neither loss nor gradient. The mask is a **loss mask**, distinct from the **causal attention mask**. Loss masking chooses what we optimize; causal masking chooses what each position is allowed to read.

Pretraining masks every position with one. Debugging fine-tuning masks the prompt positions with zero and the answer/EOS positions with one. The prompt still influences gradients through attention and embeddings; not scoring prompt predictions does not prevent learning representations of prompt tokens.

## 3.3 Deriving the softmax derivative

Let Z = sum_j exp(z_j), so p_i = exp(z_i)/Z. Quotient differentiation gives:

```text
∂p_i/∂z_j = p_i(δ_ij - p_j)
```

Here δ_ij is one when i=j and zero otherwise. Increasing a score increases its own probability but decreases others because the distribution must still sum to one.

If a later computation supplies gradient g_i = ∂L/∂p_i, then:

```text
∂L/∂z_j = sum_i g_i p_i(δ_ij-p_j)
          = p_j [g_j - sum_i g_i p_i]
```

Tensor.softmax uses exactly this vector-Jacobian product:

```python
p * (g - (g * p).sum(axis=-1, keepdims=True))
```

It does not construct the full V × V Jacobian for every row, avoiding unnecessary memory. In attention the row length is T rather than V, but the derivative is the same operation along the source-position axis.

## 3.4 Why cross-entropy produces p minus target

For one-hot target y, where y_k=1 and all other components are zero:

```text
L = -sum_i y_i log p_i
  = -sum_i y_i z_i + logsumexp(z)  because sum_i y_i = 1
∂L/∂z_j = p_j - y_j
```

With masking and averaging, the gradient is (p_j-y_j)*m_r/N. In cross_entropy.backward, the code computes exp(logp), subtracts one at the target index, multiplies by the mask/denominator and reshapes to B × T × 257.

Example with the third candidate correct:

```text
p = [0.665241,0.244728,0.090031]
y = [0,0,1]
gradient = [0.665241,0.244728,-0.909969]
```

A direct gradient-descent update z <- z - eta*gradient lowers the first two logits and raises the third. In the real model z is not a free stored parameter: the chain rule propagates this signal back through the output matrix, hidden states, blocks and embeddings.

Thus softmax itself does not look at a solution and decide whether it is correct. The reference target defines the loss. A probability distribution plus no external target is not a correctness oracle.

## 3.5 Chain rule and reverse-mode differentiation

If a scalar loss depends on a via b=f(a) and c=g(b), then:

```text
∂L/∂a = (∂L/∂c)(∂c/∂b)(∂b/∂a)
```

For arrays this becomes a sequence of vector-Jacobian products. Reverse mode is efficient for many parameters and one scalar loss because one backward traversal accumulates derivatives for all relevant parameters.

Each Tensor stores its data, gradient, parent nodes and a backward closure. Tensor.backward performs a depth-first traversal, records a topological order, seeds the scalar loss with gradient one, and invokes closures in reverse order. Shared nodes appear once in the traversal, but every incoming path adds its contribution into the same gradient array.

Persistent parameter tensors have gradients reset by model.zero_grad before a new training backward pass. Temporary graph tensors start with zero gradients when constructed. The engine is minimal: calling backward repeatedly on the same graph without resetting all internal gradients is not a supported training pattern. It has no no-grad context, graph compilation, higher-order differentiation, or automatic memory planning.

## 3.6 Addition and broadcasting

For C=A+B, each input receives the output gradient, adjusted for any broadcasting. If a bias b with shape C is added to X with shape B × T × C, then:

```text
bar(X)[b,t,c] += bar(C)[b,t,c]
bar(b)[c] += sum_b sum_t bar(C)[b,t,c]
```

unbroadcast first sums away extra leading dimensions, then sums axes where the original input size was one. Without these reductions, gradients for shared biases or position embeddings would have the wrong shape or silently wrong values.

For element-wise multiplication C=A*B:

```text
bar(A) += unbroadcast(bar(C)*B, A.shape)
bar(B) += unbroadcast(bar(C)*A, B.shape)
```

Layer-normalization gains use this multiplication rule. The gradient for gamma sums over every batch/token position where that component was used.

## 3.7 Matrix multiplication

For Y=XW with row-vector convention:

```text
bar(X) = bar(Y) W^T
bar(W) = X^T bar(Y)
```

The code swaps the final two axes to transpose batched matrices. A weight W shared across batch/token dimensions requires summing contributions from those dimensions, performed by unbroadcast after NumPy's broadcasted matmul.

For an attention product QK^T the same rule gives:

```text
bar(Q) = bar(scores) K / sqrt(D)
bar(K) = bar(scores)^T Q / sqrt(D)
```

For Z=AVals:

```text
bar(A) = bar(Z) Vals^T
bar(Vals) = A^T bar(Z)
```

The attention softmax rule links bar(A) back to bar(scores). The causal mask is a constant, so it receives an unused gradient in this minimal engine but is never optimized. Masked future scores have effectively zero softmax derivative in the normal numerical regime.

## 3.8 ReLU

```text
y_i = max(0,x_i)
bar(x_i) = bar(y_i) if x_i>0 else 0
```

At exactly zero the mathematical derivative is not unique; this implementation chooses zero. Negative units therefore do not receive a gradient through that ReLU on that example. This is one reason initialization and learning-rate scale can affect training dynamics.

## 3.9 Layer-normalization derivative

Let u=x-mean(x), r=1/sqrt(mean(u²)+epsilon), and h=u*r. Suppose g is the upstream gradient for h after the learned gain has been accounted for. For a row with C entries:

```text
bar(x) = r * [g - mean(g) - h*mean(g*h)]
```

Why these terms? The direct scaling route contributes r*g. Dependence of the row mean subtracts the mean gradient. Dependence of variance subtracts the component correlated with normalized activations. The epsilon-adjusted inverse standard deviation r is used consistently in both forward and backward.

The separately learned affine parameters have:

```text
bar(gamma) = sum_positions upstream * h
bar(beta) = sum_positions upstream
```

The compact Tensor.norm closure implements the normalization derivative; generic multiply/add closures implement gamma/beta gradients. This division keeps the normalization operation reusable.

## 3.10 Reshape, transpose and lookup

A reshape changes indexing without changing which mathematical elements are represented. Its backward reshapes the gradient to the input shape. A transpose permutes axes; backward applies the inverse permutation. These rules are crucial for separating and recombining attention heads.

For lookup Y=E[indices], backward scatter-adds each output gradient into the selected input row. Repeated indices must accumulate. `np.add.at` handles repeated-index accumulation; a naive indexed `+=` can mishandle repeated indices because advanced indexing may use buffered copies.

The same `take` method also selects Q/K/V slices from the leading dimension of the reshaped combined projection. It therefore works for both embedding tables and attention tensor selection.

## 3.11 Residual gradients

For Y=X+F(X):

```text
bar(X) = bar(Y) + J_F(X)^T bar(Y)
```

The identity route provides a direct path for information and derivatives through a block, in addition to the learned transformation route. This can help optimization; it does not mean gradients are guaranteed healthy at arbitrary depth. Again, this "feedback" is a derivative path, not a behavioral checker.

## 3.12 Gradient checking

For a normalized random direction d in a parameter array theta and small epsilon:

```text
directional finite difference = [L(theta+epsilon*d)-L(theta-epsilon*d)]/(2*epsilon)
analytic directional derivative = sum_i gradient_i*d_i
```

The test compares these values for every parameter family at float64 precision, using epsilon=1e-5 and absolute tolerance 2e-5 on a tiny model. This covers embeddings, normalization, attention matrices, feed-forward matrices and the output projection. It is stronger than merely checking that training loss falls, because a flawed derivative can sometimes still reduce a simple loss.

Nevertheless one random direction per family is not a formal proof for all shapes and inputs. ReLU nondifferentiability near zero can complicate finite differences. The implementation tests use a fixed seed and small nonpathological inputs. There is also a causality test: changing future tokens must not alter earlier logits.

## 3.13 Temperature and decoding

For positive temperature tau, sample from softmax(z/tau). As tau decreases the distribution concentrates on top scores; as tau increases it spreads. Greedy decoding instead selects argmax(z), deterministically for a fixed model/input, with NumPy's tie behavior.

There is no top-k, nucleus/top-p, beam search or correctness-aware decoding here. The app uses greedy decoding. Repeated nonsense in screenshots therefore cannot be attributed simply to high sampling temperature. More sophisticated sampling changes the search distribution, not the underlying learned competence.

## 3.14 Entropy, perplexity and confidence

Entropy H(p)=-sum_i p_i log p_i measures spread in a distribution. Cross-entropy against reference tokens measures assigned probability to observed targets. Perplexity exp(L) expresses average uncertainty on that tokenization. With byte tokens these values are not directly comparable to a subword model's token perplexity.

Uniform output over 257 symbols has loss log(257), about 5.549076 nats. The smoke runs near 5.55 are consistent with near-random prediction. A loss around 2.49 corresponds to byte perplexity around 12.1, but says nothing by itself about semantic correctness of an entire fix. A low-entropy wrong answer can be confidently invalid or behaviorally wrong.
