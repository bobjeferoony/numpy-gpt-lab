# 11. Glossary and references

## Glossary

**Activation:** an intermediate value produced by a layer for a specific input, distinct from a persistent trainable parameter.

**Adam:** optimizer using exponential first/second gradient moments with bias correction.

**AdamW:** Adam with weight decay applied separately from the adaptive gradient term.

**Attention:** learned weighted mixing of value vectors according to query/key scores.

**Autodiff:** mechanical chain-rule differentiation of a computational graph.

**Autoregressive:** predicting a next token conditioned on earlier tokens.

**Batch:** examples/windows whose losses contribute to one optimizer update.

**Byte token:** an ID corresponding to a UTF-8 byte, not necessarily a complete character or word.

**Causal mask:** restriction that blocks reading future sequence positions.

**Checkpoint:** saved model arrays and possibly optimizer/configuration state; its contents depend on the writer.

**Context:** maximum number of input token positions the checkpoint supports in one forward pass.

**Cross-attention:** attention whose queries and keys/values come from different sequences, absent from our decoder-only model.

**Cross-entropy:** negative log probability assigned to target tokens, averaged under the chosen loss mask.

**Decoder-only transformer:** stack with causal self-attention that predicts continuations without a separate encoder.

**Embedding:** trainable mapping from discrete IDs or positions into vectors.

**EOS:** end-of-answer token, ID 256 here.

**Epoch:** roughly a dataset pass; our replacement samplers use steps rather than exact epochs.

**Fine-tuning:** further training from existing weights on a more specific objective/dataset.

**Generalization:** successful behavior beyond training examples and superficial copies.

**Gradient:** derivative indicating local sensitivity of loss to a value or parameter.

**Greedy decoding:** choosing the maximum-logit token at each step.

**Head:** one lower-dimensional attention projection/mixture within multi-head attention.

**Inference:** using fixed weights to produce predictions.

**Layer normalization:** per-token normalization across feature dimensions with learned affine parameters.

**Logit:** unnormalized score before softmax.

**Loss mask:** weights deciding which predicted positions contribute to training loss.

**Mutation testing:** deliberately changing code and checking whether tests detect the change.

**Overfitting:** fitting training-specific patterns that do not transfer to the intended unseen distribution.

**Parameter:** persistent learned scalar in a model array; optimizer buffers and activations are separate.

**Perplexity:** exp(mean negative log-likelihood), meaningful only relative to the tokenization and evaluation procedure.

**Pretraining:** broad next-token learning before a narrower task adaptation stage.

**Residual connection:** addition of a sublayer's input around its output, carrying information and gradients.

**Softmax:** conversion of scores to a normalized probability distribution.

**Teacher forcing:** training each next-token prediction using the reference prefix rather than generated prefix.

**Temperature:** scaling logits before sampling to change concentration; it does not add knowledge.

**Validation:** data used for model selection and monitoring; distinct from an untouched final test.

**Weight decay:** explicit shrinking of selected parameters during optimization.

## Primary references

The following primary papers were checked while preparing this book. Their research results are not performance claims about this implementation. The detailed local equations, array shapes, parameter counts, dataset inventories and behavioral limitations are derived from the archived source and files, not copied from the papers.

1. Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017. Background for the transformer and the encoder-decoder diagram. Our code is a smaller decoder-only variation with pre-normalization and learned position embeddings.
2. Kingma and Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), 2014/2015. Background for adaptive first/second moment updates and bias correction.
3. Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), 2017/2019. Background for separating parameter shrinkage from Adam's adaptive gradient calculation.
4. Ba, Kiros and Hinton, [Layer Normalization](https://arxiv.org/abs/1607.06450), 2016. Background for normalization over a layer's feature dimensions.

## Local primary evidence

- gpt.py: exact numerical implementation and parameter layout.
- train.py and pipeline_train.py: objectives, samplers, schedules and saving behavior.
- make_data.py, make_distinct.py and prepare_pipeline.py: dataset generation and checks.
- app.py and evaluate.py: actual output checks, generation and user interaction.
- test_gpt.py and test_pipeline.py: executable mechanical verification.
- pipeline-data/sources.json and report.json: corpus provenance and counts.
- pipeline-data/PYTHON-LICENSE.txt: installed Python license copied during preparation.
- records/PUBLICATION.md: public-edition boundaries and sanitization policy.
- records/CHECKPOINTS.json: observed checkpoint architecture/state inventory.
- python-base.history.jsonl and python-base.run.json: observed pretraining progress and configuration.
- Private screenshots are excluded; observed failures are summarized in the history chapter.
- versions/: sanitized historical source snapshots.

## Licensing and attribution boundaries

Original project code, exercises, documentation and project-generated checkpoints are available under the MIT license. Bundled standard-library corpus material retains its original copyright notices and Python license; see THIRD_PARTY_NOTICES.md. Private screenshots and the supplied third-party architecture image are excluded. Source catalogs contain relative paths only.
