# 10. Limitations, verification and next experiments

## 10.1 Preserve the distinction between correctness and competence

A numerically correct implementation can still train a poor model. Passing finite-difference gradients means the tested derivatives agree locally with numerical estimates. It does not prove adequate data, optimization duration, language understanding or code-repair reliability. Likewise a parseable function can be wrong, unsafe, irrelevant or incompatible with the intended API.

The current model is best described as an inspectable educational transformer with a local code-generation interface and a developing training pipeline. Calling it a dependable autonomous debugger would go beyond the evidence.

## 10.2 Model limitations

The vocabulary is byte-based with a short context. The model has only five blocks in its larger configuration and 1.1 million parameters. There is no retrieval, tool use, repository graph, execution feedback, instruction tuning across diverse tasks, learned subword tokenizer, KV cache or beam search. Its ReLU feed-forward layer is intentionally simple.

No learned confidence calibration connects output probabilities to actual repair success. Greedy decoding avoids sampling variance but cannot correct a systematically wrong high-scoring continuation. Syntax-aware constrained decoding could reject some invalid sequences, but valid syntax alone would still not establish behavior.

## 10.3 Data limitations

The early data is extremely repetitive. The later distinct exercises improve variety but remain elementary, short and largely synthetic. The stdlib corpus improves local syntax exposure but is a limited domain, not a representative distribution of user code, tests, frameworks, errors and fixes. It contains repeated boilerplate and related modules across file-level splits.

Mutation checks use a few cases per task. Some bug examples can have alternative intended fixes. Tests stored as JSON may change dictionary key types when read back if keys were non-string; the generator validates source-controlled Python cases before JSON serialization, and no behavioral evaluator currently consumes serialized expected values. A future test runner must use a typed test format or normalize such values rather than assuming JSON preserves every Python object exactly.

The corpus manifest contains local absolute source paths. That is useful provenance inside this local archive, but should be reviewed before any future publication. The initial archival capture was private; this publication uses a separate sanitized history.

## 10.4 Training limitations

The pipeline uses random windows and a small fixed validation sample for pretraining. Neither pipeline has exact transactional pause/resume. A Ctrl+C can interrupt an update and the code saves checkpoints directly rather than via atomic file replacement. The new pipeline records init hashes but not a full RNG state or a continuation ancestry graph.

Dropout is absent. float64 arithmetic is useful for educational gradient checking but uses more memory than float32. Forward passes build backward graph objects even in inference. Per-token generation recomputes all retained context, and layer normalization/attention arrays are not fused or optimized. These are reasonable simplifications for inspection, but limit speed.

## 10.5 Evaluation limitations

The preserved app and evaluator use syntax and reference-text checks. No secure behavioral evaluator exists. The validation split is repeatedly used for checkpoint selection; it is not an untouched final test. Family labels are not a semantic deduplicator. There are no published repair-success confidence intervals or comparisons against baseline methods.

Useful baselines could include identity copying, simple fixed-rule repairs for the known exercise templates and a smaller transformer. If a complex trained model cannot beat identity or a tiny deterministic baseline on the intended benchmark, scaling it is not yet justified. Those baselines are proposed experiments, not included executable tools.

## 10.6 Current tests and what each establishes

| Check | Coverage | Limit |
|---|---|---|
| Directional finite differences | Every parameter family on a tiny model | Not all entries/shapes/inputs |
| Causality | Future-token changes leave earlier logits unchanged | One configured small test |
| Save/load equality | Exact array preservation for a tiny model | Not atomic multi-file recovery |
| Tiny learning task | Loss can fall under updates | Not repair generalization |
| Forced early stopping | Patience logic, initial best and restoration | Not best behavioral model selection |
| Stage transfer | Checkpoint config/weights accepted for fine-tuning | Not trained task competence |
| Metadata/overwrite checks | Source hashes and existing-output refusal | Not full reproducible execution state |
| Artifact hashes | Preserved bytes match capture | Not content truth or licensing review |
| NPZ finite/integrity checks | Readable archive, finite parameter arrays | Not useful weights |

Verification records in records/ document the capture and public-edition boundaries. The original GUI was not manually clicked through again solely for this repository task; no change was made to its implementation. Numerical inference and Tk initialization had been checked earlier, and source-level tests are repeated in the repository.

## 10.7 Ordered development roadmap

### A. Establish a defensible benchmark

Define intended repair domains, give each task a specification and independent tests, freeze a final test set, record model identity in every output, and measure syntax/behavior separately. Include already-correct programs and cases where the system should refuse due to insufficient information. Keep related families and source ancestry together during splitting.

### B. Improve data without inflating row counts

Add genuinely different algorithmic and API tasks, more implementations per task, clearer domain contracts, and feedback-conditioned correction trajectories. Distinguish unique tasks from mutations. Deduplicate normalized ASTs and near copies. If external corpora are introduced, record their licenses, versions, provenance and filters; do not execute imported code in the trusted-template generator.

### C. Strengthen persistence and reproducibility

Implement atomic checkpoint replacement, completed-step bookkeeping, full RNG and schedule state, optimizer restoration in the pipeline, validation configuration persistence and a run ancestry record. Add tests that interrupt at carefully controlled boundaries and verify resumed runs match uninterrupted ones within a defined numerical tolerance.

### D. Build the checker in a real execution boundary

First test the controller on trusted exercises. Then specify and validate filesystem, network, process, time and memory isolation before arbitrary generated execution. Add structured reports and bounded retries. Keep public retry tests separate from final hidden tests. Do not label a timeout-only subprocess as secure sandboxing.

### E. Train on the new protocol

Once task/feedback format is stable, teach it explicitly with matched training and inference serialization. Measure whether longer context or another tokenizer is needed. A feedback loop whose prompt never appeared in training may be unreadable to a tiny model regardless of how sensible the controller is.

### F. Optimize only after measurement

Profile CPU time and memory at actual training sizes. Possible NumPy-compatible changes include avoiding autodiff graphs at inference, reducing redundant buffers, float32 with float64 gradient-test mode and better batch packing. A GPU implementation requires revisiting the stated pure-NumPy constraint; it should not be silently introduced.

## 10.8 Why the next step is not simply "more parameters"

Parameter count raises representational capacity and can raise compute cost. It neither defines a task specification nor supplies missing examples, tests or feedback. The most useful next result would be a controlled experiment demonstrating higher unseen-task success with known data and checkpoints. That can be more informative than a substantially larger model with no reliable evaluation.
