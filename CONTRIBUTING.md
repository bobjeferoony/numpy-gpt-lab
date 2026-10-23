# Contributing

Thanks for helping improve an inspectable NumPy transformer. Open an issue to discuss a bug or design change, fork the repository, create a focused branch, and submit a pull request. Explain the problem, intended behavior, implementation and validation.

Contributions to original project code and documentation are under the project's MIT license. Preserve third-party notices. Only contribute datasets and code you have permission to share.

## Development setup

Create a Python virtual environment and install requirements.txt as shown in the README. The only third-party dependency for model mathematics is NumPy. Keep model, gradient and optimizer computations in NumPy; discuss a different backend before proposing it as a default dependency.

Run these from the repository root using your environment's Python:

```sh
python -m unittest -v test_gpt.py test_pipeline.py
python tools/audit_public.py
```

For changes to autodiff or the model, retain meaningful finite-difference and causal-mask checks. For training-control changes, test best/last restoration and interruption/resume semantics. A test should establish behavior rather than copy an implementation detail.

## Where help is most useful

1. **More genuinely different verified debugging tasks.** Give each task a clear input domain, specification, independent reference/tests and family label. Keep related mutations and renamed variants together in one split.
2. **Better evaluation.** Separate syntax, exact-reference matching and behavioral success. Do not execute arbitrary generated Python in the main app or call a timeout-only subprocess a security sandbox. Propose the execution boundary first.
3. **Exact resumability.** Save RNG, optimizer, completed-step, schedule, validation and early-stopping state; use atomic checkpoint replacement and test interrupted versus uninterrupted runs.
4. **Faster NumPy inference.** Avoid backward-graph construction when gradients are unnecessary, profile allocations and consider an explicit precision policy. Preserve reference math and gradient tests.
5. **Clearer model identity in the app.** Display checkpoint filename, architecture and training stage so comparisons use the intended weights.
6. **Feedback-conditioned training.** Discuss a stable specification/test-feedback prompt format before changing serialization. Training and inference must agree, and context limits must be measured.

## Reporting results

Include the source commit, data provenance and hashes, unique task/family counts, split policy, model configuration, steps, optimizer/schedule, validation selection and untouched test results. Distinguish a pipeline smoke test from trained debugging competence. Compare against simple baselines, not just a lower training loss.

Do not claim generated code is correct merely because it parses. Do not count renamed rows as independent problems. Do not present validation used for checkpoint selection as an untouched final benchmark.

## Keeping the repository portable

Use relative paths in documentation, scripts and committed metadata. Do not commit credentials, private code, screenshots containing personal details, absolute home-directory paths, virtual environments or raw private archives. New run metadata and local-data/ are ignored by default; inspect staged files before pushing. Existing tracked example checkpoints are deliberate research snapshots.

Historical snapshots under versions/ document earlier behavior. Make active fixes in root source; update a snapshot only to repair a publication/privacy issue and disclose that change. After changing chapter documentation, run `python tools/build_book.py` to rebuild TECHNICAL_GUIDE.md.
