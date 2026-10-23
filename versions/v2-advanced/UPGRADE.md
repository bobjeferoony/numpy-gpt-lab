# Advanced training setup

All model math remains pure NumPy, with random initial weights. Your existing model.npz is preserved. This upgrade improves the machinery; it does not supply a trained, competent debugger.

Close the app, open PowerShell in this folder, then run:

```powershell
.\.venv-win\Scripts\python.exe make_data.py
.\.venv-win\Scripts\python.exe train.py --data advanced.jsonl --out advanced.npz --context 384 --width 64 --layers 3 --steps 5000
```

This larger model runs on the CPU and may take substantial time to train. Stop with Ctrl+C to save. Continue for another 5000 steps with:

```powershell
.\.venv-win\Scripts\python.exe train.py --data advanced.jsonl --resume advanced.npz --out advanced.npz --steps 5000
```

Launch after training:

```powershell
.\.venv-win\Scripts\python.exe app.py --model advanced.npz
```

## Changes

- Default training context is now 384 bytes. Loading an old model keeps that checkpoint's original context limit.
- Batch padding is trimmed, reducing wasted attention computation.
- Learning rate uses warmup and cosine decay; gradient clipping remains enabled.
- Checkpoints include Adam momentum, variance and step count. Resume restores them. Each invocation restarts the learning-rate schedule and sampling sequence; this is not an exact random-stream continuation. Old weight-only checkpoints start fresh Adam state.
- Validation holds out whole bug families, keeping renamed variants of each template together.
- The generator creates 640 examples across 16 templates, including factorial. Fixed templates are checked against independent reference functions; buggy templates must fail at least one case. Only our source-controlled templates are executed, never user datasets.
- The app flags output that fails to parse as a Python function. Valid syntax still does not prove correctness. It never executes suggestions and generates at most 300 bytes.
- evaluate.py reports syntax and exact-reference matches without executing generated code.

## Limits

Most of the 640 examples are renamings, not independent debugging problems. Some templates overlap conceptually; the checks cover specified cases rather than all inputs. More varied, verified examples and task intent are needed for broad ability. More parameters or training steps alone will not solve this.

Create your own separate JSONL test file with buggy/fixed pairs from templates absent from training. Evaluate with:

```powershell
.\.venv-win\Scripts\python.exe evaluate.py --model advanced.npz --data my_unseen_tests.jsonl
```

That test filename is a placeholder, not an included file. The evaluator does not detect training overlap. Parsing can accept incorrect code, and exact matching can reject alternative correct fixes.

The included advanced-smoke.npz is only a four-step pipeline test, not a debugger. It scored zero valid functions and zero exact fixes on two demonstration inputs. No long training run was performed during this upgrade. Gradient, causality, save/load and learning tests pass; new training, optimizer resume, evaluation and source compilation were also checked.
