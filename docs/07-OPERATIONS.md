# 7. Practical operation and troubleshooting

## 7.1 Repository versus original working folder

Clone the public repository and open a terminal in its root. All example paths are relative to that root. Keep new experiments in a separate branch or under distinct output filenames; the historical snapshots under versions/ are references rather than active development folders.

The archive includes individual stable copies of saved files. It does not pause/kill training, and it cannot guarantee atomic consistency among multiple files written by a simultaneously active process. The saved pretraining files in this capture indicate a pause at step 422 with the best evaluated checkpoint at 400.

## 7.2 Opening the terminal on Windows

Open File Explorer. Paste the repository's full folder path into the address bar and press Enter. Right-click empty space in the folder and choose Open in Terminal. Confirm the prompt ends with the repository folder. A terminal's current directory matters because scripts use relative paths such as pipeline-data and model.npz.

To run a path containing spaces in PowerShell, put it in quotes and use the call operator `&`. Backslashes in the command examples are normal Windows path separators. Copy only the command, not the surrounding Markdown fences or the `PS ...>` prompt.

## 7.3 The MSYS2 problem encountered here

The bare command `python` resolved to an MSYS2 Python executable. Its virtual environment used `.venv/bin/python.exe`, whereas the initial instructions assumed `.venv/Scripts/python.exe`. Switching paths fixed the missing executable but did not resolve NumPy's build/dependency failure in that environment.

Using a standard Windows Python installation resolved the setup issue; the original development environment used Python 3.13.3. That environment could install the prebuilt NumPy package. The `py` launcher reported no registered Pythons even though that executable existed, so `py -0p` alone was not a complete inventory.

Virtual environments are not committed. Create a fresh one in the repository:

```powershell
python -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
.\.venv-win\Scripts\python.exe -m unittest -v test_gpt.py test_pipeline.py
```

On another machine replace the interpreter path with its standard Python installation. Tkinter is normally installed with standard Windows Python; if absent, install the appropriate Python Tcl/Tk component. The bundled runtime used for early numerical tests had NumPy but lacked a usable init.tcl, while the system Python's Tkinter initialized correctly. Thus one interpreter working for array tests did not automatically validate another interpreter's GUI dependencies.

## 7.4 Legacy model training

To reproduce the original style of run with the latest legacy trainer:

```powershell
.\.venv-win\Scripts\python.exe train.py --data advanced.jsonl --out legacy-new.npz --context 384 --width 128 --layers 5 --steps 5000
```

For distinct examples substitute distinct.jsonl. This command starts random weights unless --resume is supplied. The current legacy trainer has family splitting, warmup/decay and resume; the first archive version did not have all those features. Use the archived source if reproducing that earlier code rather than mixing old instructions with current behavior.

Resume legacy state:

```powershell
.\.venv-win\Scripts\python.exe train.py --data distinct.jsonl --resume distinct-large.npz --out distinct-large-continued.npz --steps 1000
```

This restores optimizer arrays if present, but not an exact RNG/schedule stream. Unlike the pipeline trainer, legacy train.py can overwrite its chosen output path, so a new output name is prudent.

## 7.5 Two-stage pipeline

The corpus is already preserved. Regeneration is unnecessary unless intentionally creating a new dataset. To regenerate into a new directory:

```powershell
.\.venv-win\Scripts\python.exe prepare_pipeline.py --out pipeline-data-new
```

The preparation command refuses an existing directory. On another Python version its source collection and file sizes can differ; a new manifest identifies that dataset. Do not silently replace the archived corpus while expecting its recorded SHA-256 hashes to remain valid.

Start fresh pretraining:

```powershell
.\.venv-win\Scripts\python.exe pipeline_train.py --stage pretrain --out python-base-new.npz --steps 5000
```

Continue the captured paused weights, with a fresh optimizer and schedule:

```powershell
.\.venv-win\Scripts\python.exe pipeline_train.py --stage pretrain --init python-base.last.npz --out python-base-continued.npz --steps 5000
```

Then, after that run completes or saves cleanly, start fine-tuning:

```powershell
.\.venv-win\Scripts\python.exe pipeline_train.py --stage finetune --init python-base-continued.npz --out python-debugger.npz --steps 3000 --lr 0.0001
```

The second command needs the first command's completed exported file. The pipeline requires --init for fine-tuning. When --init is supplied, the loaded checkpoint's architecture and context take precedence over width/layers/context CLI defaults. It cannot enlarge an existing checkpoint merely by supplying --width.

## 7.6 Stopping for the night

Press Ctrl+C once. Wait for saving to complete and the shell prompt to return. The new pipeline saves last state and exports best weights. Do not assume closing the terminal performs the same cleanup. A process kill, disk-full error or power failure can interrupt saving.

The pipeline has no exact --resume. To continue weights later, use --init with a new output name. If preserving a fully evaluated point is more important than retaining every recent update, initialize from .best.npz rather than a potentially interrupted .last.npz. These choices trade recent progress against confidence in checkpoint consistency.

## 7.7 Launching and selecting models

```powershell
.\.venv-win\Scripts\python.exe app.py --model distinct-large.npz
```

Or select python-debugger.npz only after producing it. `python-base.npz` is not the debugging fine-tune. `smoke.npz` and advanced-smoke.npz are pipeline demonstrations, not usable debuggers. The app defaults to model.npz if --model is omitted, so always pass the intended checkpoint when comparing versions.

The current title/status do not display a checksum or full architecture. A future UI should show model filename, parameter count, training stage and a short hash to avoid unknowingly testing the wrong weights.

## 7.8 Input length and generation limits

The original checkpoint's context is 192 bytes; later ones use 384. Model context is saved in config. The app builds BUG/FIX markers around input, so the code alone cannot fill the whole limit. Unicode consumes multiple UTF-8 bytes.

Generation emits at most 300 bytes from the app. If EOS is not reached, output may end mid-function. Once the prompt plus generated bytes exceeds context, only the latest context bytes are passed to the model, and learned positions restart at zero for that window. This can discard the original task and break syntax. There is no KV cache; every generated token recomputes the forward pass over the retained context. This is simple but slow.

## 7.9 Interpreting output messages

"Invalid or missing Python function" means parsing failed or no expected top-level function node was present. It is not evidence that more steps alone will solve the issue. "Python parses; correctness has not been checked" is only a syntax statement. The app never executes suggestions and never automatically learns from a user correction.

For a first mechanical check, use a familiar short template and compare its entire output against the expected function. For generalization, use a separate task specification and unseen tests. Do not keep adding a failed evaluation example to training and then call its later success generalization.

## 7.10 CPU, RAM and runtime

Standard NumPy here uses CPU matrix multiplication. The RTX 3050 Ti is idle for this workload. Sixteen GB of RAM is sufficient for the demonstrated small-batch configurations, but longer context and parallel training processes can multiply memory use. Running several copies does not pool learning; it creates separate optimizers and can slow every process through resource contention.

Threading behavior depends on NumPy's BLAS build and environment. A wall-clock estimate needs a local benchmark at the actual width, depth, context and batch size. The archive does not invent a universal training duration. A tiny-model smoke run verifies code paths, not the speed of the million-parameter configuration.

## 7.11 Inspecting history and model files

```powershell
git log --oneline --decorate
git tag --list
git status
.\.venv-win\Scripts\python.exe tools/audit_public.py
```

The records/CHECKPOINTS.json table identifies architecture, step when available, optimizer-state presence and finite parameter checks at capture time. tools/audit_public.py checks the staged/public file set for obvious local paths and credential patterns; it is not a model-quality test.

Do not regenerate derived data in-place and expect archive verification to stay clean. Work on a new branch or new filenames, commit intentional changes, and keep the preservation records as the original baseline.
