# NumPy GPT Lab

A small GPT-style transformer built and trained from scratch with **NumPy only for model mathematics**, plus a local Python-debugging app. This is an educational project for understanding transformers, backpropagation and the limits of small training datasets.

**Contributions are welcome.** The project code, original exercises, documentation and project-generated weights are offered under the [MIT license](LICENSE). Bundled Python standard-library corpus material retains its original licensing; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## What works

- Byte-level causal transformer with multi-head attention, residual connections and layer normalization.
- Custom reverse-mode autodiff, stable softmax/cross-entropy, Adam and AdamW.
- Plain-Python pretraining followed by debugging fine-tuning.
- Early stopping, best/last checkpoints and training logs.
- 100 different checked exercises and 162 verified synthetic repair pairs.
- A local Tkinter app that displays suggestions and flags invalid Python syntax.
- A detailed [technical book](TECHNICAL_GUIDE.md), including maths, derivatives, experiments and design reasoning.

**This is not a reliable general-purpose debugger.** Mechanical tests pass, but independent behavioral repair accuracy is not established. The app does not execute suggestions, retry against tests or learn automatically from user interaction. Standard NumPy uses the CPU; this implementation does not use a GPU.

## Install

Use standard Python 3.10+ with Tkinter and Git. On Windows, use a standard Windows Python installation rather than MSYS2 Python unless you are comfortable configuring its dependencies.

```powershell
git clone https://github.com/bobjeferoony/numpy-gpt-lab.git
cd numpy-gpt-lab
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest -v test_gpt.py test_pipeline.py
.\.venv\Scripts\python.exe app.py --model distinct-large.npz
```

On Linux/macOS:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest -v test_gpt.py test_pipeline.py
.venv/bin/python app.py --model distinct-large.npz
```

Linux distributions may package Tkinter separately. The supplied checkpoints are research snapshots, not endorsed working debuggers. `python-base.npz` is partially pretrained plain-Python weights, not a debugging fine-tune. See [checkpoint inventory](records/CHECKPOINTS.json).

## Train

The prepared corpus and verified pairs are in pipeline-data/. To collect a corpus from your own installed Python into a new directory, run `python prepare_pipeline.py --out local-data` and pass `--data local-data` to the trainer. Recorded source paths are relative to the standard library; no local home-directory paths are needed.

```sh
python pipeline_train.py --stage pretrain --out new-base.npz --steps 5000
python pipeline_train.py --stage finetune --init new-base.npz --out new-debugger.npz --steps 3000 --lr 0.0001
python app.py --model new-debugger.npz
```

Run the second command after the first finishes. The defaults give 1,106,560 parameters and a 384-byte context, with CPU training that can take substantial time. Ctrl+C saves last weights and exports the best evaluated weights; wait for saving to finish. `--init` retains weights but resets optimizer state and the schedule. Exact full-state pipeline resume is not implemented.

## Explore and contribute

| Resource | Purpose |
|---|---|
| [Book index](docs/00-INDEX.md) | Read the architecture and mathematical derivations chapter by chapter |
| [All 100 exercises](DISTINCT_PROBLEMS.md) | Inspect buggy code, reference fixes and test cases |
| [Contributing](CONTRIBUTING.md) | Setup, test commands and contribution guidelines |
| [Good first improvements](CONTRIBUTING.md#where-help-is-most-useful) | Data diversity, evaluation, reproducibility and inference performance |
| [Version history](docs/08-HISTORY.md) | Earlier source snapshots and what each stage implemented |
| [Public edition record](records/PUBLICATION.md) | Sanitization boundaries and fresh public history |

Historical snapshots in versions/ are sanitized educational references, not byte-for-byte copies of private archives. They retain older behavior and limitations; use root code for current work. This public repository has fresh history so earlier private-directory details are not present in old commits. The original complete private archive is kept separately.
