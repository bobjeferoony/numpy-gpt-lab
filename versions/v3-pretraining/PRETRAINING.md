# Python pretraining, then debugging

This workflow trains from random weights using only NumPy for all model math. It uses local Python standard-library source, not pretrained weights or API services. Existing models and datasets are preserved.

## Prepared data

The first preparation run found 560 source files: 8,846,516 UTF-8 training bytes and 1,008,619 validation bytes. A byte is a token for this tokenizer. This is about 9 million training tokens, but each training run samples windows; it does not necessarily see every byte. With batch 2, context 384 and 5000 steps it consumes 3.84 million sampled training token positions, with overlap.

Sources are documented with paths and SHA-256 hashes in pipeline-data/sources.json. PYTHON-LICENSE.txt preserves the installed Python license. Files containing third-party notices retain their comments in the corpus. The pipeline uses installed standard-library .py files, excludes site-packages/tests and deduplicates exact files. Whole files are assigned to training or validation; similar modules can still share code. This is an initial syntax corpus, not an independent debugging benchmark. Files are parsed, never imported or executed during corpus collection.

Synthetic debugging data has 162 verified pairs derived from the 100 distinct exercises. Each original fixed function passes its explicit tests. Each mutation must fail a test, either by wrong output or exception; surviving/equivalent mutations are discarded. Operator mutations use ASTs. These are related examples, not 162 independent tasks. Arbitrary downloaded/user code is not executed: mutation checks are confined to our source-controlled exercises, which use finite loops and built-in operations. The generator is not a general sandbox.

Arithmetic families are held out of debugging training, including every mutation. The pretraining corpus may contain similar concepts. This measures transfer of debugging training, not complete conceptual novelty. The tests document intended domains; they do not prove full correctness.

## Run

Open PowerShell in the project folder. Preparation has already been done. To regenerate in a new directory, use prepare_pipeline.py --out another-directory; it refuses to overwrite an existing directory.

Pretrain the 1,106,560-parameter GPT on Python:

```powershell
.\.venv-win\Scripts\python.exe pipeline_train.py --stage pretrain --out python-base.npz --steps 5000
```

Then fine-tune the resulting best weights on debugging:

```powershell
.\.venv-win\Scripts\python.exe pipeline_train.py --stage finetune --init python-base.npz --out python-debugger.npz --steps 3000 --lr 0.0001
```

Open the app:

```powershell
.\.venv-win\Scripts\python.exe app.py --model python-debugger.npz
```

You must finish pretraining before fine-tuning. This new pipeline does not start a long run automatically. Your CPU does the training; NumPy does not use your NVIDIA GPU. Training quality must be measured; this amount of code and these few debugging tasks cannot guarantee a useful debugger.

## Saved results and stopping

- The requested output, such as python-base.npz, contains the **best validation weights**, restored at the end.
- .best.npz contains best weights and optimizer state.
- .last.npz contains last weights and optimizer state, even when early stopping restores an earlier model.
- .history.jsonl records losses, learning rate and best loss. .run.json records configuration, arguments and dataset hashes.
- Ctrl+C saves the last state and writes the best weights to the requested output. Let it finish saving. A hard terminal/process kill does not execute this cleanup; use the latest saved best/last checkpoint instead.
- Early stopping triggers after 8 validation evaluations without a new best. Validation runs every 50 steps by default. Both values are configurable with --patience and --eval-every.
- --init loads weights only and starts fresh AdamW state and schedule. It intentionally resets the optimizer between pretraining and fine-tuning. This trainer has no exact optimizer-resume flag. To continue interrupted training, use --init RUN.last.npz with a new --out; this continues weights with a reset optimizer. Do not pass these stage checkpoints to the legacy train.py unless you intend its different training workflow.
- Existing output names are refused to avoid overwriting prior runs. Choose a new --out for a repeat experiment.

AdamW adds decoupled weight decay of 0.01 to matrix weights except embeddings; normalization and biases are excluded. The trainer retains gradient clipping, warmup and cosine decay. Dropout has not been added. Pretraining scores every byte. Fine-tuning scores only the reference fix after the BUG/FIX prompt. The app uses the same prompt format and greedy decoding.

## Evaluation and limitations

For syntax and exact-reference matches on the held-out debugging exercises:

```powershell
.\.venv-win\Scripts\python.exe evaluate.py --model python-debugger.npz --data pipeline-data/bugs-validation.jsonl --limit 1000
```

These checks do not execute model-generated code and do not establish behavior. A secure behavioral evaluator is not included. Training-generation test checks must not be confused with tests of the trained model. Validation is used for early stopping, so you still need independent tasks for a final benchmark. More training data and varied bug families remain necessary.

Pipeline smoke checks cover pretraining, transfer to fine-tuning, early stopping, best-weight restoration, metadata and overwrite protection. They establish mechanics, not debugging quality. Run all tests with python -m unittest -v test_gpt.py test_pipeline.py using the project environment.
