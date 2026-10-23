# 8. History, versions and corrections

## 8.1 What this history is and is not

Development occurred over the conversation of 22â€“23 October 2026 in the Europe/London timezone. No pre-existing project Git history was found. This repository was created afterwards to preserve the available evidence. Its commit dates describe archival work, not the exact times each feature was first implemented.

The original complete archive remains private. This public edition starts new Git history to avoid exposing local paths in old commits. Historical source snapshots have been sanitized; they are not claimed to be byte-for-byte original releases. Screenshots and original release ZIPs are excluded.

## 8.2 Public historical source snapshots

| Snapshot | Location | Stage |
|---|---|---|
| Original | versions/v1-original | Small GPT, autodiff, 16 repair pairs, basic trainer and app |
| Advanced | versions/v2-advanced | Family validation, longer context, optimizer restoration and 640 renamed pairs |
| Pretraining | versions/v3-pretraining | Distinct tasks, local Python corpus preparation and two-stage training |
| Current public edition | Root | Current implementation, portable source metadata, research checkpoints and contribution documentation |

The public-v1-source, public-v2-source and public-v3-source tags identify sanitized snapshots. public-initial identifies the complete public edition. These commits are publication/recovery stages, not fabricated original development timestamps. Active development belongs in root source. The private original history remains separate.

## 8.3 Initial scope and hardware

The user first requested a free local LLM in an app, then clarified that existing pretrained models were not the desired answer: the model should be built from scratch. Its purpose narrowed to debugging code. Hardware was stated as 16 GB RAM and a laptop RTX 3050 Ti. The user required pure NumPy, and Python became the first debugging language.

That constraint led to a CPU numerical implementation, a small byte vocabulary, a small decoder-only transformer, custom autodiff, Adam, JSONL repair pairs and a standard-library Tkinter GUI. The first implementation used width 48, two blocks and context 192. It had 90,528 learned scalars, not millions.

## 8.4 Original tests and smoke evidence

The original tests checked directional finite-difference gradients across parameter families, causal masking, checkpoint round-trip and decreasing loss on a tiny learning task. All passed. A ten-step width-16/one-block smoke run decreased training loss from roughly 5.57 to 5.32. This demonstrated a working numerical pipeline, not debugging competence.

The early runtime had NumPy 2.3.5 available for numerical checks, but its bundled Tcl setup failed when creating a Tk window. System Python's Tkinter initialized. Later user setup found MSYS2 first on PATH, causing environment-layout and package-build issues, until standard Python 3.13.3 was used for .venv-win.

## 8.5 User examples and observed failures

Examples proposed during testing included maximum's reversed comparison, binary search bounds that can loop forever, factorial's exclusive range bound, and second-largest distinct value with duplicates. The binary-search example exceeded the tiny input budget and was not added as an implemented training family. The factorial example fit.

The first factorial screenshot showed garbled text rather than a Python correction. Another screenshot on second-largest showed an invalid return expression. A later larger-model screenshot reproduced much of factorial but generated `for i in ran range(1, 1):`, still invalid. These are observations of failure, not a quantified benchmark.

The private development screenshots showed UI output but did not embed a reliable checkpoint identity; they are omitted from this public edition. The intended launch commands selected advanced.npz or large.npz, but we should not infer an unrecorded model hash from the screenshot alone.

## 8.6 Advanced trainer and 640-row data

The first upgrade added a 384-byte default trainer context, family-based validation, dynamic padding trimming, learning-rate warmup/decay, restoreable Adam arrays in legacy checkpoints, an evaluator and AST-based app output warnings. The generated dataset contained 16 task templates with 40 function-name variants each.

The limitation was repeatedly important: 640 rows were not 640 independent debugging problems. Actual held-out families were contains, last and multiply. Factorial was training data. A four-step advanced smoke run was preserved only as a pipeline test.

## 8.7 Million-parameter model and distinct examples

The user asked about around one million parameters. Width 128, five blocks, context 384 and four heads gives exactly 1,106,560. A saved large.npz reached step 5000. The user then asked for more distinct tasks, leading to 100 different exercises and an 80/20 family split. A separate distinct-large.npz also reached step 5000.

An earlier discussion suggested a random 160/20/20 split for a possible 200-example dataset. That was a proposal only. No 200-example dataset was created. The actual 100-task dataset has 80 training and 20 validation tasks, with no untouched independent final test set.

The user supplied a useful critique emphasizing more data, pretraining, best-checkpoint saving, regularization and better evaluation. We accepted that direction, with corrections: warmup/cosine decay, clipping, greedy decoding and matching prompt format were already present; noisy small-batch losses did not alone prove an excessive learning rate; and no evidence supported asserting that step 200 was universally best.

## 8.8 Pretraining pipeline and recorded progress

The next implementation collected local Python standard-library source rather than downloading a public dataset. It produced 8,846,516 training bytes and 1,008,619 validation bytes across 560 files. Mutation filtering produced 162 verified repair pairs from the 100 original tasks. The trainer added AdamW, fixed-window pretraining validation, best/last checkpoints, early stopping, stage transfer, history logs and SHA-256 source metadata.

The current archived python-base run contains:

| Evaluated step | Validation loss |
|---:|---:|
| 50 | 4.27025658272502 |
| 100 | 3.204874046639049 |
| 150 | 2.8051645243379184 |
| 200 | 2.6708151883942786 |
| 250 | 2.585471905160796 |
| 300 | 2.591397254488756 |
| 350 | 2.5247563731963276 |
| 400 | 2.4919140802352753 |

The last checkpoint reports step 422; best reports 400. The best weights-only export is also preserved. These are saved-file observations, not invented completion of the requested 5000 steps. No new-pipeline debugging fine-tune is present among the real large checkpoints. The small pipeline-finetune-check files are tiny integration smoke runs.

## 8.9 Proposed checker architecture

The user asked whether softmax checks an entire generated solution against what the model learned, and proposed a generator plus feedback/checker unit. We distinguished output probability conversion from correctness checking, and autoregressive continuation from parameter updates. The user supplied an encoder-decoder transformer diagram with a GPT-style orange outline. We explained the decoder-only interpretation, residual arrows and cross-attention difference.

The generator/checker design remains a proposal. Only syntax parsing is implemented in the app. No behavioral sandbox, test-driven retry controller, online correction learning or reinforcement-learning optimizer has been created. The book now describes the intended controller interface and its acceptance criteria without representing those as working code.

## 8.10 Corrections to earlier explanations

1. The first setup instructions assumed standard Windows Python; the installed PATH selected MSYS2. Both the folder layout and package installation issue were subsequently diagnosed.
2. Completing a training run does not establish that the model can handle a familiar task, much less an unseen one. Earlier suggested test examples should be read as probes, not guaranteed supported inputs.
3. Family exclusion was initially uncertain for factorial. Reproduction confirms factorial was included for advanced.jsonl.
4. More parameters and more rows did not establish better generalization. Renamed variants were insufficient diversity; the later dataset improved variety but remained small.
5. The pipeline's pause continuation retains weights but resets optimizer/schedule. It is not exact pause/resume despite saving optimizer arrays in best/last NPZ files.
6. Pretraining loss is a language-modeling measurement, not code-repair success. The captured improvements should not be presented as a solved debugger.
7. The diagram's residual connections are not a solution-checking feedback mechanism.

## 8.11 Status matrix

| Feature | Status |
|---|---|
| NumPy transformer and custom autodiff | Implemented; mechanical tests pass |
| Byte tokenizer and EOS | Implemented |
| Local Tkinter app | Implemented; numerical inference and Tk initialization checked |
| Syntax warning in app | Implemented |
| Legacy optimizer restoration | Implemented; RNG/schedule reset |
| Plain Python pretraining | Implemented; partially trained large run preserved |
| Debugging fine-tuning stage | Implemented; tiny integration tests pass |
| Best-checkpoint restoration/early stopping | Implemented; forced test passes |
| AdamW | Implemented in new pipeline |
| Dataset generation and finite test verification | Implemented for trusted templates |
| Arbitrary downloaded-code mutation | Not implemented |
| Dropout | Discussed, not implemented |
| GPU acceleration | Not implemented under standard NumPy |
| Learned subword tokenizer | Not implemented |
| Exact full-state pipeline resume | Not implemented |
| Behavioral sandbox and retry checker | Proposed only |
| Automatic learning from user feedback | Not implemented |
| Independent code-repair benchmark success | Not established |
| Private GitHub backup | Original archive retained separately; see records/PUBLICATION.md |
