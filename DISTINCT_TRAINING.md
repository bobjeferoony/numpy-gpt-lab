# Train on the distinct exercises

The new distinct.jsonl contains 100 different exercises rather than renamed variants: 20 arithmetic, 15 predicates, 15 sequences, 15 strings, 10 filtering, 10 maps/sets, 10 loops and 5 boundaries. Some share bug patterns or overlap conceptually. They are different tasks, not 100 independent root causes. This is still a small educational dataset.

Each reference fix passes at least two explicit input/output checks, and each buggy version fails at least one. Tests assume the task's intended domain: for example second_distinct requires two distinct numbers, and ordinary average requires a nonempty list. safe_mean separately covers the empty case. These checks do not prove correctness for every input. User code is not executed; only the source-controlled generator templates are checked.

See DISTINCT_PROBLEMS.md for every buggy function, reference fix, test case and split. With the current trainer seed, all 20 arithmetic tasks are validation-only; 80 other tasks are used for training. This is a challenging transfer split, not a random 80/20 row split. Validation participates in monitoring, so it is not an independent final test set.

Your old datasets and model files are preserved. Close the app, open the project terminal and run:

```powershell
.\.venv-win\Scripts\python.exe train.py --data distinct.jsonl --out distinct-large.npz --context 384 --width 128 --layers 5 --steps 5000
```

This starts random weights with the new dataset so the old training data does not contaminate its held-out split. It does not start training automatically. The CPU training may take substantial time. Stop with Ctrl+C and wait for the save confirmation if necessary. To continue an interrupted run:

```powershell
.\.venv-win\Scripts\python.exe train.py --data distinct.jsonl --resume distinct-large.npz --out distinct-large.npz --steps 5000
```

Open the model:

```powershell
.\.venv-win\Scripts\python.exe app.py --model distinct-large.npz
```

Evaluate on held-out exercises:

```powershell
.\.venv-win\Scripts\python.exe evaluate.py --model distinct-large.npz --data distinct-heldout.jsonl --limit 20
```

The evaluator checks syntax and exact reference matches, not behavioral correctness; it never executes model outputs. The JSONL tests are for inspection and generator validation, not automatically run against generated suggestions. Some code is ambiguous without a task description. More variety helps, but these 80 training examples cannot establish general debugging ability, and successful training does not guarantee valid output.

To regenerate and recheck the data later: python make_distinct.py. It overwrites only distinct.jsonl, distinct-heldout.jsonl and DISTINCT_PROBLEMS.md.
