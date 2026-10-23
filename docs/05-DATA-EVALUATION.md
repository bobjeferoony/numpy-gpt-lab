# 5. Data, verification and measuring real progress

## 5.1 The original sixteen pairs

examples.jsonl contains short buggy/fixed Python snippets for arithmetic, list indexing, simple predicates and related one-line repairs. There are 16 rows. The first trainer randomly held out roughly one fifth of rows, leaving 13 for training and 3 for validation. It was a mechanical demonstration of a training loop, not a sufficient language-learning corpus.

A model exposed only to those pairs is trying to learn Python syntax, prompt formatting, variable naming, copying and bug repair simultaneously from very few tokens. Even successfully reproducing those answers would establish memorization or narrow pattern learning, not arbitrary debugging.

## 5.2 The 640-row dataset and why its size was misleading

make_data.py defines 16 templates and generates 40 function-name variants of each. The variants preserve the same parameter names, body and repair, changing names such as `factorial_12`. There are 640 rows but still only 16 underlying exercises. The archive retains all of them rather than retroactively relabeling the old experiment as broader data.

The upgraded legacy trainer sorts family labels, shuffles them with NumPy seed 123, then holds out max(1, floor(family_count/5)) families. For advanced.jsonl the held-out families are contains, last and multiply. The result is 520 training pairs and 120 validation pairs.

Factorial is in the training split for that dataset. The earlier speculation that it might have been held out was resolved by reproducing the split. A screenshot of a malformed factorial suggestion therefore demonstrates a failure even on an underlying familiar template, assuming the app loaded the intended large checkpoint. The screenshot itself does not prove which filename was loaded because the original GUI does not display checkpoint identity.

## 5.3 The 100 different exercises

make_distinct.py replaces pure renaming with different short tasks. Categories and counts:

| Family | Tasks |
|---|---:|
| Arithmetic | 20 |
| Predicates | 15 |
| Sequences | 15 |
| Strings | 15 |
| Filtering | 10 |
| Maps and sets | 10 |
| Loops | 10 |
| Boundary cases | 5 |

The detailed catalog is DISTINCT_PROBLEMS.md. Each task has at least two explicit cases. Fixed functions must return expected outputs on all of them. The intentionally buggy version must give a different result or throw an exception on at least one. All code executed by this generator is source-controlled template code, not arbitrary imported user snippets.

The current split holds out the arithmetic family: 80 tasks train, 20 validate. That is a deliberately difficult family-transfer split. It may be less representative of the same-family repairs a user first tries, but it avoids some superficial template leakage. Some concepts still overlap, such as multiply as arithmetic and product as a loop, so it does not ensure complete conceptual separation.

"Distinct" is a practical description of different exercises, not a formal count of independent bug mechanisms. Positive/negative predicates and inclusive/exclusive comparisons share underlying operations. This is more useful variety than renamed duplicates but still small for training language competence from scratch.

## 5.4 Parameter count versus data count

1,106,560 / 80 = 13,832 parameters per training example. This ratio illustrates how little task variety there is, but it is not a universal overfitting threshold. Examples differ in length, redundancy and information content; architectures and training methods differ. There is no rule that one example supports a fixed number of parameters.

A better report includes unique tasks, distinct bug families, scored tokens, source diversity, exact/near duplicates, repeated exposures, held-out repair accuracy, and confidence intervals. A larger model can memorize a small dataset, but the observed malformed outputs show that even perfect memorization was not established for this project.

## 5.5 Preparing a local Python corpus

prepare_pipeline.py finds the installed interpreter's standard-library source directory using sysconfig. It scans .py files, excludes site-packages, tests/test, idlelib, ensurepip and caches, and skips files over 500,000 bytes. It reads with Python's tokenize.open to honor encoding declarations, parses the text with ast.parse and skips invalid/undecodable sources. It does not import or execute corpus modules.

Each accepted text is encoded as UTF-8 and exact duplicate normalized byte contents are removed by SHA-256. For deterministic whole-file splitting, the relative file path is hashed; a modulus rule assigns roughly 10% of files to validation. A file remains entirely in one split. The concatenated bytes plus separating newlines are saved in separate .bin files.

The captured corpus has 560 files, 8,846,516 training bytes and 1,008,619 validation bytes. A sources.json manifest records source paths, original byte hashes, assigned split and normalized UTF-8 size. The installed Python license is copied to PYTHON-LICENSE.txt; comments in the corpus retain applicable source notices. This is a local source archive, not a download of The Stack, CodeSearchNet or arbitrary GitHub repositories. Those sources were discussed but not ingested.

File-based splitting does not eliminate duplication across sibling modules, copied idioms or repeated license headers. Content normalization/near-duplicate filtering and package-level splits could improve it. Very common boilerplate can lower next-byte loss without much debugging benefit. The file path hash is reproducible under the same path names and content collection rules, not a semantic independence guarantee.

## 5.6 Verified synthetic mutation pipeline

The mutation generator begins with fixed functions from the source-controlled 100 exercises. It collects AST operators and swaps selected kinds, for example addition/subtraction, true/floor division, equality/inequality, strict/non-strict comparisons, membership/non-membership and Boolean and/or. It also retains the original deliberately buggy version.

For each candidate:

1. Parse/construct candidate Python from the trusted template.
2. Define the candidate in an isolated Python namespace within the generator process.
3. Run the task's known input/output cases.
4. Retain the candidate only if at least one case fails or raises an exception.
5. Deduplicate exact buggy/fixed pairs.
6. Require the serialized pair to fit the 384-byte training context.
7. Assign all related mutations to the same family split.

The result is 162 verified pairs, not tens of thousands. The generation stage does not magically create new task specifications. A candidate's failure establishes that a particular mutation is caught by at least one finite case; it does not establish that the fixed reference is universally correct.

These executions are safe only under the narrow assumption that our own short templates and allowed mutations remain understood and finite. They run in-process and are not a security sandbox. This script must not be adapted to execute untrusted downloaded code simply by removing that restriction. A timeout on a subprocess alone would also not constitute security isolation.

AST operator objects can be shared by Python's parsed tree representation. The mutation code locates the target node by identity; a copied shared operator can affect more than one occurrence. Every retained result still undergoes tests, but we should not claim every mutation is exactly one isolated textual edit. That is a useful future refinement for cleaner causal experiments.

## 5.7 Splitting before augmentation

If one task's original function is in training and its mutated/renamed counterpart is in validation, the apparent validation improvement can come from nearly identical patterns. Splitting by family before assigning mutations keeps those near relatives together. The pipeline explicitly rejects overlapping family labels between debugging training and validation files.

That guard checks labels, not arbitrary semantic code similarity. A mislabeled duplicate could pass it. A robust pipeline should canonicalize ASTs, strip irrelevant identifiers where appropriate, compare normalized source, trace repository ancestry, and separate related tasks before any augmentation. A final benchmark should also avoid examples included in pretraining when possible or disclose such overlap.

## 5.8 Loss, syntax, exact match and behavior

Each metric answers a different question:

| Metric | Evidence it supplies | What it misses |
|---|---|---|
| Teacher-forced byte loss | Probability assigned to reference continuations | Whether free generation succeeds |
| Python parsing | Output conforms to Python grammar | Runtime behavior, intent, safety |
| Presence of a top-level function | A minimal expected output shape | Whether it is the requested function |
| Exact reference match | Reproduction of one chosen target | Alternative correct repairs |
| Passing fixed tests | Correct results on tested cases | Untested inputs and hidden side effects |
| Independent hidden tests | Stronger evidence on unseen cases | Still not a universal proof |

evaluate.py uses the existing generator and reports parsing plus exact-match counts. It never executes the generated output. A function with an unrelated extra expression could pass its shape check; a correct alternate implementation could fail exact matching. That evaluator is a diagnostic, not a semantic benchmark.

The app's AST check similarly asks whether any top-level FunctionDef or AsyncFunctionDef exists after parsing. It does not validate the requested name, signature, return behavior or absence of additional code. Its label explicitly says parsing is not correctness.

## 5.9 Example ambiguity and domain contracts

`second_largest([2,7,7,4])` can reasonably mean 7 if duplicates count, or 4 if "distinct" is required. Without intent in the prompt, the model cannot infer a uniquely correct repair from code alone. The corresponding exercise assumes at least two distinct elements.

Other contracts matter too: ordinary average assumes a nonempty list; first/last assume nonempty input; divide assumes nonzero denominator. Separate safe variants cover empty/zero cases. A fix should not silently impose a new contract unless the task calls for it. Future data should include a task description and tests as explicit prompt fields, and generation must use precisely that same format.

## 5.10 Statistical uncertainty

With n independent tasks and k successes, an observed success rate k/n is only an estimate. A rough binomial standard error is sqrt(p*(1-p)/n), though this normal approximation behaves poorly near zero/one and for tiny samples. With n=20 and p=0.5 it is about 0.112, meaning substantial uncertainty. Exact or Wilson intervals are better for reporting small counts.

Mutated siblings are correlated, so treating 162 pairs as 162 independent task draws can underestimate uncertainty. Report both task-level and mutation-level outcomes, and bootstrap by task/family rather than blindly by row if estimating intervals.

Validation used to choose a checkpoint is no longer an untouched final test. Keep a final set with separately written requirements, unseen families or independently sourced tasks, and frozen tests. Do not repeatedly add failed benchmark examples to training while continuing to call that benchmark unseen.

## 5.11 What the current evidence actually says

Gradient, causality, serialization, learning and pipeline-control tests pass. The original app screenshots show malformed output on factorial and an unseen second-distinct task. The captured plain-Python pretraining loss fell from approximately uniform prediction to 2.4919 on its sampled validation windows by step 400. That is evidence of improved byte modeling on this corpus, not evidence that the model now repairs code correctly.

There is no saved independent behavioral benchmark showing general debugging success. The next meaningful experiment is to complete/revisit pretraining, fine-tune with a documented split, and assess generated repairs on independent tasks using safe behavior checks. Increasing parameter count or adding more renamed rows alone cannot substitute for that evidence.
