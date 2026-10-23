# Technical book: a GPT debugger from first principles

This book describes the source captured on 23 October 2026, the earlier surviving versions, and the reasoning developed during the project. The equations are derived here against our implementation. The linked primary papers provide background; their published systems and results are not claims about this small model.

## Reading order

1. [Purpose, boundaries and mathematical notation](01-PURPOSE.md)
2. [Transformer forward pass: every operation and tensor shape](02-ARCHITECTURE.md)
3. [Softmax, cross-entropy and complete backpropagation](03-MATHS.md)
4. [Training, Adam/AdamW, pretraining, early stopping and checkpoints](04-TRAINING.md)
5. [Data construction, mutation verification, leakage and evaluation](05-DATA-EVALUATION.md)
6. [Generator and checker: the proposed feedback system](06-FEEDBACK-DESIGN.md)
7. [Windows setup, commands, hardware and troubleshooting](07-OPERATIONS.md)
8. [Project history, recovered versions, observations and corrections](08-HISTORY.md)
9. [Worked examples and code walkthrough](09-WORKED-EXAMPLES.md)
10. [Limitations, verification and development roadmap](10-LIMITATIONS.md)
11. [Glossary and primary references](11-REFERENCES.md)

## Distinguishing kinds of evidence

**Implemented:** behavior visible in the preserved source. **Verified mechanically:** behavior exercised by a test or a file integrity check. **Observed:** a result in a screenshot or saved log. **Inferred:** an interpretation of those observations. **Proposed:** design work which is not present in the executable system.

For example, differentiating attention is implemented and has gradient checks. A screenshot of an invalid factorial answer is observed. Insufficient data is a plausible contributing explanation, but that image alone does not isolate every cause. A separate test-running agent that retries corrections is proposed, not implemented. Keeping these categories separate is essential to understanding what this project has achieved.

## Scope of the archive

This is a research notebook and local repository, not a claim that we recreated ChatGPT. All learned weights began randomly in this project. NumPy performs numerical computation, backed by whatever CPU numerical libraries the installed NumPy distribution uses. Python standard-library facilities provide the GUI, AST parsing, serialization, tests and operating-system interaction. "Pure NumPy" refers to model/training math, not an assertion that every line of the application is a NumPy call.

The original technical guides remain in the root and historical snapshots. When an old guide describes the default context as 192 bytes or lacks resume support, consult its version and this book rather than assuming every description applies simultaneously.
