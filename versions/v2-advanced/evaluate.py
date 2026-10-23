"""Measure output without executing model-generated code."""
import argparse
import ast
import json
from pathlib import Path
from gpt import GPT

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='advanced.npz')
    ap.add_argument('--data', required=True, help='Separate, unseen JSONL test set')
    ap.add_argument('--limit', type=int, default=10)
    args = ap.parse_args()
    if args.limit < 1:
        ap.error('Limit must be positive')
    model = GPT.load(args.model)
    rows = [json.loads(s) for s in Path(args.data).read_text(encoding='utf-8').splitlines() if s.strip()][:args.limit]
    exact = parses = 0
    for row in rows:
        answer = model.generate(row['buggy'], tokens=min(400, len(row['fixed'].encode('utf-8'))+40))
        try:
            tree = ast.parse(answer)
            parses += int(any(isinstance(n, ast.FunctionDef) for n in tree.body))
        except (SyntaxError, ValueError):
            pass
        exact += int(answer.strip() == row['fixed'].strip())
        print(json.dumps(dict(input=row['buggy'], suggestion=answer)))
    print(f'{len(rows)} examples; {parses} parse as Python functions; {exact} exact reference matches.')
    print('Parsing does not prove correctness. Different valid fixes may fail exact matching.')
