"""Prepare local stdlib pretraining and tested mutations of trusted exercises."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sysconfig
import sys
from make_distinct import build
from train import encode


def digest(b):
    return hashlib.sha256(b).hexdigest()


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    root = Path(sysconfig.get_path('stdlib'))
    manifest = []
    train, valid = [], []
    seen = set()
    for path in sorted(root.rglob('*.py')):
        relative = path.relative_to(root)
        if any(p in {'site-packages', 'test', 'tests', 'idlelib', 'ensurepip', '__pycache__'} for p in relative.parts):
            continue
        if path.stat().st_size > 500_000:
            continue
        raw = path.read_bytes()
        try:
            import tokenize
            with tokenize.open(path) as f:
                text = f.read()
            ast.parse(text)
        except (SyntaxError, UnicodeError):
            continue
        data = text.encode('utf-8')
        key = digest(data)
        if key in seen:
            continue
        seen.add(key)
        # Whole files stay together; closely related modules can still overlap.
        split = 'validation' if int(digest(str(relative).encode())[:8],16)%10 == 0 else 'training'
        (valid if split=='validation' else train).extend(data+b'\n\n')
        manifest.append(dict(source=relative.as_posix(), sha256=digest(raw), split=split, bytes=len(data)))
    if not train or not valid:
        raise ValueError('Not enough standard-library source files found')
    (out/'python-train.bin').write_bytes(bytes(train))
    (out/'python-validation.bin').write_bytes(bytes(valid))
    license_path = Path(sys.base_prefix)/'LICENSE.txt'
    if license_path.exists():
        (out/'PYTHON-LICENSE.txt').write_bytes(license_path.read_bytes())
    (out/'sources.json').write_text(json.dumps(dict(python=sys.version, files=manifest),indent=2),encoding='utf-8')

    rows, unique = [], set()
    swaps = {ast.Add:ast.Sub, ast.Sub:ast.Add, ast.Mult:ast.Add, ast.Div:ast.FloorDiv,
             ast.FloorDiv:ast.Div, ast.Mod:ast.FloorDiv, ast.Pow:ast.Mult,
             ast.Eq:ast.NotEq, ast.NotEq:ast.Eq, ast.Lt:ast.LtE, ast.LtE:ast.Lt,
             ast.Gt:ast.GtE, ast.GtE:ast.Gt, ast.In:ast.NotIn, ast.NotIn:ast.In,
             ast.And:ast.Or, ast.Or:ast.And}
    for source in build():
        fixed = source['fixed']
        tree = ast.parse(fixed)
        candidates = [source['buggy']]
        nodes = list(ast.walk(tree))
        for index,node in enumerate(nodes):
            if type(node) not in swaps:
                continue
            mutated = copy.deepcopy(tree)
            target = list(ast.walk(mutated))[index]
            # Locate this operator by identity within its parent.
            for parent in ast.walk(mutated):
                for field,value in ast.iter_fields(parent):
                    if value is target:
                        setattr(parent,field,swaps[type(node)]())
                    elif isinstance(value,list):
                        for j,child in enumerate(value):
                            if child is target:
                                value[j] = swaps[type(node)]()
            candidates.append(ast.unparse(ast.fix_missing_locations(mutated)))
        for buggy in candidates:
            key = (buggy,fixed)
            if key in unique:
                continue
            # These are only our finite source-controlled templates; no user corpus is executed.
            ns = {}
            try:
                exec(buggy,ns)
            except (SyntaxError,ValueError):
                continue
            killed = False
            for case in source['tests']:
                try:
                    if ns[source['task']](*case['args']) != case['expected']:
                        killed = True
                except Exception:
                    killed = True
            if not killed:
                continue
            row = dict(source,buggy=buggy)
            try:
                encode(row,384)
            except ValueError:
                continue
            unique.add(key); rows.append(row)
    # Split before augmentation: all mutations and related tasks in a family stay together.
    held = {'arithmetic'}
    for name,subset in [('bugs-train.jsonl',[r for r in rows if r['family'] not in held]),
                        ('bugs-validation.jsonl',[r for r in rows if r['family'] in held])]:
        (out/name).write_text(''.join(json.dumps(r)+'\n' for r in subset),encoding='utf-8')
    report = dict(corpus_files=len(manifest), training_bytes=len(train), validation_bytes=len(valid),
                  verified_bug_pairs=len(rows), original_tasks=100,
                  note='Mutations are related examples, not independent tasks. No pretrained weights or remote downloads.')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='pipeline-data')
    prepare(Path(ap.parse_args().out))
