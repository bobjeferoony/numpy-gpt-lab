"""Generate and test our own small Python bug templates; never execute user data."""
import json
from pathlib import Path
import random


def build():
    families = {
        'factorial': ('n', '    r = 1\n    for i in range(1, n{end}):\n        r *= i\n    return r', {'end': ''}, {'end': ' + 1'}, [(0,), (1,), (5,)]),
        'sum_range': ('n', '    return sum(range(n{end}))', {'end': ''}, {'end': ' + 1'}, [(1,), (3,), (6,)]),
        'last': ('xs', '    return xs[{index}]', {'index': '0'}, {'index': '-1'}, [([1, 2, 3],), ([4, 7],)]),
        'first': ('xs', '    return xs[{index}]', {'index': '1'}, {'index': '0'}, [([1, 2, 3],), ([4, 7],)]),
        'reverse': ('xs', '    return xs[::{step}]', {'step': '1'}, {'step': '-1'}, [([1, 2, 3],), ([4, 7],)]),
        'even': ('n', '    return n % 2 == {value}', {'value': '1'}, {'value': '0'}, [(0,), (1,), (4,)]),
        'odd': ('n', '    return n % 2 == {value}', {'value': '0'}, {'value': '1'}, [(0,), (1,), (4,)]),
        'maximum': ('a, b', '    return a if a {op} b else b', {'op': '<'}, {'op': '>'}, [(3, 8), (9, 1)]),
        'minimum': ('a, b', '    return a if a {op} b else b', {'op': '>'}, {'op': '<'}, [(3, 8), (9, 1)]),
        'add': ('a, b', '    return a {op} b', {'op': '-'}, {'op': '+'}, [(3, 8), (9, 1)]),
        'multiply': ('a, b', '    return a {op} b', {'op': '+'}, {'op': '*'}, [(3, 8), (9, 1)]),
        'square': ('n', '    return n {op} 2', {'op': '*'}, {'op': '**'}, [(3,), (5,)]),
        'positive': ('n', '    return n {op} 0', {'op': '<'}, {'op': '>'}, [(-2,), (0,), (4,)]),
        'empty': ('xs', '    return len(xs) {op} 0', {'op': '>'}, {'op': '=='}, [([],), ([1],)]),
        'contains': ('xs, n', '    return n {op} xs', {'op': 'not in'}, {'op': 'in'}, [([1, 2], 2), ([1], 3)]),
        'average': ('xs', '    return sum(xs){suffix}', {'suffix': ''}, {'suffix': ' / len(xs)'}, [([2, 4],), ([3, 6, 9],)]),
    }
    refs = {'factorial': lambda n: __import__('math').factorial(n), 'sum_range': lambda n: n*(n+1)//2,
            'last': lambda xs: xs[-1], 'first': lambda xs: xs[0], 'reverse': lambda xs: list(reversed(xs)),
            'even': lambda n: n%2==0, 'odd': lambda n: n%2!=0, 'maximum': max, 'minimum': min,
            'add': lambda a,b:a+b, 'multiply':lambda a,b:a*b, 'square':lambda n:n*n,
            'positive':lambda n:n>0, 'empty':lambda xs:len(xs)==0, 'contains':lambda xs,n:n in xs,
            'average':lambda xs:sum(xs)/len(xs)}
    rows = []
    for family, (params, body, bad, good, cases) in families.items():
        for variant in range(40):
            name = family if variant == 0 else f'{family}_{variant}'
            buggy = f'def {name}({params}):\n' + body.format(**bad)
            fixed = f'def {name}({params}):\n' + body.format(**good)
            # Only these source-controlled templates are executed, not JSONL files.
            ns = {}; exec(fixed, ns)
            assert all(ns[name](*args) == refs[family](*args) for args in cases)
            ns = {}; exec(buggy, ns)
            assert any(ns[name](*args) != refs[family](*args) for args in cases)
            rows.append(dict(buggy=buggy, fixed=fixed, family=family))
    return rows


if __name__ == '__main__':
    rows = build()
    random.Random(42).shuffle(rows)
    Path('advanced.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n', encoding='utf-8')
    print(f'Wrote {len(rows)} checked examples in 16 template families. Variants are mostly renamings, not independent bugs.')
