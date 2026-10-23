"""Source-controlled exercises with behavior tests; no execution of user files."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from train import encode


def build():
    rows = []
    def add(group, name, params, bad, good, cases):
        buggy = f'def {name}({params}):\n' + '\n'.join('    '+s for s in bad.splitlines())
        fixed = f'def {name}({params}):\n' + '\n'.join('    '+s for s in good.splitlines())
        ns = {}; exec(fixed, ns)
        for args, expected in cases:
            assert ns[name](*args) == expected, (name, args, expected)
        ns = {}; exec(buggy, ns)
        failures = 0
        for args, expected in cases:
            try:
                failures += ns[name](*args) != expected
            except Exception:
                failures += 1
        assert failures, name
        row = dict(buggy=buggy, fixed=fixed, family=group, task=name,
                   tests=[dict(args=args, expected=expected) for args,expected in cases])
        encode(row, 384)
        rows.append(row)

    # Each entry changes behavior, not merely identifiers.
    def short(group, name, params, bad, good, cases):
        add(group, name, params, 'return '+bad, 'return '+good, cases)

    arithmetic = [
        ('add','a, b','a-b','a+b',[((3,8),11),((-2,2),0)]),
        ('subtract','a, b','a+b','a-b',[((3,8),-5),((9,2),7)]),
        ('multiply','a, b','a+b','a*b',[((3,8),24),((0,9),0)]),
        ('divide','a, b','a//b','a/b',[((5,2),2.5),((9,3),3)]),
        ('square','n','n*2','n**2',[((3,),9),((-4,),16)]),
        ('cube','n','n**2','n**3',[((3,),27),((-2,),-8)]),
        ('absolute','n','-n if n>0 else n','-n if n<0 else n',[((-3,),3),((4,),4)]),
        ('average','xs','sum(xs)','sum(xs)/len(xs)',[(([2,4],),3),(([1,2,6],),3)]),
        ('midpoint','a, b','a+b/2','(a+b)/2',[((2,8),5),((-4,4),0)]),
        ('percent','part, total','part/total','100*part/total',[((1,4),25),((2,2),100)]),
        ('celsius','f','f-32*5/9','(f-32)*5/9',[((32,),0),((212,),100)]),
        ('fahrenheit','c','c+32*9/5','c*9/5+32',[((0,),32),((100,),212)]),
        ('rectangle_area','w, h','2*(w+h)','w*h',[((3,4),12),((2,5),10)]),
        ('perimeter','w, h','w*h','2*(w+h)',[((3,4),14),((2,5),14)]),
        ('triangle_area','b, h','b*h','b*h/2',[((4,3),6),((2,8),8)]),
        ('seconds_to_minutes','n','n*60','n/60',[((120,),2),((30,),.5)]),
        ('remainder','a, b','a//b','a%b',[((7,3),1),((8,4),0)]),
        ('ceil_div','a, b','a//b','-(-a//b)',[((7,3),3),((6,3),2)]),
        ('distance','a, b','a-b','abs(a-b)',[((2,8),6),((8,2),6)]),
        ('clamp','x, lo, hi','max(hi,min(lo,x))','max(lo,min(hi,x))',[((5,0,10),5),((-2,0,10),0),((12,0,10),10)]),
    ]
    for item in arithmetic: short('arithmetic', *item)

    predicates = [
        ('even','n','n%2==1','n%2==0',[((2,),True),((3,),False)]),
        ('odd','n','n%2==0','n%2==1',[((2,),False),((3,),True)]),
        ('positive','n','n>=0','n>0',[((0,),False),((2,),True)]),
        ('negative','n','n<=0','n<0',[((0,),False),((-2,),True)]),
        ('nonnegative','n','n>0','n>=0',[((0,),True),((-2,),False)]),
        ('divisible','n, k','n%k==1','n%k==0',[((6,3),True),((7,3),False)]),
        ('between','x, lo, hi','lo<x<hi','lo<=x<=hi',[((1,1,3),True),((4,1,3),False)]),
        ('outside','x, lo, hi','x<lo and x>hi','x<lo or x>hi',[((0,1,3),True),((2,1,3),False)]),
        ('both','a, b','a or b','a and b',[((True,False),False),((True,True),True)]),
        ('either','a, b','a and b','a or b',[((True,False),True),((False,False),False)]),
        ('exclusive','a, b','a==b','a!=b',[((True,False),True),((True,True),False)]),
        ('empty','xs','len(xs)>0','len(xs)==0',[(([],),True),(([1],),False)]),
        ('contains','xs, x','x not in xs','x in xs',[(([1,2],2),True),(([1],3),False)]),
        ('all_positive','xs','any(x>0 for x in xs)','all(x>0 for x in xs)',[(([1,-1],),False),(([1,2],),True)]),
        ('any_negative','xs','all(x<0 for x in xs)','any(x<0 for x in xs)',[(([1,-1],),True),(([1,2],),False)]),
    ]
    for item in predicates: short('predicates', *item)

    sequences = [
        ('first','xs','xs[1]','xs[0]',[(([3,8],),3),(([9],),9)]),
        ('last','xs','xs[len(xs)]','xs[-1]',[(([3,8],),8),(([9],),9)]),
        ('reverse','xs','xs[::1]','xs[::-1]',[(([1,2,3],),[3,2,1]),(([],),[])]),
        ('drop_first','xs','xs[:-1]','xs[1:]',[(([1,2,3],),[2,3]),(([1],),[])]),
        ('drop_last','xs','xs[1:]','xs[:-1]',[(([1,2,3],),[1,2]),(([1],),[])]),
        ('take','xs, n','xs[n:]','xs[:n]',[(([1,2,3],2),[1,2]),(([1],0),[])]),
        ('tail','xs, n','xs[:n]','xs[-n:] if n else []',[(([1,2,3],2),[2,3]),(([1],0),[])]),
        ('rotate_left','xs','xs[-1:]+xs[:-1]','xs[1:]+xs[:1]',[(([1,2,3],),[2,3,1]),(([],),[])]),
        ('rotate_right','xs','xs[1:]+xs[:1]','xs[-1:]+xs[:-1]',[(([1,2,3],),[3,1,2]),(([],),[])]),
        ('descending','xs','sorted(xs)','sorted(xs,reverse=True)',[(([3,1,2],),[3,2,1]),(([],),[])]),
        ('second_distinct','xs','sorted(xs)[-2]','sorted(set(xs))[-2]',[(([2,7,7,4],),4),(([1,2],),1)]),
        ('unique_count','xs','len(xs)','len(set(xs))',[(([1,1,2],),2),(([],),0)]),
        ('even_positions','xs','xs[1::2]','xs[::2]',[(([1,2,3,4],),[1,3]),(([],),[])]),
        ('middle','xs','xs[len(xs)]','xs[len(xs)//2]',[(([1,2,3],),2),(([8],),8)]),
        ('flatten','xs','[xs for x in xs]','[v for x in xs for v in x]',[(([[1,2],[3]],),[1,2,3]),(([],),[])]),
    ]
    for item in sequences: short('sequences', *item)

    strings = [
        ('trim','s','s.lstrip()','s.strip()',[((' a ',),'a'),(('x',),'x')]),
        ('lowercase','s','s.upper()','s.lower()',[(('AbC',),'abc'),(('',),'')]),
        ('uppercase','s','s.lower()','s.upper()',[(('AbC',),'ABC'),(('',),'')]),
        ('word_count','s','len(s)','len(s.split())',[(('a  b',),2),(('',),0)]),
        ('join_words','xs','"".join(xs)','" ".join(xs)',[((['a','b'],),'a b'),(([],),'')]),
        ('palindrome','s','s==s[::1]','s==s[::-1]',[(('aba',),True),(('abc',),False)]),
        ('starts','s, prefix','s.endswith(prefix)','s.startswith(prefix)',[(('abc','ab'),True),(('abc','bc'),False)]),
        ('ends','s, suffix','s.startswith(suffix)','s.endswith(suffix)',[(('abc','bc'),True),(('abc','ab'),False)]),
        ('replace_all','s, a, b','s.replace(a,b,1)','s.replace(a,b)',[(('aaa','a','b'),'bbb'),(('abc','x','y'),'abc')]),
        ('vowel_count','s','sum(c in "aeiou" for c in s)','sum(c in "aeiou" for c in s.lower())',[(('AEiou',),5),(('xyz',),0)]),
        ('remove_spaces','s','s.strip()','s.replace(" ","")',[((' a b ',),'ab'),(('x',),'x')]),
        ('initials','s','"".join(w[-1] for w in s.split())','"".join(w[0] for w in s.split())',[(('Ada Lovelace',),'AL'),(('',),'')]),
        ('is_digits','s','s.isalpha()','s.isdigit()',[(('123',),True),(('abc',),False)]),
        ('split_csv','s','s.split()','s.split(",")',[(('a,b',),['a','b']),(('a',),['a'])]),
        ('remove_prefix','s, p','s[len(p):]','s[len(p):] if s.startswith(p) else s',[(('abc','ab'),'c'),(('xyz','ab'),'xyz')]),
    ]
    for item in strings: short('strings', *item)

    filters = [
        ('keep_even','xs','[x for x in xs if x%2]','[x for x in xs if x%2==0]',[(([1,2,3,4],),[2,4]),(([],),[])]),
        ('keep_odd','xs','[x for x in xs if x%2==0]','[x for x in xs if x%2]',[(([1,2,3,4],),[1,3]),(([],),[])]),
        ('keep_positive','xs','[x for x in xs if x>=0]','[x for x in xs if x>0]',[(([-1,0,2],),[2]),(([],),[])]),
        ('keep_nonzero','xs','[x for x in xs if x==0]','[x for x in xs if x!=0]',[(([-1,0,2],),[-1,2]),(([],),[])]),
        ('sum_squares','xs','sum(xs)**2','sum(x*x for x in xs)',[(([1,2,3],),14),(([],),0)]),
        ('count_positive','xs','sum(x>=0 for x in xs)','sum(x>0 for x in xs)',[(([-1,0,2],),1),(([],),0)]),
        ('double_each','xs','xs*2','[x*2 for x in xs]',[(([1,2],),[2,4]),(([],),[])]),
        ('clip_negative','xs','[min(0,x) for x in xs]','[max(0,x) for x in xs]',[(([-1,2],),[0,2]),(([],),[])]),
        ('remove_none','xs','[x for x in xs if x]','[x for x in xs if x is not None]',[(([None,0,False,2],),[0,False,2]),(([],),[])]),
        ('long_words','xs, n','[s for s in xs if len(s)>n]','[s for s in xs if len(s)>=n]',[((['a','ab','abc'],2),['ab','abc']),(([],2),[])]),
    ]
    for item in filters: short('filtering', *item)

    maps = [
        ('lookup','d, k','d[k]','d.get(k)',[(({'a':1},'b'),None),(({'a':1},'a'),1)]),
        ('invert','d','{k:v for k,v in d.items()}','{v:k for k,v in d.items()}',[(({'a':1,'b':2},),{1:'a',2:'b'}),(({},),{})]),
        ('sum_values','d','sum(d)','sum(d.values())',[(({'a':1,'b':2},),3),(({},),0)]),
        ('merge','a, b','{**b,**a}','{**a,**b}',[(({'x':1},{'x':2}),{'x':2}),(({},{}),{})]),
        ('has_key','d, k','k in d.values()','k in d',[(({'a':1},'a'),True),(({},'a'),False)]),
        ('common','a, b','sorted(set(a)|set(b))','sorted(set(a)&set(b))',[(([1,2],[2,3]),[2]),(([],[1]),[])]),
        ('union','a, b','sorted(set(a)&set(b))','sorted(set(a)|set(b))',[(([1,2],[2,3]),[1,2,3]),(([],[1]),[1])]),
        ('difference','a, b','sorted(set(b)-set(a))','sorted(set(a)-set(b))',[(([1,2],[2,3]),[1]),(([],[1]),[])]),
        ('subset','a, b','set(a)<set(b)','set(a)<=set(b)',[(([1],[1]),True),(([1,2],[1]),False)]),
        ('duplicates','xs','len(xs)==len(set(xs))','len(xs)!=len(set(xs))',[(([1,1],),True),(([1,2],),False)]),
    ]
    for item in maps: short('maps_sets', *item)

    loops = [
        ('factorial','n','r = 1\nfor i in range(1,n):\n    r *= i\nreturn r','r = 1\nfor i in range(1,n+1):\n    r *= i\nreturn r',[((0,),1),((5,),120)]),
        ('product','xs','r = 0\nfor x in xs:\n    r *= x\nreturn r','r = 1\nfor x in xs:\n    r *= x\nreturn r',[(([2,3,4],),24),(([],),1)]),
        ('sum_to','n','return sum(range(n))','return sum(range(n+1))',[((0,),0),((5,),15)]),
        ('find_index','xs, target','for i,x in enumerate(xs):\n    if x==target:\n        return x\nreturn -1','for i,x in enumerate(xs):\n    if x==target:\n        return i\nreturn -1',[(([4,8],8),1),(([4],9),-1)]),
        ('count_item','xs, target','r = 0\nfor x in xs:\n    if x==target:\n        r = 1\nreturn r','r = 0\nfor x in xs:\n    if x==target:\n        r += 1\nreturn r',[(([2,2,3],2),2),(([],2),0)]),
        ('running_sum','xs','r = []\ns = 0\nfor x in xs:\n    s = x\n    r.append(s)\nreturn r','r = []\ns = 0\nfor x in xs:\n    s += x\n    r.append(s)\nreturn r',[(([1,2,3],),[1,3,6]),(([],),[])]),
        ('is_sorted','xs','return all(a<b for a,b in zip(xs,xs[1:]))','return all(a<=b for a,b in zip(xs,xs[1:]))',[(([1,1,2],),True),(([2,1],),False)]),
        ('adjacent_diffs','xs','return [a-b for a,b in zip(xs,xs[1:])]','return [b-a for a,b in zip(xs,xs[1:])]',[(([1,4,9],),[3,5]),(([1],),[])]),
        ('indices','xs','return list(range(len(xs)+1))','return list(range(len(xs)))',[((['a','b'],),[0,1]),(([],),[])]),
        ('fibonacci','n','a,b = 0,1\nfor _ in range(n+1):\n    a,b = b,a+b\nreturn a','a,b = 0,1\nfor _ in range(n):\n    a,b = b,a+b\nreturn a',[((0,),0),((1,),1),((7,),13)]),
    ]
    for item in loops: add('loops', *item)

    edges = [
        ('safe_first','xs','return xs[0]','return xs[0] if xs else None',[(([],),None),(([2],),2)]),
        ('safe_mean','xs','return sum(xs)/len(xs)','return sum(xs)/len(xs) if xs else 0',[(([],),0),(([2,4],),3)]),
        ('safe_divide','a, b','return a/b','return a/b if b else None',[((2,0),None),((3,2),1.5)]),
        ('maximum','a, b','return a if a<b else b','return a if a>b else b',[((2,8),8),((9,1),9)]),
        ('minimum','a, b','return a if a>b else b','return a if a<b else b',[((2,8),2),((9,1),1)]),
    ]
    for item in edges: add('boundaries', *item)
    assert len(rows) == 100, len(rows)
    assert len({r['buggy'] for r in rows}) == len(rows)
    return rows


if __name__ == '__main__':
    rows = build()
    groups = sorted({r['family'] for r in rows})
    np.random.default_rng(123).shuffle(groups)
    held = set(groups[:max(1,len(groups)//5)])
    Path('distinct.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n', encoding='utf-8')
    test = [r for r in rows if r['family'] in held]
    Path('distinct-heldout.jsonl').write_text('\n'.join(json.dumps(r) for r in test)+'\n', encoding='utf-8')
    lines = ['# 100 distinct debugging exercises', '', 'Each exercise has at least two checked cases. No renamed duplicate rows. Checks cover these cases, not every input. Families are held out using the current trainer seed; this validation set is not an independent final test set.', '']
    for row in rows:
        split = 'Validation' if row['family'] in held else 'Training'
        lines += [f"## {row['task']} ({split}; {row['family']})", '', 'Buggy:', '```python', row['buggy'], '```', '', 'Reference fix:', '```python', row['fixed'], '```', '', 'Cases: `'+json.dumps(row['tests'])+'`', '']
    Path('DISTINCT_PROBLEMS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Checked',len(rows),'different exercises:',dict(Counter(r['family'] for r in rows)))
    print('Training:',len(rows)-len(test),'validation:',len(test),'held-out families:', sorted(held))
