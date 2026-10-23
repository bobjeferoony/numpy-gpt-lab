# 100 distinct debugging exercises

Each exercise has at least two checked cases. No renamed duplicate rows. Checks cover these cases, not every input. Families are held out using the current trainer seed; this validation set is not an independent final test set.

## add (Validation; arithmetic)

Buggy:
```python
def add(a, b):
    return a-b
```

Reference fix:
```python
def add(a, b):
    return a+b
```

Cases: `[{"args": [3, 8], "expected": 11}, {"args": [-2, 2], "expected": 0}]`

## subtract (Validation; arithmetic)

Buggy:
```python
def subtract(a, b):
    return a+b
```

Reference fix:
```python
def subtract(a, b):
    return a-b
```

Cases: `[{"args": [3, 8], "expected": -5}, {"args": [9, 2], "expected": 7}]`

## multiply (Validation; arithmetic)

Buggy:
```python
def multiply(a, b):
    return a+b
```

Reference fix:
```python
def multiply(a, b):
    return a*b
```

Cases: `[{"args": [3, 8], "expected": 24}, {"args": [0, 9], "expected": 0}]`

## divide (Validation; arithmetic)

Buggy:
```python
def divide(a, b):
    return a//b
```

Reference fix:
```python
def divide(a, b):
    return a/b
```

Cases: `[{"args": [5, 2], "expected": 2.5}, {"args": [9, 3], "expected": 3}]`

## square (Validation; arithmetic)

Buggy:
```python
def square(n):
    return n*2
```

Reference fix:
```python
def square(n):
    return n**2
```

Cases: `[{"args": [3], "expected": 9}, {"args": [-4], "expected": 16}]`

## cube (Validation; arithmetic)

Buggy:
```python
def cube(n):
    return n**2
```

Reference fix:
```python
def cube(n):
    return n**3
```

Cases: `[{"args": [3], "expected": 27}, {"args": [-2], "expected": -8}]`

## absolute (Validation; arithmetic)

Buggy:
```python
def absolute(n):
    return -n if n>0 else n
```

Reference fix:
```python
def absolute(n):
    return -n if n<0 else n
```

Cases: `[{"args": [-3], "expected": 3}, {"args": [4], "expected": 4}]`

## average (Validation; arithmetic)

Buggy:
```python
def average(xs):
    return sum(xs)
```

Reference fix:
```python
def average(xs):
    return sum(xs)/len(xs)
```

Cases: `[{"args": [[2, 4]], "expected": 3}, {"args": [[1, 2, 6]], "expected": 3}]`

## midpoint (Validation; arithmetic)

Buggy:
```python
def midpoint(a, b):
    return a+b/2
```

Reference fix:
```python
def midpoint(a, b):
    return (a+b)/2
```

Cases: `[{"args": [2, 8], "expected": 5}, {"args": [-4, 4], "expected": 0}]`

## percent (Validation; arithmetic)

Buggy:
```python
def percent(part, total):
    return part/total
```

Reference fix:
```python
def percent(part, total):
    return 100*part/total
```

Cases: `[{"args": [1, 4], "expected": 25}, {"args": [2, 2], "expected": 100}]`

## celsius (Validation; arithmetic)

Buggy:
```python
def celsius(f):
    return f-32*5/9
```

Reference fix:
```python
def celsius(f):
    return (f-32)*5/9
```

Cases: `[{"args": [32], "expected": 0}, {"args": [212], "expected": 100}]`

## fahrenheit (Validation; arithmetic)

Buggy:
```python
def fahrenheit(c):
    return c+32*9/5
```

Reference fix:
```python
def fahrenheit(c):
    return c*9/5+32
```

Cases: `[{"args": [0], "expected": 32}, {"args": [100], "expected": 212}]`

## rectangle_area (Validation; arithmetic)

Buggy:
```python
def rectangle_area(w, h):
    return 2*(w+h)
```

Reference fix:
```python
def rectangle_area(w, h):
    return w*h
```

Cases: `[{"args": [3, 4], "expected": 12}, {"args": [2, 5], "expected": 10}]`

## perimeter (Validation; arithmetic)

Buggy:
```python
def perimeter(w, h):
    return w*h
```

Reference fix:
```python
def perimeter(w, h):
    return 2*(w+h)
```

Cases: `[{"args": [3, 4], "expected": 14}, {"args": [2, 5], "expected": 14}]`

## triangle_area (Validation; arithmetic)

Buggy:
```python
def triangle_area(b, h):
    return b*h
```

Reference fix:
```python
def triangle_area(b, h):
    return b*h/2
```

Cases: `[{"args": [4, 3], "expected": 6}, {"args": [2, 8], "expected": 8}]`

## seconds_to_minutes (Validation; arithmetic)

Buggy:
```python
def seconds_to_minutes(n):
    return n*60
```

Reference fix:
```python
def seconds_to_minutes(n):
    return n/60
```

Cases: `[{"args": [120], "expected": 2}, {"args": [30], "expected": 0.5}]`

## remainder (Validation; arithmetic)

Buggy:
```python
def remainder(a, b):
    return a//b
```

Reference fix:
```python
def remainder(a, b):
    return a%b
```

Cases: `[{"args": [7, 3], "expected": 1}, {"args": [8, 4], "expected": 0}]`

## ceil_div (Validation; arithmetic)

Buggy:
```python
def ceil_div(a, b):
    return a//b
```

Reference fix:
```python
def ceil_div(a, b):
    return -(-a//b)
```

Cases: `[{"args": [7, 3], "expected": 3}, {"args": [6, 3], "expected": 2}]`

## distance (Validation; arithmetic)

Buggy:
```python
def distance(a, b):
    return a-b
```

Reference fix:
```python
def distance(a, b):
    return abs(a-b)
```

Cases: `[{"args": [2, 8], "expected": 6}, {"args": [8, 2], "expected": 6}]`

## clamp (Validation; arithmetic)

Buggy:
```python
def clamp(x, lo, hi):
    return max(hi,min(lo,x))
```

Reference fix:
```python
def clamp(x, lo, hi):
    return max(lo,min(hi,x))
```

Cases: `[{"args": [5, 0, 10], "expected": 5}, {"args": [-2, 0, 10], "expected": 0}, {"args": [12, 0, 10], "expected": 10}]`

## even (Training; predicates)

Buggy:
```python
def even(n):
    return n%2==1
```

Reference fix:
```python
def even(n):
    return n%2==0
```

Cases: `[{"args": [2], "expected": true}, {"args": [3], "expected": false}]`

## odd (Training; predicates)

Buggy:
```python
def odd(n):
    return n%2==0
```

Reference fix:
```python
def odd(n):
    return n%2==1
```

Cases: `[{"args": [2], "expected": false}, {"args": [3], "expected": true}]`

## positive (Training; predicates)

Buggy:
```python
def positive(n):
    return n>=0
```

Reference fix:
```python
def positive(n):
    return n>0
```

Cases: `[{"args": [0], "expected": false}, {"args": [2], "expected": true}]`

## negative (Training; predicates)

Buggy:
```python
def negative(n):
    return n<=0
```

Reference fix:
```python
def negative(n):
    return n<0
```

Cases: `[{"args": [0], "expected": false}, {"args": [-2], "expected": true}]`

## nonnegative (Training; predicates)

Buggy:
```python
def nonnegative(n):
    return n>0
```

Reference fix:
```python
def nonnegative(n):
    return n>=0
```

Cases: `[{"args": [0], "expected": true}, {"args": [-2], "expected": false}]`

## divisible (Training; predicates)

Buggy:
```python
def divisible(n, k):
    return n%k==1
```

Reference fix:
```python
def divisible(n, k):
    return n%k==0
```

Cases: `[{"args": [6, 3], "expected": true}, {"args": [7, 3], "expected": false}]`

## between (Training; predicates)

Buggy:
```python
def between(x, lo, hi):
    return lo<x<hi
```

Reference fix:
```python
def between(x, lo, hi):
    return lo<=x<=hi
```

Cases: `[{"args": [1, 1, 3], "expected": true}, {"args": [4, 1, 3], "expected": false}]`

## outside (Training; predicates)

Buggy:
```python
def outside(x, lo, hi):
    return x<lo and x>hi
```

Reference fix:
```python
def outside(x, lo, hi):
    return x<lo or x>hi
```

Cases: `[{"args": [0, 1, 3], "expected": true}, {"args": [2, 1, 3], "expected": false}]`

## both (Training; predicates)

Buggy:
```python
def both(a, b):
    return a or b
```

Reference fix:
```python
def both(a, b):
    return a and b
```

Cases: `[{"args": [true, false], "expected": false}, {"args": [true, true], "expected": true}]`

## either (Training; predicates)

Buggy:
```python
def either(a, b):
    return a and b
```

Reference fix:
```python
def either(a, b):
    return a or b
```

Cases: `[{"args": [true, false], "expected": true}, {"args": [false, false], "expected": false}]`

## exclusive (Training; predicates)

Buggy:
```python
def exclusive(a, b):
    return a==b
```

Reference fix:
```python
def exclusive(a, b):
    return a!=b
```

Cases: `[{"args": [true, false], "expected": true}, {"args": [true, true], "expected": false}]`

## empty (Training; predicates)

Buggy:
```python
def empty(xs):
    return len(xs)>0
```

Reference fix:
```python
def empty(xs):
    return len(xs)==0
```

Cases: `[{"args": [[]], "expected": true}, {"args": [[1]], "expected": false}]`

## contains (Training; predicates)

Buggy:
```python
def contains(xs, x):
    return x not in xs
```

Reference fix:
```python
def contains(xs, x):
    return x in xs
```

Cases: `[{"args": [[1, 2], 2], "expected": true}, {"args": [[1], 3], "expected": false}]`

## all_positive (Training; predicates)

Buggy:
```python
def all_positive(xs):
    return any(x>0 for x in xs)
```

Reference fix:
```python
def all_positive(xs):
    return all(x>0 for x in xs)
```

Cases: `[{"args": [[1, -1]], "expected": false}, {"args": [[1, 2]], "expected": true}]`

## any_negative (Training; predicates)

Buggy:
```python
def any_negative(xs):
    return all(x<0 for x in xs)
```

Reference fix:
```python
def any_negative(xs):
    return any(x<0 for x in xs)
```

Cases: `[{"args": [[1, -1]], "expected": true}, {"args": [[1, 2]], "expected": false}]`

## first (Training; sequences)

Buggy:
```python
def first(xs):
    return xs[1]
```

Reference fix:
```python
def first(xs):
    return xs[0]
```

Cases: `[{"args": [[3, 8]], "expected": 3}, {"args": [[9]], "expected": 9}]`

## last (Training; sequences)

Buggy:
```python
def last(xs):
    return xs[len(xs)]
```

Reference fix:
```python
def last(xs):
    return xs[-1]
```

Cases: `[{"args": [[3, 8]], "expected": 8}, {"args": [[9]], "expected": 9}]`

## reverse (Training; sequences)

Buggy:
```python
def reverse(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse(xs):
    return xs[::-1]
```

Cases: `[{"args": [[1, 2, 3]], "expected": [3, 2, 1]}, {"args": [[]], "expected": []}]`

## drop_first (Training; sequences)

Buggy:
```python
def drop_first(xs):
    return xs[:-1]
```

Reference fix:
```python
def drop_first(xs):
    return xs[1:]
```

Cases: `[{"args": [[1, 2, 3]], "expected": [2, 3]}, {"args": [[1]], "expected": []}]`

## drop_last (Training; sequences)

Buggy:
```python
def drop_last(xs):
    return xs[1:]
```

Reference fix:
```python
def drop_last(xs):
    return xs[:-1]
```

Cases: `[{"args": [[1, 2, 3]], "expected": [1, 2]}, {"args": [[1]], "expected": []}]`

## take (Training; sequences)

Buggy:
```python
def take(xs, n):
    return xs[n:]
```

Reference fix:
```python
def take(xs, n):
    return xs[:n]
```

Cases: `[{"args": [[1, 2, 3], 2], "expected": [1, 2]}, {"args": [[1], 0], "expected": []}]`

## tail (Training; sequences)

Buggy:
```python
def tail(xs, n):
    return xs[:n]
```

Reference fix:
```python
def tail(xs, n):
    return xs[-n:] if n else []
```

Cases: `[{"args": [[1, 2, 3], 2], "expected": [2, 3]}, {"args": [[1], 0], "expected": []}]`

## rotate_left (Training; sequences)

Buggy:
```python
def rotate_left(xs):
    return xs[-1:]+xs[:-1]
```

Reference fix:
```python
def rotate_left(xs):
    return xs[1:]+xs[:1]
```

Cases: `[{"args": [[1, 2, 3]], "expected": [2, 3, 1]}, {"args": [[]], "expected": []}]`

## rotate_right (Training; sequences)

Buggy:
```python
def rotate_right(xs):
    return xs[1:]+xs[:1]
```

Reference fix:
```python
def rotate_right(xs):
    return xs[-1:]+xs[:-1]
```

Cases: `[{"args": [[1, 2, 3]], "expected": [3, 1, 2]}, {"args": [[]], "expected": []}]`

## descending (Training; sequences)

Buggy:
```python
def descending(xs):
    return sorted(xs)
```

Reference fix:
```python
def descending(xs):
    return sorted(xs,reverse=True)
```

Cases: `[{"args": [[3, 1, 2]], "expected": [3, 2, 1]}, {"args": [[]], "expected": []}]`

## second_distinct (Training; sequences)

Buggy:
```python
def second_distinct(xs):
    return sorted(xs)[-2]
```

Reference fix:
```python
def second_distinct(xs):
    return sorted(set(xs))[-2]
```

Cases: `[{"args": [[2, 7, 7, 4]], "expected": 4}, {"args": [[1, 2]], "expected": 1}]`

## unique_count (Training; sequences)

Buggy:
```python
def unique_count(xs):
    return len(xs)
```

Reference fix:
```python
def unique_count(xs):
    return len(set(xs))
```

Cases: `[{"args": [[1, 1, 2]], "expected": 2}, {"args": [[]], "expected": 0}]`

## even_positions (Training; sequences)

Buggy:
```python
def even_positions(xs):
    return xs[1::2]
```

Reference fix:
```python
def even_positions(xs):
    return xs[::2]
```

Cases: `[{"args": [[1, 2, 3, 4]], "expected": [1, 3]}, {"args": [[]], "expected": []}]`

## middle (Training; sequences)

Buggy:
```python
def middle(xs):
    return xs[len(xs)]
```

Reference fix:
```python
def middle(xs):
    return xs[len(xs)//2]
```

Cases: `[{"args": [[1, 2, 3]], "expected": 2}, {"args": [[8]], "expected": 8}]`

## flatten (Training; sequences)

Buggy:
```python
def flatten(xs):
    return [xs for x in xs]
```

Reference fix:
```python
def flatten(xs):
    return [v for x in xs for v in x]
```

Cases: `[{"args": [[[1, 2], [3]]], "expected": [1, 2, 3]}, {"args": [[]], "expected": []}]`

## trim (Training; strings)

Buggy:
```python
def trim(s):
    return s.lstrip()
```

Reference fix:
```python
def trim(s):
    return s.strip()
```

Cases: `[{"args": [" a "], "expected": "a"}, {"args": ["x"], "expected": "x"}]`

## lowercase (Training; strings)

Buggy:
```python
def lowercase(s):
    return s.upper()
```

Reference fix:
```python
def lowercase(s):
    return s.lower()
```

Cases: `[{"args": ["AbC"], "expected": "abc"}, {"args": [""], "expected": ""}]`

## uppercase (Training; strings)

Buggy:
```python
def uppercase(s):
    return s.lower()
```

Reference fix:
```python
def uppercase(s):
    return s.upper()
```

Cases: `[{"args": ["AbC"], "expected": "ABC"}, {"args": [""], "expected": ""}]`

## word_count (Training; strings)

Buggy:
```python
def word_count(s):
    return len(s)
```

Reference fix:
```python
def word_count(s):
    return len(s.split())
```

Cases: `[{"args": ["a  b"], "expected": 2}, {"args": [""], "expected": 0}]`

## join_words (Training; strings)

Buggy:
```python
def join_words(xs):
    return "".join(xs)
```

Reference fix:
```python
def join_words(xs):
    return " ".join(xs)
```

Cases: `[{"args": [["a", "b"]], "expected": "a b"}, {"args": [[]], "expected": ""}]`

## palindrome (Training; strings)

Buggy:
```python
def palindrome(s):
    return s==s[::1]
```

Reference fix:
```python
def palindrome(s):
    return s==s[::-1]
```

Cases: `[{"args": ["aba"], "expected": true}, {"args": ["abc"], "expected": false}]`

## starts (Training; strings)

Buggy:
```python
def starts(s, prefix):
    return s.endswith(prefix)
```

Reference fix:
```python
def starts(s, prefix):
    return s.startswith(prefix)
```

Cases: `[{"args": ["abc", "ab"], "expected": true}, {"args": ["abc", "bc"], "expected": false}]`

## ends (Training; strings)

Buggy:
```python
def ends(s, suffix):
    return s.startswith(suffix)
```

Reference fix:
```python
def ends(s, suffix):
    return s.endswith(suffix)
```

Cases: `[{"args": ["abc", "bc"], "expected": true}, {"args": ["abc", "ab"], "expected": false}]`

## replace_all (Training; strings)

Buggy:
```python
def replace_all(s, a, b):
    return s.replace(a,b,1)
```

Reference fix:
```python
def replace_all(s, a, b):
    return s.replace(a,b)
```

Cases: `[{"args": ["aaa", "a", "b"], "expected": "bbb"}, {"args": ["abc", "x", "y"], "expected": "abc"}]`

## vowel_count (Training; strings)

Buggy:
```python
def vowel_count(s):
    return sum(c in "aeiou" for c in s)
```

Reference fix:
```python
def vowel_count(s):
    return sum(c in "aeiou" for c in s.lower())
```

Cases: `[{"args": ["AEiou"], "expected": 5}, {"args": ["xyz"], "expected": 0}]`

## remove_spaces (Training; strings)

Buggy:
```python
def remove_spaces(s):
    return s.strip()
```

Reference fix:
```python
def remove_spaces(s):
    return s.replace(" ","")
```

Cases: `[{"args": [" a b "], "expected": "ab"}, {"args": ["x"], "expected": "x"}]`

## initials (Training; strings)

Buggy:
```python
def initials(s):
    return "".join(w[-1] for w in s.split())
```

Reference fix:
```python
def initials(s):
    return "".join(w[0] for w in s.split())
```

Cases: `[{"args": ["Ada Lovelace"], "expected": "AL"}, {"args": [""], "expected": ""}]`

## is_digits (Training; strings)

Buggy:
```python
def is_digits(s):
    return s.isalpha()
```

Reference fix:
```python
def is_digits(s):
    return s.isdigit()
```

Cases: `[{"args": ["123"], "expected": true}, {"args": ["abc"], "expected": false}]`

## split_csv (Training; strings)

Buggy:
```python
def split_csv(s):
    return s.split()
```

Reference fix:
```python
def split_csv(s):
    return s.split(",")
```

Cases: `[{"args": ["a,b"], "expected": ["a", "b"]}, {"args": ["a"], "expected": ["a"]}]`

## remove_prefix (Training; strings)

Buggy:
```python
def remove_prefix(s, p):
    return s[len(p):]
```

Reference fix:
```python
def remove_prefix(s, p):
    return s[len(p):] if s.startswith(p) else s
```

Cases: `[{"args": ["abc", "ab"], "expected": "c"}, {"args": ["xyz", "ab"], "expected": "xyz"}]`

## keep_even (Training; filtering)

Buggy:
```python
def keep_even(xs):
    return [x for x in xs if x%2]
```

Reference fix:
```python
def keep_even(xs):
    return [x for x in xs if x%2==0]
```

Cases: `[{"args": [[1, 2, 3, 4]], "expected": [2, 4]}, {"args": [[]], "expected": []}]`

## keep_odd (Training; filtering)

Buggy:
```python
def keep_odd(xs):
    return [x for x in xs if x%2==0]
```

Reference fix:
```python
def keep_odd(xs):
    return [x for x in xs if x%2]
```

Cases: `[{"args": [[1, 2, 3, 4]], "expected": [1, 3]}, {"args": [[]], "expected": []}]`

## keep_positive (Training; filtering)

Buggy:
```python
def keep_positive(xs):
    return [x for x in xs if x>=0]
```

Reference fix:
```python
def keep_positive(xs):
    return [x for x in xs if x>0]
```

Cases: `[{"args": [[-1, 0, 2]], "expected": [2]}, {"args": [[]], "expected": []}]`

## keep_nonzero (Training; filtering)

Buggy:
```python
def keep_nonzero(xs):
    return [x for x in xs if x==0]
```

Reference fix:
```python
def keep_nonzero(xs):
    return [x for x in xs if x!=0]
```

Cases: `[{"args": [[-1, 0, 2]], "expected": [-1, 2]}, {"args": [[]], "expected": []}]`

## sum_squares (Training; filtering)

Buggy:
```python
def sum_squares(xs):
    return sum(xs)**2
```

Reference fix:
```python
def sum_squares(xs):
    return sum(x*x for x in xs)
```

Cases: `[{"args": [[1, 2, 3]], "expected": 14}, {"args": [[]], "expected": 0}]`

## count_positive (Training; filtering)

Buggy:
```python
def count_positive(xs):
    return sum(x>=0 for x in xs)
```

Reference fix:
```python
def count_positive(xs):
    return sum(x>0 for x in xs)
```

Cases: `[{"args": [[-1, 0, 2]], "expected": 1}, {"args": [[]], "expected": 0}]`

## double_each (Training; filtering)

Buggy:
```python
def double_each(xs):
    return xs*2
```

Reference fix:
```python
def double_each(xs):
    return [x*2 for x in xs]
```

Cases: `[{"args": [[1, 2]], "expected": [2, 4]}, {"args": [[]], "expected": []}]`

## clip_negative (Training; filtering)

Buggy:
```python
def clip_negative(xs):
    return [min(0,x) for x in xs]
```

Reference fix:
```python
def clip_negative(xs):
    return [max(0,x) for x in xs]
```

Cases: `[{"args": [[-1, 2]], "expected": [0, 2]}, {"args": [[]], "expected": []}]`

## remove_none (Training; filtering)

Buggy:
```python
def remove_none(xs):
    return [x for x in xs if x]
```

Reference fix:
```python
def remove_none(xs):
    return [x for x in xs if x is not None]
```

Cases: `[{"args": [[null, 0, false, 2]], "expected": [0, false, 2]}, {"args": [[]], "expected": []}]`

## long_words (Training; filtering)

Buggy:
```python
def long_words(xs, n):
    return [s for s in xs if len(s)>n]
```

Reference fix:
```python
def long_words(xs, n):
    return [s for s in xs if len(s)>=n]
```

Cases: `[{"args": [["a", "ab", "abc"], 2], "expected": ["ab", "abc"]}, {"args": [[], 2], "expected": []}]`

## lookup (Training; maps_sets)

Buggy:
```python
def lookup(d, k):
    return d[k]
```

Reference fix:
```python
def lookup(d, k):
    return d.get(k)
```

Cases: `[{"args": [{"a": 1}, "b"], "expected": null}, {"args": [{"a": 1}, "a"], "expected": 1}]`

## invert (Training; maps_sets)

Buggy:
```python
def invert(d):
    return {k:v for k,v in d.items()}
```

Reference fix:
```python
def invert(d):
    return {v:k for k,v in d.items()}
```

Cases: `[{"args": [{"a": 1, "b": 2}], "expected": {"1": "a", "2": "b"}}, {"args": [{}], "expected": {}}]`

## sum_values (Training; maps_sets)

Buggy:
```python
def sum_values(d):
    return sum(d)
```

Reference fix:
```python
def sum_values(d):
    return sum(d.values())
```

Cases: `[{"args": [{"a": 1, "b": 2}], "expected": 3}, {"args": [{}], "expected": 0}]`

## merge (Training; maps_sets)

Buggy:
```python
def merge(a, b):
    return {**b,**a}
```

Reference fix:
```python
def merge(a, b):
    return {**a,**b}
```

Cases: `[{"args": [{"x": 1}, {"x": 2}], "expected": {"x": 2}}, {"args": [{}, {}], "expected": {}}]`

## has_key (Training; maps_sets)

Buggy:
```python
def has_key(d, k):
    return k in d.values()
```

Reference fix:
```python
def has_key(d, k):
    return k in d
```

Cases: `[{"args": [{"a": 1}, "a"], "expected": true}, {"args": [{}, "a"], "expected": false}]`

## common (Training; maps_sets)

Buggy:
```python
def common(a, b):
    return sorted(set(a)|set(b))
```

Reference fix:
```python
def common(a, b):
    return sorted(set(a)&set(b))
```

Cases: `[{"args": [[1, 2], [2, 3]], "expected": [2]}, {"args": [[], [1]], "expected": []}]`

## union (Training; maps_sets)

Buggy:
```python
def union(a, b):
    return sorted(set(a)&set(b))
```

Reference fix:
```python
def union(a, b):
    return sorted(set(a)|set(b))
```

Cases: `[{"args": [[1, 2], [2, 3]], "expected": [1, 2, 3]}, {"args": [[], [1]], "expected": [1]}]`

## difference (Training; maps_sets)

Buggy:
```python
def difference(a, b):
    return sorted(set(b)-set(a))
```

Reference fix:
```python
def difference(a, b):
    return sorted(set(a)-set(b))
```

Cases: `[{"args": [[1, 2], [2, 3]], "expected": [1]}, {"args": [[], [1]], "expected": []}]`

## subset (Training; maps_sets)

Buggy:
```python
def subset(a, b):
    return set(a)<set(b)
```

Reference fix:
```python
def subset(a, b):
    return set(a)<=set(b)
```

Cases: `[{"args": [[1], [1]], "expected": true}, {"args": [[1, 2], [1]], "expected": false}]`

## duplicates (Training; maps_sets)

Buggy:
```python
def duplicates(xs):
    return len(xs)==len(set(xs))
```

Reference fix:
```python
def duplicates(xs):
    return len(xs)!=len(set(xs))
```

Cases: `[{"args": [[1, 1]], "expected": true}, {"args": [[1, 2]], "expected": false}]`

## factorial (Training; loops)

Buggy:
```python
def factorial(n):
    r = 1
    for i in range(1,n):
        r *= i
    return r
```

Reference fix:
```python
def factorial(n):
    r = 1
    for i in range(1,n+1):
        r *= i
    return r
```

Cases: `[{"args": [0], "expected": 1}, {"args": [5], "expected": 120}]`

## product (Training; loops)

Buggy:
```python
def product(xs):
    r = 0
    for x in xs:
        r *= x
    return r
```

Reference fix:
```python
def product(xs):
    r = 1
    for x in xs:
        r *= x
    return r
```

Cases: `[{"args": [[2, 3, 4]], "expected": 24}, {"args": [[]], "expected": 1}]`

## sum_to (Training; loops)

Buggy:
```python
def sum_to(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_to(n):
    return sum(range(n+1))
```

Cases: `[{"args": [0], "expected": 0}, {"args": [5], "expected": 15}]`

## find_index (Training; loops)

Buggy:
```python
def find_index(xs, target):
    for i,x in enumerate(xs):
        if x==target:
            return x
    return -1
```

Reference fix:
```python
def find_index(xs, target):
    for i,x in enumerate(xs):
        if x==target:
            return i
    return -1
```

Cases: `[{"args": [[4, 8], 8], "expected": 1}, {"args": [[4], 9], "expected": -1}]`

## count_item (Training; loops)

Buggy:
```python
def count_item(xs, target):
    r = 0
    for x in xs:
        if x==target:
            r = 1
    return r
```

Reference fix:
```python
def count_item(xs, target):
    r = 0
    for x in xs:
        if x==target:
            r += 1
    return r
```

Cases: `[{"args": [[2, 2, 3], 2], "expected": 2}, {"args": [[], 2], "expected": 0}]`

## running_sum (Training; loops)

Buggy:
```python
def running_sum(xs):
    r = []
    s = 0
    for x in xs:
        s = x
        r.append(s)
    return r
```

Reference fix:
```python
def running_sum(xs):
    r = []
    s = 0
    for x in xs:
        s += x
        r.append(s)
    return r
```

Cases: `[{"args": [[1, 2, 3]], "expected": [1, 3, 6]}, {"args": [[]], "expected": []}]`

## is_sorted (Training; loops)

Buggy:
```python
def is_sorted(xs):
    return all(a<b for a,b in zip(xs,xs[1:]))
```

Reference fix:
```python
def is_sorted(xs):
    return all(a<=b for a,b in zip(xs,xs[1:]))
```

Cases: `[{"args": [[1, 1, 2]], "expected": true}, {"args": [[2, 1]], "expected": false}]`

## adjacent_diffs (Training; loops)

Buggy:
```python
def adjacent_diffs(xs):
    return [a-b for a,b in zip(xs,xs[1:])]
```

Reference fix:
```python
def adjacent_diffs(xs):
    return [b-a for a,b in zip(xs,xs[1:])]
```

Cases: `[{"args": [[1, 4, 9]], "expected": [3, 5]}, {"args": [[1]], "expected": []}]`

## indices (Training; loops)

Buggy:
```python
def indices(xs):
    return list(range(len(xs)+1))
```

Reference fix:
```python
def indices(xs):
    return list(range(len(xs)))
```

Cases: `[{"args": [["a", "b"]], "expected": [0, 1]}, {"args": [[]], "expected": []}]`

## fibonacci (Training; loops)

Buggy:
```python
def fibonacci(n):
    a,b = 0,1
    for _ in range(n+1):
        a,b = b,a+b
    return a
```

Reference fix:
```python
def fibonacci(n):
    a,b = 0,1
    for _ in range(n):
        a,b = b,a+b
    return a
```

Cases: `[{"args": [0], "expected": 0}, {"args": [1], "expected": 1}, {"args": [7], "expected": 13}]`

## safe_first (Training; boundaries)

Buggy:
```python
def safe_first(xs):
    return xs[0]
```

Reference fix:
```python
def safe_first(xs):
    return xs[0] if xs else None
```

Cases: `[{"args": [[]], "expected": null}, {"args": [[2]], "expected": 2}]`

## safe_mean (Training; boundaries)

Buggy:
```python
def safe_mean(xs):
    return sum(xs)/len(xs)
```

Reference fix:
```python
def safe_mean(xs):
    return sum(xs)/len(xs) if xs else 0
```

Cases: `[{"args": [[]], "expected": 0}, {"args": [[2, 4]], "expected": 3}]`

## safe_divide (Training; boundaries)

Buggy:
```python
def safe_divide(a, b):
    return a/b
```

Reference fix:
```python
def safe_divide(a, b):
    return a/b if b else None
```

Cases: `[{"args": [2, 0], "expected": null}, {"args": [3, 2], "expected": 1.5}]`

## maximum (Training; boundaries)

Buggy:
```python
def maximum(a, b):
    return a if a<b else b
```

Reference fix:
```python
def maximum(a, b):
    return a if a>b else b
```

Cases: `[{"args": [2, 8], "expected": 8}, {"args": [9, 1], "expected": 9}]`

## minimum (Training; boundaries)

Buggy:
```python
def minimum(a, b):
    return a if a>b else b
```

Reference fix:
```python
def minimum(a, b):
    return a if a<b else b
```

Cases: `[{"args": [2, 8], "expected": 2}, {"args": [9, 1], "expected": 1}]`
