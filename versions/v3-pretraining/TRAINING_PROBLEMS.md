# Dataset audit

Reconstructed from the current advanced.jsonl and the trainer seed (123). The checkpoint does not record a dataset hash or loss history, so this is not proof of the historical dataset contents.

640 rows: 520 training and 120 validation. Only 16 distinct templates; 40 function-name variants each.

## All 16 templates

### add — Training

Buggy:
```python
def add(a, b):
    return a - b
```

Reference fix:
```python
def add(a, b):
    return a + b
```

### average — Training

Buggy:
```python
def average(xs):
    return sum(xs)
```

Reference fix:
```python
def average(xs):
    return sum(xs) / len(xs)
```

### contains — Validation only

Buggy:
```python
def contains(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains(xs, n):
    return n in xs
```

### empty — Training

Buggy:
```python
def empty(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty(xs):
    return len(xs) == 0
```

### even — Training

Buggy:
```python
def even(n):
    return n % 2 == 1
```

Reference fix:
```python
def even(n):
    return n % 2 == 0
```

### factorial — Training

Buggy:
```python
def factorial(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### first — Training

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

### last — Validation only

Buggy:
```python
def last(xs):
    return xs[0]
```

Reference fix:
```python
def last(xs):
    return xs[-1]
```

### maximum — Training

Buggy:
```python
def maximum(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum(a, b):
    return a if a > b else b
```

### minimum — Training

Buggy:
```python
def minimum(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum(a, b):
    return a if a < b else b
```

### multiply — Validation only

Buggy:
```python
def multiply(a, b):
    return a + b
```

Reference fix:
```python
def multiply(a, b):
    return a * b
```

### odd — Training

Buggy:
```python
def odd(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd(n):
    return n % 2 == 1
```

### positive — Training

Buggy:
```python
def positive(n):
    return n < 0
```

Reference fix:
```python
def positive(n):
    return n > 0
```

### reverse — Training

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

### square — Training

Buggy:
```python
def square(n):
    return n * 2
```

Reference fix:
```python
def square(n):
    return n ** 2
```

### sum_range — Training

Buggy:
```python
def sum_range(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range(n):
    return sum(range(n + 1))
```

## All 640 pairs

### 1. reverse (training)

Buggy:
```python
def reverse_12(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_12(xs):
    return xs[::-1]
```

### 2. contains (validation)

Buggy:
```python
def contains_22(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_22(xs, n):
    return n in xs
```

### 3. even (training)

Buggy:
```python
def even_19(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_19(n):
    return n % 2 == 0
```

### 4. maximum (training)

Buggy:
```python
def maximum_5(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_5(a, b):
    return a if a > b else b
```

### 5. minimum (training)

Buggy:
```python
def minimum_37(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_37(a, b):
    return a if a < b else b
```

### 6. positive (training)

Buggy:
```python
def positive_13(n):
    return n < 0
```

Reference fix:
```python
def positive_13(n):
    return n > 0
```

### 7. even (training)

Buggy:
```python
def even_21(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_21(n):
    return n % 2 == 0
```

### 8. maximum (training)

Buggy:
```python
def maximum_27(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_27(a, b):
    return a if a > b else b
```

### 9. last (validation)

Buggy:
```python
def last_38(xs):
    return xs[0]
```

Reference fix:
```python
def last_38(xs):
    return xs[-1]
```

### 10. positive (training)

Buggy:
```python
def positive_4(n):
    return n < 0
```

Reference fix:
```python
def positive_4(n):
    return n > 0
```

### 11. maximum (training)

Buggy:
```python
def maximum_34(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_34(a, b):
    return a if a > b else b
```

### 12. multiply (validation)

Buggy:
```python
def multiply_9(a, b):
    return a + b
```

Reference fix:
```python
def multiply_9(a, b):
    return a * b
```

### 13. empty (training)

Buggy:
```python
def empty_9(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_9(xs):
    return len(xs) == 0
```

### 14. positive (training)

Buggy:
```python
def positive_33(n):
    return n < 0
```

Reference fix:
```python
def positive_33(n):
    return n > 0
```

### 15. sum_range (training)

Buggy:
```python
def sum_range_22(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_22(n):
    return sum(range(n + 1))
```

### 16. reverse (training)

Buggy:
```python
def reverse_2(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_2(xs):
    return xs[::-1]
```

### 17. factorial (training)

Buggy:
```python
def factorial_12(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_12(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 18. positive (training)

Buggy:
```python
def positive_22(n):
    return n < 0
```

Reference fix:
```python
def positive_22(n):
    return n > 0
```

### 19. empty (training)

Buggy:
```python
def empty_22(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_22(xs):
    return len(xs) == 0
```

### 20. odd (training)

Buggy:
```python
def odd_39(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_39(n):
    return n % 2 == 1
```

### 21. last (validation)

Buggy:
```python
def last_12(xs):
    return xs[0]
```

Reference fix:
```python
def last_12(xs):
    return xs[-1]
```

### 22. empty (training)

Buggy:
```python
def empty_37(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_37(xs):
    return len(xs) == 0
```

### 23. factorial (training)

Buggy:
```python
def factorial_23(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_23(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 24. contains (validation)

Buggy:
```python
def contains_26(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_26(xs, n):
    return n in xs
```

### 25. even (training)

Buggy:
```python
def even_30(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_30(n):
    return n % 2 == 0
```

### 26. add (training)

Buggy:
```python
def add_16(a, b):
    return a - b
```

Reference fix:
```python
def add_16(a, b):
    return a + b
```

### 27. first (training)

Buggy:
```python
def first_16(xs):
    return xs[1]
```

Reference fix:
```python
def first_16(xs):
    return xs[0]
```

### 28. maximum (training)

Buggy:
```python
def maximum_7(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_7(a, b):
    return a if a > b else b
```

### 29. minimum (training)

Buggy:
```python
def minimum_15(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_15(a, b):
    return a if a < b else b
```

### 30. factorial (training)

Buggy:
```python
def factorial_14(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_14(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 31. empty (training)

Buggy:
```python
def empty_33(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_33(xs):
    return len(xs) == 0
```

### 32. minimum (training)

Buggy:
```python
def minimum_5(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_5(a, b):
    return a if a < b else b
```

### 33. reverse (training)

Buggy:
```python
def reverse_35(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_35(xs):
    return xs[::-1]
```

### 34. first (training)

Buggy:
```python
def first_20(xs):
    return xs[1]
```

Reference fix:
```python
def first_20(xs):
    return xs[0]
```

### 35. positive (training)

Buggy:
```python
def positive_30(n):
    return n < 0
```

Reference fix:
```python
def positive_30(n):
    return n > 0
```

### 36. positive (training)

Buggy:
```python
def positive_6(n):
    return n < 0
```

Reference fix:
```python
def positive_6(n):
    return n > 0
```

### 37. square (training)

Buggy:
```python
def square_37(n):
    return n * 2
```

Reference fix:
```python
def square_37(n):
    return n ** 2
```

### 38. minimum (training)

Buggy:
```python
def minimum(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum(a, b):
    return a if a < b else b
```

### 39. add (training)

Buggy:
```python
def add_18(a, b):
    return a - b
```

Reference fix:
```python
def add_18(a, b):
    return a + b
```

### 40. empty (training)

Buggy:
```python
def empty_18(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_18(xs):
    return len(xs) == 0
```

### 41. odd (training)

Buggy:
```python
def odd_21(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_21(n):
    return n % 2 == 1
```

### 42. average (training)

Buggy:
```python
def average_1(xs):
    return sum(xs)
```

Reference fix:
```python
def average_1(xs):
    return sum(xs) / len(xs)
```

### 43. odd (training)

Buggy:
```python
def odd_26(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_26(n):
    return n % 2 == 1
```

### 44. even (training)

Buggy:
```python
def even_15(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_15(n):
    return n % 2 == 0
```

### 45. add (training)

Buggy:
```python
def add_6(a, b):
    return a - b
```

Reference fix:
```python
def add_6(a, b):
    return a + b
```

### 46. maximum (training)

Buggy:
```python
def maximum_37(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_37(a, b):
    return a if a > b else b
```

### 47. multiply (validation)

Buggy:
```python
def multiply_33(a, b):
    return a + b
```

Reference fix:
```python
def multiply_33(a, b):
    return a * b
```

### 48. positive (training)

Buggy:
```python
def positive_26(n):
    return n < 0
```

Reference fix:
```python
def positive_26(n):
    return n > 0
```

### 49. first (training)

Buggy:
```python
def first_11(xs):
    return xs[1]
```

Reference fix:
```python
def first_11(xs):
    return xs[0]
```

### 50. last (validation)

Buggy:
```python
def last_7(xs):
    return xs[0]
```

Reference fix:
```python
def last_7(xs):
    return xs[-1]
```

### 51. sum_range (training)

Buggy:
```python
def sum_range_12(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_12(n):
    return sum(range(n + 1))
```

### 52. factorial (training)

Buggy:
```python
def factorial_8(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_8(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 53. add (training)

Buggy:
```python
def add_37(a, b):
    return a - b
```

Reference fix:
```python
def add_37(a, b):
    return a + b
```

### 54. contains (validation)

Buggy:
```python
def contains_7(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_7(xs, n):
    return n in xs
```

### 55. contains (validation)

Buggy:
```python
def contains_35(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_35(xs, n):
    return n in xs
```

### 56. factorial (training)

Buggy:
```python
def factorial_15(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_15(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 57. maximum (training)

Buggy:
```python
def maximum_17(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_17(a, b):
    return a if a > b else b
```

### 58. first (training)

Buggy:
```python
def first_18(xs):
    return xs[1]
```

Reference fix:
```python
def first_18(xs):
    return xs[0]
```

### 59. maximum (training)

Buggy:
```python
def maximum_2(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_2(a, b):
    return a if a > b else b
```

### 60. factorial (training)

Buggy:
```python
def factorial_33(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_33(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 61. contains (validation)

Buggy:
```python
def contains_37(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_37(xs, n):
    return n in xs
```

### 62. sum_range (training)

Buggy:
```python
def sum_range_29(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_29(n):
    return sum(range(n + 1))
```

### 63. square (training)

Buggy:
```python
def square_18(n):
    return n * 2
```

Reference fix:
```python
def square_18(n):
    return n ** 2
```

### 64. maximum (training)

Buggy:
```python
def maximum_35(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_35(a, b):
    return a if a > b else b
```

### 65. empty (training)

Buggy:
```python
def empty_3(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_3(xs):
    return len(xs) == 0
```

### 66. factorial (training)

Buggy:
```python
def factorial_13(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_13(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 67. empty (training)

Buggy:
```python
def empty_2(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_2(xs):
    return len(xs) == 0
```

### 68. contains (validation)

Buggy:
```python
def contains_31(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_31(xs, n):
    return n in xs
```

### 69. last (validation)

Buggy:
```python
def last_39(xs):
    return xs[0]
```

Reference fix:
```python
def last_39(xs):
    return xs[-1]
```

### 70. last (validation)

Buggy:
```python
def last_36(xs):
    return xs[0]
```

Reference fix:
```python
def last_36(xs):
    return xs[-1]
```

### 71. contains (validation)

Buggy:
```python
def contains_28(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_28(xs, n):
    return n in xs
```

### 72. multiply (validation)

Buggy:
```python
def multiply_22(a, b):
    return a + b
```

Reference fix:
```python
def multiply_22(a, b):
    return a * b
```

### 73. factorial (training)

Buggy:
```python
def factorial_24(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_24(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 74. contains (validation)

Buggy:
```python
def contains_11(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_11(xs, n):
    return n in xs
```

### 75. average (training)

Buggy:
```python
def average_25(xs):
    return sum(xs)
```

Reference fix:
```python
def average_25(xs):
    return sum(xs) / len(xs)
```

### 76. reverse (training)

Buggy:
```python
def reverse_23(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_23(xs):
    return xs[::-1]
```

### 77. multiply (validation)

Buggy:
```python
def multiply_34(a, b):
    return a + b
```

Reference fix:
```python
def multiply_34(a, b):
    return a * b
```

### 78. add (training)

Buggy:
```python
def add_39(a, b):
    return a - b
```

Reference fix:
```python
def add_39(a, b):
    return a + b
```

### 79. maximum (training)

Buggy:
```python
def maximum(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum(a, b):
    return a if a > b else b
```

### 80. factorial (training)

Buggy:
```python
def factorial_5(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_5(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 81. empty (training)

Buggy:
```python
def empty_11(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_11(xs):
    return len(xs) == 0
```

### 82. factorial (training)

Buggy:
```python
def factorial_18(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_18(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 83. multiply (validation)

Buggy:
```python
def multiply_3(a, b):
    return a + b
```

Reference fix:
```python
def multiply_3(a, b):
    return a * b
```

### 84. square (training)

Buggy:
```python
def square_26(n):
    return n * 2
```

Reference fix:
```python
def square_26(n):
    return n ** 2
```

### 85. even (training)

Buggy:
```python
def even_9(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_9(n):
    return n % 2 == 0
```

### 86. minimum (training)

Buggy:
```python
def minimum_39(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_39(a, b):
    return a if a < b else b
```

### 87. odd (training)

Buggy:
```python
def odd_14(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_14(n):
    return n % 2 == 1
```

### 88. sum_range (training)

Buggy:
```python
def sum_range_2(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_2(n):
    return sum(range(n + 1))
```

### 89. first (training)

Buggy:
```python
def first_14(xs):
    return xs[1]
```

Reference fix:
```python
def first_14(xs):
    return xs[0]
```

### 90. contains (validation)

Buggy:
```python
def contains_23(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_23(xs, n):
    return n in xs
```

### 91. last (validation)

Buggy:
```python
def last_16(xs):
    return xs[0]
```

Reference fix:
```python
def last_16(xs):
    return xs[-1]
```

### 92. square (training)

Buggy:
```python
def square_11(n):
    return n * 2
```

Reference fix:
```python
def square_11(n):
    return n ** 2
```

### 93. minimum (training)

Buggy:
```python
def minimum_25(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_25(a, b):
    return a if a < b else b
```

### 94. even (training)

Buggy:
```python
def even_22(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_22(n):
    return n % 2 == 0
```

### 95. add (training)

Buggy:
```python
def add_17(a, b):
    return a - b
```

Reference fix:
```python
def add_17(a, b):
    return a + b
```

### 96. factorial (training)

Buggy:
```python
def factorial_21(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_21(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 97. contains (validation)

Buggy:
```python
def contains_25(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_25(xs, n):
    return n in xs
```

### 98. add (training)

Buggy:
```python
def add_4(a, b):
    return a - b
```

Reference fix:
```python
def add_4(a, b):
    return a + b
```

### 99. square (training)

Buggy:
```python
def square_14(n):
    return n * 2
```

Reference fix:
```python
def square_14(n):
    return n ** 2
```

### 100. maximum (training)

Buggy:
```python
def maximum_28(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_28(a, b):
    return a if a > b else b
```

### 101. positive (training)

Buggy:
```python
def positive_1(n):
    return n < 0
```

Reference fix:
```python
def positive_1(n):
    return n > 0
```

### 102. empty (training)

Buggy:
```python
def empty_32(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_32(xs):
    return len(xs) == 0
```

### 103. reverse (training)

Buggy:
```python
def reverse_37(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_37(xs):
    return xs[::-1]
```

### 104. add (training)

Buggy:
```python
def add_23(a, b):
    return a - b
```

Reference fix:
```python
def add_23(a, b):
    return a + b
```

### 105. even (training)

Buggy:
```python
def even_7(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_7(n):
    return n % 2 == 0
```

### 106. first (training)

Buggy:
```python
def first_12(xs):
    return xs[1]
```

Reference fix:
```python
def first_12(xs):
    return xs[0]
```

### 107. sum_range (training)

Buggy:
```python
def sum_range_37(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_37(n):
    return sum(range(n + 1))
```

### 108. sum_range (training)

Buggy:
```python
def sum_range_34(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_34(n):
    return sum(range(n + 1))
```

### 109. last (validation)

Buggy:
```python
def last_20(xs):
    return xs[0]
```

Reference fix:
```python
def last_20(xs):
    return xs[-1]
```

### 110. contains (validation)

Buggy:
```python
def contains_33(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_33(xs, n):
    return n in xs
```

### 111. add (training)

Buggy:
```python
def add_33(a, b):
    return a - b
```

Reference fix:
```python
def add_33(a, b):
    return a + b
```

### 112. multiply (validation)

Buggy:
```python
def multiply_23(a, b):
    return a + b
```

Reference fix:
```python
def multiply_23(a, b):
    return a * b
```

### 113. reverse (training)

Buggy:
```python
def reverse_26(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_26(xs):
    return xs[::-1]
```

### 114. contains (validation)

Buggy:
```python
def contains_39(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_39(xs, n):
    return n in xs
```

### 115. last (validation)

Buggy:
```python
def last_26(xs):
    return xs[0]
```

Reference fix:
```python
def last_26(xs):
    return xs[-1]
```

### 116. sum_range (training)

Buggy:
```python
def sum_range_30(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_30(n):
    return sum(range(n + 1))
```

### 117. reverse (training)

Buggy:
```python
def reverse_27(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_27(xs):
    return xs[::-1]
```

### 118. even (training)

Buggy:
```python
def even_37(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_37(n):
    return n % 2 == 0
```

### 119. sum_range (training)

Buggy:
```python
def sum_range_32(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_32(n):
    return sum(range(n + 1))
```

### 120. reverse (training)

Buggy:
```python
def reverse_24(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_24(xs):
    return xs[::-1]
```

### 121. odd (training)

Buggy:
```python
def odd_25(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_25(n):
    return n % 2 == 1
```

### 122. square (training)

Buggy:
```python
def square_1(n):
    return n * 2
```

Reference fix:
```python
def square_1(n):
    return n ** 2
```

### 123. minimum (training)

Buggy:
```python
def minimum_8(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_8(a, b):
    return a if a < b else b
```

### 124. last (validation)

Buggy:
```python
def last_6(xs):
    return xs[0]
```

Reference fix:
```python
def last_6(xs):
    return xs[-1]
```

### 125. first (training)

Buggy:
```python
def first_30(xs):
    return xs[1]
```

Reference fix:
```python
def first_30(xs):
    return xs[0]
```

### 126. odd (training)

Buggy:
```python
def odd_13(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_13(n):
    return n % 2 == 1
```

### 127. even (training)

Buggy:
```python
def even(n):
    return n % 2 == 1
```

Reference fix:
```python
def even(n):
    return n % 2 == 0
```

### 128. positive (training)

Buggy:
```python
def positive_35(n):
    return n < 0
```

Reference fix:
```python
def positive_35(n):
    return n > 0
```

### 129. reverse (training)

Buggy:
```python
def reverse_33(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_33(xs):
    return xs[::-1]
```

### 130. average (training)

Buggy:
```python
def average_35(xs):
    return sum(xs)
```

Reference fix:
```python
def average_35(xs):
    return sum(xs) / len(xs)
```

### 131. sum_range (training)

Buggy:
```python
def sum_range_35(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_35(n):
    return sum(range(n + 1))
```

### 132. contains (validation)

Buggy:
```python
def contains_21(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_21(xs, n):
    return n in xs
```

### 133. last (validation)

Buggy:
```python
def last_3(xs):
    return xs[0]
```

Reference fix:
```python
def last_3(xs):
    return xs[-1]
```

### 134. even (training)

Buggy:
```python
def even_12(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_12(n):
    return n % 2 == 0
```

### 135. square (training)

Buggy:
```python
def square_31(n):
    return n * 2
```

Reference fix:
```python
def square_31(n):
    return n ** 2
```

### 136. first (training)

Buggy:
```python
def first_9(xs):
    return xs[1]
```

Reference fix:
```python
def first_9(xs):
    return xs[0]
```

### 137. sum_range (training)

Buggy:
```python
def sum_range_36(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_36(n):
    return sum(range(n + 1))
```

### 138. minimum (training)

Buggy:
```python
def minimum_20(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_20(a, b):
    return a if a < b else b
```

### 139. empty (training)

Buggy:
```python
def empty_13(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_13(xs):
    return len(xs) == 0
```

### 140. odd (training)

Buggy:
```python
def odd_7(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_7(n):
    return n % 2 == 1
```

### 141. odd (training)

Buggy:
```python
def odd_28(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_28(n):
    return n % 2 == 1
```

### 142. add (training)

Buggy:
```python
def add_22(a, b):
    return a - b
```

Reference fix:
```python
def add_22(a, b):
    return a + b
```

### 143. average (training)

Buggy:
```python
def average_22(xs):
    return sum(xs)
```

Reference fix:
```python
def average_22(xs):
    return sum(xs) / len(xs)
```

### 144. positive (training)

Buggy:
```python
def positive_18(n):
    return n < 0
```

Reference fix:
```python
def positive_18(n):
    return n > 0
```

### 145. multiply (validation)

Buggy:
```python
def multiply_24(a, b):
    return a + b
```

Reference fix:
```python
def multiply_24(a, b):
    return a * b
```

### 146. square (training)

Buggy:
```python
def square_20(n):
    return n * 2
```

Reference fix:
```python
def square_20(n):
    return n ** 2
```

### 147. positive (training)

Buggy:
```python
def positive_15(n):
    return n < 0
```

Reference fix:
```python
def positive_15(n):
    return n > 0
```

### 148. first (training)

Buggy:
```python
def first_35(xs):
    return xs[1]
```

Reference fix:
```python
def first_35(xs):
    return xs[0]
```

### 149. empty (training)

Buggy:
```python
def empty_5(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_5(xs):
    return len(xs) == 0
```

### 150. square (training)

Buggy:
```python
def square_28(n):
    return n * 2
```

Reference fix:
```python
def square_28(n):
    return n ** 2
```

### 151. empty (training)

Buggy:
```python
def empty_30(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_30(xs):
    return len(xs) == 0
```

### 152. first (training)

Buggy:
```python
def first_27(xs):
    return xs[1]
```

Reference fix:
```python
def first_27(xs):
    return xs[0]
```

### 153. add (training)

Buggy:
```python
def add_25(a, b):
    return a - b
```

Reference fix:
```python
def add_25(a, b):
    return a + b
```

### 154. contains (validation)

Buggy:
```python
def contains_16(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_16(xs, n):
    return n in xs
```

### 155. reverse (training)

Buggy:
```python
def reverse_38(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_38(xs):
    return xs[::-1]
```

### 156. even (training)

Buggy:
```python
def even_11(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_11(n):
    return n % 2 == 0
```

### 157. minimum (training)

Buggy:
```python
def minimum_30(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_30(a, b):
    return a if a < b else b
```

### 158. last (validation)

Buggy:
```python
def last_8(xs):
    return xs[0]
```

Reference fix:
```python
def last_8(xs):
    return xs[-1]
```

### 159. add (training)

Buggy:
```python
def add_35(a, b):
    return a - b
```

Reference fix:
```python
def add_35(a, b):
    return a + b
```

### 160. first (training)

Buggy:
```python
def first_32(xs):
    return xs[1]
```

Reference fix:
```python
def first_32(xs):
    return xs[0]
```

### 161. reverse (training)

Buggy:
```python
def reverse_32(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_32(xs):
    return xs[::-1]
```

### 162. reverse (training)

Buggy:
```python
def reverse_30(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_30(xs):
    return xs[::-1]
```

### 163. empty (training)

Buggy:
```python
def empty_25(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_25(xs):
    return len(xs) == 0
```

### 164. first (training)

Buggy:
```python
def first_28(xs):
    return xs[1]
```

Reference fix:
```python
def first_28(xs):
    return xs[0]
```

### 165. empty (training)

Buggy:
```python
def empty_31(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_31(xs):
    return len(xs) == 0
```

### 166. odd (training)

Buggy:
```python
def odd_15(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_15(n):
    return n % 2 == 1
```

### 167. minimum (training)

Buggy:
```python
def minimum_11(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_11(a, b):
    return a if a < b else b
```

### 168. add (training)

Buggy:
```python
def add_8(a, b):
    return a - b
```

Reference fix:
```python
def add_8(a, b):
    return a + b
```

### 169. reverse (training)

Buggy:
```python
def reverse_17(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_17(xs):
    return xs[::-1]
```

### 170. first (training)

Buggy:
```python
def first_17(xs):
    return xs[1]
```

Reference fix:
```python
def first_17(xs):
    return xs[0]
```

### 171. maximum (training)

Buggy:
```python
def maximum_33(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_33(a, b):
    return a if a > b else b
```

### 172. first (training)

Buggy:
```python
def first_10(xs):
    return xs[1]
```

Reference fix:
```python
def first_10(xs):
    return xs[0]
```

### 173. reverse (training)

Buggy:
```python
def reverse_4(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_4(xs):
    return xs[::-1]
```

### 174. positive (training)

Buggy:
```python
def positive_20(n):
    return n < 0
```

Reference fix:
```python
def positive_20(n):
    return n > 0
```

### 175. positive (training)

Buggy:
```python
def positive_19(n):
    return n < 0
```

Reference fix:
```python
def positive_19(n):
    return n > 0
```

### 176. factorial (training)

Buggy:
```python
def factorial_17(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_17(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 177. add (training)

Buggy:
```python
def add_21(a, b):
    return a - b
```

Reference fix:
```python
def add_21(a, b):
    return a + b
```

### 178. minimum (training)

Buggy:
```python
def minimum_38(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_38(a, b):
    return a if a < b else b
```

### 179. square (training)

Buggy:
```python
def square_13(n):
    return n * 2
```

Reference fix:
```python
def square_13(n):
    return n ** 2
```

### 180. multiply (validation)

Buggy:
```python
def multiply_4(a, b):
    return a + b
```

Reference fix:
```python
def multiply_4(a, b):
    return a * b
```

### 181. reverse (training)

Buggy:
```python
def reverse_8(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_8(xs):
    return xs[::-1]
```

### 182. multiply (validation)

Buggy:
```python
def multiply_36(a, b):
    return a + b
```

Reference fix:
```python
def multiply_36(a, b):
    return a * b
```

### 183. positive (training)

Buggy:
```python
def positive_10(n):
    return n < 0
```

Reference fix:
```python
def positive_10(n):
    return n > 0
```

### 184. sum_range (training)

Buggy:
```python
def sum_range_18(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_18(n):
    return sum(range(n + 1))
```

### 185. odd (training)

Buggy:
```python
def odd(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd(n):
    return n % 2 == 1
```

### 186. average (training)

Buggy:
```python
def average_31(xs):
    return sum(xs)
```

Reference fix:
```python
def average_31(xs):
    return sum(xs) / len(xs)
```

### 187. multiply (validation)

Buggy:
```python
def multiply_27(a, b):
    return a + b
```

Reference fix:
```python
def multiply_27(a, b):
    return a * b
```

### 188. maximum (training)

Buggy:
```python
def maximum_39(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_39(a, b):
    return a if a > b else b
```

### 189. sum_range (training)

Buggy:
```python
def sum_range_16(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_16(n):
    return sum(range(n + 1))
```

### 190. minimum (training)

Buggy:
```python
def minimum_4(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_4(a, b):
    return a if a < b else b
```

### 191. even (training)

Buggy:
```python
def even_35(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_35(n):
    return n % 2 == 0
```

### 192. minimum (training)

Buggy:
```python
def minimum_26(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_26(a, b):
    return a if a < b else b
```

### 193. add (training)

Buggy:
```python
def add_36(a, b):
    return a - b
```

Reference fix:
```python
def add_36(a, b):
    return a + b
```

### 194. reverse (training)

Buggy:
```python
def reverse_19(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_19(xs):
    return xs[::-1]
```

### 195. square (training)

Buggy:
```python
def square_38(n):
    return n * 2
```

Reference fix:
```python
def square_38(n):
    return n ** 2
```

### 196. add (training)

Buggy:
```python
def add_9(a, b):
    return a - b
```

Reference fix:
```python
def add_9(a, b):
    return a + b
```

### 197. even (training)

Buggy:
```python
def even_18(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_18(n):
    return n % 2 == 0
```

### 198. even (training)

Buggy:
```python
def even_1(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_1(n):
    return n % 2 == 0
```

### 199. reverse (training)

Buggy:
```python
def reverse_11(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_11(xs):
    return xs[::-1]
```

### 200. average (training)

Buggy:
```python
def average_17(xs):
    return sum(xs)
```

Reference fix:
```python
def average_17(xs):
    return sum(xs) / len(xs)
```

### 201. square (training)

Buggy:
```python
def square_12(n):
    return n * 2
```

Reference fix:
```python
def square_12(n):
    return n ** 2
```

### 202. reverse (training)

Buggy:
```python
def reverse_28(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_28(xs):
    return xs[::-1]
```

### 203. add (training)

Buggy:
```python
def add(a, b):
    return a - b
```

Reference fix:
```python
def add(a, b):
    return a + b
```

### 204. average (training)

Buggy:
```python
def average_12(xs):
    return sum(xs)
```

Reference fix:
```python
def average_12(xs):
    return sum(xs) / len(xs)
```

### 205. last (validation)

Buggy:
```python
def last_25(xs):
    return xs[0]
```

Reference fix:
```python
def last_25(xs):
    return xs[-1]
```

### 206. sum_range (training)

Buggy:
```python
def sum_range_5(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_5(n):
    return sum(range(n + 1))
```

### 207. square (training)

Buggy:
```python
def square_10(n):
    return n * 2
```

Reference fix:
```python
def square_10(n):
    return n ** 2
```

### 208. odd (training)

Buggy:
```python
def odd_11(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_11(n):
    return n % 2 == 1
```

### 209. sum_range (training)

Buggy:
```python
def sum_range_20(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_20(n):
    return sum(range(n + 1))
```

### 210. factorial (training)

Buggy:
```python
def factorial_39(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_39(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 211. square (training)

Buggy:
```python
def square_7(n):
    return n * 2
```

Reference fix:
```python
def square_7(n):
    return n ** 2
```

### 212. first (training)

Buggy:
```python
def first_38(xs):
    return xs[1]
```

Reference fix:
```python
def first_38(xs):
    return xs[0]
```

### 213. maximum (training)

Buggy:
```python
def maximum_32(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_32(a, b):
    return a if a > b else b
```

### 214. odd (training)

Buggy:
```python
def odd_5(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_5(n):
    return n % 2 == 1
```

### 215. minimum (training)

Buggy:
```python
def minimum_35(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_35(a, b):
    return a if a < b else b
```

### 216. last (validation)

Buggy:
```python
def last_10(xs):
    return xs[0]
```

Reference fix:
```python
def last_10(xs):
    return xs[-1]
```

### 217. odd (training)

Buggy:
```python
def odd_22(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_22(n):
    return n % 2 == 1
```

### 218. odd (training)

Buggy:
```python
def odd_16(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_16(n):
    return n % 2 == 1
```

### 219. sum_range (training)

Buggy:
```python
def sum_range_23(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_23(n):
    return sum(range(n + 1))
```

### 220. reverse (training)

Buggy:
```python
def reverse_14(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_14(xs):
    return xs[::-1]
```

### 221. minimum (training)

Buggy:
```python
def minimum_21(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_21(a, b):
    return a if a < b else b
```

### 222. last (validation)

Buggy:
```python
def last_11(xs):
    return xs[0]
```

Reference fix:
```python
def last_11(xs):
    return xs[-1]
```

### 223. even (training)

Buggy:
```python
def even_27(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_27(n):
    return n % 2 == 0
```

### 224. multiply (validation)

Buggy:
```python
def multiply_2(a, b):
    return a + b
```

Reference fix:
```python
def multiply_2(a, b):
    return a * b
```

### 225. average (training)

Buggy:
```python
def average_19(xs):
    return sum(xs)
```

Reference fix:
```python
def average_19(xs):
    return sum(xs) / len(xs)
```

### 226. contains (validation)

Buggy:
```python
def contains_15(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_15(xs, n):
    return n in xs
```

### 227. empty (training)

Buggy:
```python
def empty_17(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_17(xs):
    return len(xs) == 0
```

### 228. empty (training)

Buggy:
```python
def empty_23(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_23(xs):
    return len(xs) == 0
```

### 229. square (training)

Buggy:
```python
def square(n):
    return n * 2
```

Reference fix:
```python
def square(n):
    return n ** 2
```

### 230. last (validation)

Buggy:
```python
def last_30(xs):
    return xs[0]
```

Reference fix:
```python
def last_30(xs):
    return xs[-1]
```

### 231. multiply (validation)

Buggy:
```python
def multiply_12(a, b):
    return a + b
```

Reference fix:
```python
def multiply_12(a, b):
    return a * b
```

### 232. average (training)

Buggy:
```python
def average_20(xs):
    return sum(xs)
```

Reference fix:
```python
def average_20(xs):
    return sum(xs) / len(xs)
```

### 233. first (training)

Buggy:
```python
def first_29(xs):
    return xs[1]
```

Reference fix:
```python
def first_29(xs):
    return xs[0]
```

### 234. average (training)

Buggy:
```python
def average_14(xs):
    return sum(xs)
```

Reference fix:
```python
def average_14(xs):
    return sum(xs) / len(xs)
```

### 235. even (training)

Buggy:
```python
def even_13(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_13(n):
    return n % 2 == 0
```

### 236. positive (training)

Buggy:
```python
def positive_11(n):
    return n < 0
```

Reference fix:
```python
def positive_11(n):
    return n > 0
```

### 237. multiply (validation)

Buggy:
```python
def multiply_14(a, b):
    return a + b
```

Reference fix:
```python
def multiply_14(a, b):
    return a * b
```

### 238. add (training)

Buggy:
```python
def add_15(a, b):
    return a - b
```

Reference fix:
```python
def add_15(a, b):
    return a + b
```

### 239. square (training)

Buggy:
```python
def square_4(n):
    return n * 2
```

Reference fix:
```python
def square_4(n):
    return n ** 2
```

### 240. reverse (training)

Buggy:
```python
def reverse_16(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_16(xs):
    return xs[::-1]
```

### 241. square (training)

Buggy:
```python
def square_36(n):
    return n * 2
```

Reference fix:
```python
def square_36(n):
    return n ** 2
```

### 242. odd (training)

Buggy:
```python
def odd_1(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_1(n):
    return n % 2 == 1
```

### 243. factorial (training)

Buggy:
```python
def factorial_19(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_19(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 244. factorial (training)

Buggy:
```python
def factorial_28(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_28(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 245. factorial (training)

Buggy:
```python
def factorial_2(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_2(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 246. first (training)

Buggy:
```python
def first_23(xs):
    return xs[1]
```

Reference fix:
```python
def first_23(xs):
    return xs[0]
```

### 247. maximum (training)

Buggy:
```python
def maximum_13(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_13(a, b):
    return a if a > b else b
```

### 248. reverse (training)

Buggy:
```python
def reverse_20(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_20(xs):
    return xs[::-1]
```

### 249. positive (training)

Buggy:
```python
def positive_5(n):
    return n < 0
```

Reference fix:
```python
def positive_5(n):
    return n > 0
```

### 250. last (validation)

Buggy:
```python
def last_33(xs):
    return xs[0]
```

Reference fix:
```python
def last_33(xs):
    return xs[-1]
```

### 251. maximum (training)

Buggy:
```python
def maximum_6(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_6(a, b):
    return a if a > b else b
```

### 252. maximum (training)

Buggy:
```python
def maximum_36(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_36(a, b):
    return a if a > b else b
```

### 253. empty (training)

Buggy:
```python
def empty_7(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_7(xs):
    return len(xs) == 0
```

### 254. positive (training)

Buggy:
```python
def positive_39(n):
    return n < 0
```

Reference fix:
```python
def positive_39(n):
    return n > 0
```

### 255. empty (training)

Buggy:
```python
def empty_12(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_12(xs):
    return len(xs) == 0
```

### 256. sum_range (training)

Buggy:
```python
def sum_range_26(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_26(n):
    return sum(range(n + 1))
```

### 257. reverse (training)

Buggy:
```python
def reverse_10(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_10(xs):
    return xs[::-1]
```

### 258. minimum (training)

Buggy:
```python
def minimum_36(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_36(a, b):
    return a if a < b else b
```

### 259. factorial (training)

Buggy:
```python
def factorial_22(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_22(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 260. empty (training)

Buggy:
```python
def empty_19(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_19(xs):
    return len(xs) == 0
```

### 261. minimum (training)

Buggy:
```python
def minimum_31(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_31(a, b):
    return a if a < b else b
```

### 262. average (training)

Buggy:
```python
def average(xs):
    return sum(xs)
```

Reference fix:
```python
def average(xs):
    return sum(xs) / len(xs)
```

### 263. factorial (training)

Buggy:
```python
def factorial_26(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_26(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 264. first (training)

Buggy:
```python
def first_8(xs):
    return xs[1]
```

Reference fix:
```python
def first_8(xs):
    return xs[0]
```

### 265. positive (training)

Buggy:
```python
def positive_12(n):
    return n < 0
```

Reference fix:
```python
def positive_12(n):
    return n > 0
```

### 266. odd (training)

Buggy:
```python
def odd_18(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_18(n):
    return n % 2 == 1
```

### 267. maximum (training)

Buggy:
```python
def maximum_14(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_14(a, b):
    return a if a > b else b
```

### 268. maximum (training)

Buggy:
```python
def maximum_29(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_29(a, b):
    return a if a > b else b
```

### 269. contains (validation)

Buggy:
```python
def contains_10(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_10(xs, n):
    return n in xs
```

### 270. last (validation)

Buggy:
```python
def last_27(xs):
    return xs[0]
```

Reference fix:
```python
def last_27(xs):
    return xs[-1]
```

### 271. first (training)

Buggy:
```python
def first_24(xs):
    return xs[1]
```

Reference fix:
```python
def first_24(xs):
    return xs[0]
```

### 272. first (training)

Buggy:
```python
def first_19(xs):
    return xs[1]
```

Reference fix:
```python
def first_19(xs):
    return xs[0]
```

### 273. minimum (training)

Buggy:
```python
def minimum_27(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_27(a, b):
    return a if a < b else b
```

### 274. sum_range (training)

Buggy:
```python
def sum_range_14(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_14(n):
    return sum(range(n + 1))
```

### 275. sum_range (training)

Buggy:
```python
def sum_range_19(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_19(n):
    return sum(range(n + 1))
```

### 276. minimum (training)

Buggy:
```python
def minimum_1(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_1(a, b):
    return a if a < b else b
```

### 277. sum_range (training)

Buggy:
```python
def sum_range_28(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_28(n):
    return sum(range(n + 1))
```

### 278. sum_range (training)

Buggy:
```python
def sum_range_13(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_13(n):
    return sum(range(n + 1))
```

### 279. maximum (training)

Buggy:
```python
def maximum_23(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_23(a, b):
    return a if a > b else b
```

### 280. minimum (training)

Buggy:
```python
def minimum_16(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_16(a, b):
    return a if a < b else b
```

### 281. square (training)

Buggy:
```python
def square_22(n):
    return n * 2
```

Reference fix:
```python
def square_22(n):
    return n ** 2
```

### 282. first (training)

Buggy:
```python
def first_34(xs):
    return xs[1]
```

Reference fix:
```python
def first_34(xs):
    return xs[0]
```

### 283. odd (training)

Buggy:
```python
def odd_38(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_38(n):
    return n % 2 == 1
```

### 284. square (training)

Buggy:
```python
def square_21(n):
    return n * 2
```

Reference fix:
```python
def square_21(n):
    return n ** 2
```

### 285. square (training)

Buggy:
```python
def square_8(n):
    return n * 2
```

Reference fix:
```python
def square_8(n):
    return n ** 2
```

### 286. first (training)

Buggy:
```python
def first_25(xs):
    return xs[1]
```

Reference fix:
```python
def first_25(xs):
    return xs[0]
```

### 287. reverse (training)

Buggy:
```python
def reverse_29(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_29(xs):
    return xs[::-1]
```

### 288. first (training)

Buggy:
```python
def first_5(xs):
    return xs[1]
```

Reference fix:
```python
def first_5(xs):
    return xs[0]
```

### 289. positive (training)

Buggy:
```python
def positive_29(n):
    return n < 0
```

Reference fix:
```python
def positive_29(n):
    return n > 0
```

### 290. reverse (training)

Buggy:
```python
def reverse_18(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_18(xs):
    return xs[::-1]
```

### 291. maximum (training)

Buggy:
```python
def maximum_38(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_38(a, b):
    return a if a > b else b
```

### 292. positive (training)

Buggy:
```python
def positive_14(n):
    return n < 0
```

Reference fix:
```python
def positive_14(n):
    return n > 0
```

### 293. positive (training)

Buggy:
```python
def positive_28(n):
    return n < 0
```

Reference fix:
```python
def positive_28(n):
    return n > 0
```

### 294. add (training)

Buggy:
```python
def add_1(a, b):
    return a - b
```

Reference fix:
```python
def add_1(a, b):
    return a + b
```

### 295. odd (training)

Buggy:
```python
def odd_35(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_35(n):
    return n % 2 == 1
```

### 296. empty (training)

Buggy:
```python
def empty(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty(xs):
    return len(xs) == 0
```

### 297. sum_range (training)

Buggy:
```python
def sum_range_11(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_11(n):
    return sum(range(n + 1))
```

### 298. maximum (training)

Buggy:
```python
def maximum_8(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_8(a, b):
    return a if a > b else b
```

### 299. maximum (training)

Buggy:
```python
def maximum_22(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_22(a, b):
    return a if a > b else b
```

### 300. factorial (training)

Buggy:
```python
def factorial_4(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_4(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 301. factorial (training)

Buggy:
```python
def factorial_37(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_37(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 302. positive (training)

Buggy:
```python
def positive_27(n):
    return n < 0
```

Reference fix:
```python
def positive_27(n):
    return n > 0
```

### 303. contains (validation)

Buggy:
```python
def contains(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains(xs, n):
    return n in xs
```

### 304. first (training)

Buggy:
```python
def first_33(xs):
    return xs[1]
```

Reference fix:
```python
def first_33(xs):
    return xs[0]
```

### 305. multiply (validation)

Buggy:
```python
def multiply_15(a, b):
    return a + b
```

Reference fix:
```python
def multiply_15(a, b):
    return a * b
```

### 306. even (training)

Buggy:
```python
def even_2(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_2(n):
    return n % 2 == 0
```

### 307. minimum (training)

Buggy:
```python
def minimum_34(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_34(a, b):
    return a if a < b else b
```

### 308. positive (training)

Buggy:
```python
def positive_3(n):
    return n < 0
```

Reference fix:
```python
def positive_3(n):
    return n > 0
```

### 309. minimum (training)

Buggy:
```python
def minimum_19(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_19(a, b):
    return a if a < b else b
```

### 310. average (training)

Buggy:
```python
def average_30(xs):
    return sum(xs)
```

Reference fix:
```python
def average_30(xs):
    return sum(xs) / len(xs)
```

### 311. first (training)

Buggy:
```python
def first_13(xs):
    return xs[1]
```

Reference fix:
```python
def first_13(xs):
    return xs[0]
```

### 312. positive (training)

Buggy:
```python
def positive_38(n):
    return n < 0
```

Reference fix:
```python
def positive_38(n):
    return n > 0
```

### 313. odd (training)

Buggy:
```python
def odd_27(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_27(n):
    return n % 2 == 1
```

### 314. maximum (training)

Buggy:
```python
def maximum_9(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_9(a, b):
    return a if a > b else b
```

### 315. maximum (training)

Buggy:
```python
def maximum_18(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_18(a, b):
    return a if a > b else b
```

### 316. contains (validation)

Buggy:
```python
def contains_13(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_13(xs, n):
    return n in xs
```

### 317. minimum (training)

Buggy:
```python
def minimum_9(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_9(a, b):
    return a if a < b else b
```

### 318. factorial (training)

Buggy:
```python
def factorial_20(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_20(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 319. minimum (training)

Buggy:
```python
def minimum_6(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_6(a, b):
    return a if a < b else b
```

### 320. multiply (validation)

Buggy:
```python
def multiply_19(a, b):
    return a + b
```

Reference fix:
```python
def multiply_19(a, b):
    return a * b
```

### 321. square (training)

Buggy:
```python
def square_2(n):
    return n * 2
```

Reference fix:
```python
def square_2(n):
    return n ** 2
```

### 322. maximum (training)

Buggy:
```python
def maximum_11(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_11(a, b):
    return a if a > b else b
```

### 323. square (training)

Buggy:
```python
def square_3(n):
    return n * 2
```

Reference fix:
```python
def square_3(n):
    return n ** 2
```

### 324. positive (training)

Buggy:
```python
def positive_23(n):
    return n < 0
```

Reference fix:
```python
def positive_23(n):
    return n > 0
```

### 325. first (training)

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

### 326. multiply (validation)

Buggy:
```python
def multiply_26(a, b):
    return a + b
```

Reference fix:
```python
def multiply_26(a, b):
    return a * b
```

### 327. maximum (training)

Buggy:
```python
def maximum_24(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_24(a, b):
    return a if a > b else b
```

### 328. contains (validation)

Buggy:
```python
def contains_6(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_6(xs, n):
    return n in xs
```

### 329. average (training)

Buggy:
```python
def average_26(xs):
    return sum(xs)
```

Reference fix:
```python
def average_26(xs):
    return sum(xs) / len(xs)
```

### 330. sum_range (training)

Buggy:
```python
def sum_range_1(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_1(n):
    return sum(range(n + 1))
```

### 331. odd (training)

Buggy:
```python
def odd_20(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_20(n):
    return n % 2 == 1
```

### 332. minimum (training)

Buggy:
```python
def minimum_29(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_29(a, b):
    return a if a < b else b
```

### 333. contains (validation)

Buggy:
```python
def contains_32(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_32(xs, n):
    return n in xs
```

### 334. positive (training)

Buggy:
```python
def positive_17(n):
    return n < 0
```

Reference fix:
```python
def positive_17(n):
    return n > 0
```

### 335. empty (training)

Buggy:
```python
def empty_6(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_6(xs):
    return len(xs) == 0
```

### 336. odd (training)

Buggy:
```python
def odd_4(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_4(n):
    return n % 2 == 1
```

### 337. maximum (training)

Buggy:
```python
def maximum_19(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_19(a, b):
    return a if a > b else b
```

### 338. add (training)

Buggy:
```python
def add_11(a, b):
    return a - b
```

Reference fix:
```python
def add_11(a, b):
    return a + b
```

### 339. positive (training)

Buggy:
```python
def positive(n):
    return n < 0
```

Reference fix:
```python
def positive(n):
    return n > 0
```

### 340. reverse (training)

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

### 341. add (training)

Buggy:
```python
def add_31(a, b):
    return a - b
```

Reference fix:
```python
def add_31(a, b):
    return a + b
```

### 342. odd (training)

Buggy:
```python
def odd_37(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_37(n):
    return n % 2 == 1
```

### 343. contains (validation)

Buggy:
```python
def contains_24(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_24(xs, n):
    return n in xs
```

### 344. positive (training)

Buggy:
```python
def positive_2(n):
    return n < 0
```

Reference fix:
```python
def positive_2(n):
    return n > 0
```

### 345. last (validation)

Buggy:
```python
def last_31(xs):
    return xs[0]
```

Reference fix:
```python
def last_31(xs):
    return xs[-1]
```

### 346. first (training)

Buggy:
```python
def first_31(xs):
    return xs[1]
```

Reference fix:
```python
def first_31(xs):
    return xs[0]
```

### 347. add (training)

Buggy:
```python
def add_24(a, b):
    return a - b
```

Reference fix:
```python
def add_24(a, b):
    return a + b
```

### 348. sum_range (training)

Buggy:
```python
def sum_range_39(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_39(n):
    return sum(range(n + 1))
```

### 349. odd (training)

Buggy:
```python
def odd_9(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_9(n):
    return n % 2 == 1
```

### 350. minimum (training)

Buggy:
```python
def minimum_18(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_18(a, b):
    return a if a < b else b
```

### 351. contains (validation)

Buggy:
```python
def contains_19(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_19(xs, n):
    return n in xs
```

### 352. add (training)

Buggy:
```python
def add_20(a, b):
    return a - b
```

Reference fix:
```python
def add_20(a, b):
    return a + b
```

### 353. empty (training)

Buggy:
```python
def empty_28(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_28(xs):
    return len(xs) == 0
```

### 354. even (training)

Buggy:
```python
def even_32(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_32(n):
    return n % 2 == 0
```

### 355. square (training)

Buggy:
```python
def square_23(n):
    return n * 2
```

Reference fix:
```python
def square_23(n):
    return n ** 2
```

### 356. reverse (training)

Buggy:
```python
def reverse_39(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_39(xs):
    return xs[::-1]
```

### 357. factorial (training)

Buggy:
```python
def factorial_1(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_1(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 358. reverse (training)

Buggy:
```python
def reverse_34(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_34(xs):
    return xs[::-1]
```

### 359. average (training)

Buggy:
```python
def average_23(xs):
    return sum(xs)
```

Reference fix:
```python
def average_23(xs):
    return sum(xs) / len(xs)
```

### 360. multiply (validation)

Buggy:
```python
def multiply_28(a, b):
    return a + b
```

Reference fix:
```python
def multiply_28(a, b):
    return a * b
```

### 361. even (training)

Buggy:
```python
def even_5(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_5(n):
    return n % 2 == 0
```

### 362. square (training)

Buggy:
```python
def square_34(n):
    return n * 2
```

Reference fix:
```python
def square_34(n):
    return n ** 2
```

### 363. odd (training)

Buggy:
```python
def odd_6(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_6(n):
    return n % 2 == 1
```

### 364. multiply (validation)

Buggy:
```python
def multiply_18(a, b):
    return a + b
```

Reference fix:
```python
def multiply_18(a, b):
    return a * b
```

### 365. multiply (validation)

Buggy:
```python
def multiply_7(a, b):
    return a + b
```

Reference fix:
```python
def multiply_7(a, b):
    return a * b
```

### 366. last (validation)

Buggy:
```python
def last_5(xs):
    return xs[0]
```

Reference fix:
```python
def last_5(xs):
    return xs[-1]
```

### 367. multiply (validation)

Buggy:
```python
def multiply_21(a, b):
    return a + b
```

Reference fix:
```python
def multiply_21(a, b):
    return a * b
```

### 368. sum_range (training)

Buggy:
```python
def sum_range_7(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_7(n):
    return sum(range(n + 1))
```

### 369. factorial (training)

Buggy:
```python
def factorial_7(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_7(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 370. multiply (validation)

Buggy:
```python
def multiply_16(a, b):
    return a + b
```

Reference fix:
```python
def multiply_16(a, b):
    return a * b
```

### 371. minimum (training)

Buggy:
```python
def minimum_13(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_13(a, b):
    return a if a < b else b
```

### 372. average (training)

Buggy:
```python
def average_38(xs):
    return sum(xs)
```

Reference fix:
```python
def average_38(xs):
    return sum(xs) / len(xs)
```

### 373. positive (training)

Buggy:
```python
def positive_16(n):
    return n < 0
```

Reference fix:
```python
def positive_16(n):
    return n > 0
```

### 374. square (training)

Buggy:
```python
def square_25(n):
    return n * 2
```

Reference fix:
```python
def square_25(n):
    return n ** 2
```

### 375. even (training)

Buggy:
```python
def even_26(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_26(n):
    return n % 2 == 0
```

### 376. factorial (training)

Buggy:
```python
def factorial_38(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_38(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 377. square (training)

Buggy:
```python
def square_16(n):
    return n * 2
```

Reference fix:
```python
def square_16(n):
    return n ** 2
```

### 378. even (training)

Buggy:
```python
def even_36(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_36(n):
    return n % 2 == 0
```

### 379. multiply (validation)

Buggy:
```python
def multiply_20(a, b):
    return a + b
```

Reference fix:
```python
def multiply_20(a, b):
    return a * b
```

### 380. empty (training)

Buggy:
```python
def empty_16(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_16(xs):
    return len(xs) == 0
```

### 381. multiply (validation)

Buggy:
```python
def multiply(a, b):
    return a + b
```

Reference fix:
```python
def multiply(a, b):
    return a * b
```

### 382. contains (validation)

Buggy:
```python
def contains_27(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_27(xs, n):
    return n in xs
```

### 383. even (training)

Buggy:
```python
def even_29(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_29(n):
    return n % 2 == 0
```

### 384. empty (training)

Buggy:
```python
def empty_36(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_36(xs):
    return len(xs) == 0
```

### 385. multiply (validation)

Buggy:
```python
def multiply_6(a, b):
    return a + b
```

Reference fix:
```python
def multiply_6(a, b):
    return a * b
```

### 386. last (validation)

Buggy:
```python
def last_18(xs):
    return xs[0]
```

Reference fix:
```python
def last_18(xs):
    return xs[-1]
```

### 387. contains (validation)

Buggy:
```python
def contains_34(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_34(xs, n):
    return n in xs
```

### 388. sum_range (training)

Buggy:
```python
def sum_range_15(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_15(n):
    return sum(range(n + 1))
```

### 389. multiply (validation)

Buggy:
```python
def multiply_39(a, b):
    return a + b
```

Reference fix:
```python
def multiply_39(a, b):
    return a * b
```

### 390. add (training)

Buggy:
```python
def add_12(a, b):
    return a - b
```

Reference fix:
```python
def add_12(a, b):
    return a + b
```

### 391. even (training)

Buggy:
```python
def even_6(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_6(n):
    return n % 2 == 0
```

### 392. factorial (training)

Buggy:
```python
def factorial_31(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_31(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 393. sum_range (training)

Buggy:
```python
def sum_range_10(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_10(n):
    return sum(range(n + 1))
```

### 394. minimum (training)

Buggy:
```python
def minimum_10(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_10(a, b):
    return a if a < b else b
```

### 395. minimum (training)

Buggy:
```python
def minimum_14(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_14(a, b):
    return a if a < b else b
```

### 396. average (training)

Buggy:
```python
def average_5(xs):
    return sum(xs)
```

Reference fix:
```python
def average_5(xs):
    return sum(xs) / len(xs)
```

### 397. average (training)

Buggy:
```python
def average_21(xs):
    return sum(xs)
```

Reference fix:
```python
def average_21(xs):
    return sum(xs) / len(xs)
```

### 398. square (training)

Buggy:
```python
def square_15(n):
    return n * 2
```

Reference fix:
```python
def square_15(n):
    return n ** 2
```

### 399. even (training)

Buggy:
```python
def even_39(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_39(n):
    return n % 2 == 0
```

### 400. even (training)

Buggy:
```python
def even_10(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_10(n):
    return n % 2 == 0
```

### 401. average (training)

Buggy:
```python
def average_15(xs):
    return sum(xs)
```

Reference fix:
```python
def average_15(xs):
    return sum(xs) / len(xs)
```

### 402. reverse (training)

Buggy:
```python
def reverse_21(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_21(xs):
    return xs[::-1]
```

### 403. average (training)

Buggy:
```python
def average_6(xs):
    return sum(xs)
```

Reference fix:
```python
def average_6(xs):
    return sum(xs) / len(xs)
```

### 404. square (training)

Buggy:
```python
def square_27(n):
    return n * 2
```

Reference fix:
```python
def square_27(n):
    return n ** 2
```

### 405. sum_range (training)

Buggy:
```python
def sum_range_9(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_9(n):
    return sum(range(n + 1))
```

### 406. empty (training)

Buggy:
```python
def empty_15(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_15(xs):
    return len(xs) == 0
```

### 407. last (validation)

Buggy:
```python
def last_17(xs):
    return xs[0]
```

Reference fix:
```python
def last_17(xs):
    return xs[-1]
```

### 408. even (training)

Buggy:
```python
def even_8(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_8(n):
    return n % 2 == 0
```

### 409. multiply (validation)

Buggy:
```python
def multiply_11(a, b):
    return a + b
```

Reference fix:
```python
def multiply_11(a, b):
    return a * b
```

### 410. multiply (validation)

Buggy:
```python
def multiply_1(a, b):
    return a + b
```

Reference fix:
```python
def multiply_1(a, b):
    return a * b
```

### 411. first (training)

Buggy:
```python
def first_4(xs):
    return xs[1]
```

Reference fix:
```python
def first_4(xs):
    return xs[0]
```

### 412. odd (training)

Buggy:
```python
def odd_2(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_2(n):
    return n % 2 == 1
```

### 413. maximum (training)

Buggy:
```python
def maximum_15(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_15(a, b):
    return a if a > b else b
```

### 414. maximum (training)

Buggy:
```python
def maximum_12(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_12(a, b):
    return a if a > b else b
```

### 415. empty (training)

Buggy:
```python
def empty_21(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_21(xs):
    return len(xs) == 0
```

### 416. empty (training)

Buggy:
```python
def empty_35(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_35(xs):
    return len(xs) == 0
```

### 417. square (training)

Buggy:
```python
def square_17(n):
    return n * 2
```

Reference fix:
```python
def square_17(n):
    return n ** 2
```

### 418. last (validation)

Buggy:
```python
def last_29(xs):
    return xs[0]
```

Reference fix:
```python
def last_29(xs):
    return xs[-1]
```

### 419. positive (training)

Buggy:
```python
def positive_9(n):
    return n < 0
```

Reference fix:
```python
def positive_9(n):
    return n > 0
```

### 420. minimum (training)

Buggy:
```python
def minimum_22(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_22(a, b):
    return a if a < b else b
```

### 421. contains (validation)

Buggy:
```python
def contains_9(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_9(xs, n):
    return n in xs
```

### 422. first (training)

Buggy:
```python
def first_1(xs):
    return xs[1]
```

Reference fix:
```python
def first_1(xs):
    return xs[0]
```

### 423. odd (training)

Buggy:
```python
def odd_23(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_23(n):
    return n % 2 == 1
```

### 424. multiply (validation)

Buggy:
```python
def multiply_31(a, b):
    return a + b
```

Reference fix:
```python
def multiply_31(a, b):
    return a * b
```

### 425. reverse (training)

Buggy:
```python
def reverse_9(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_9(xs):
    return xs[::-1]
```

### 426. factorial (training)

Buggy:
```python
def factorial_16(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_16(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 427. factorial (training)

Buggy:
```python
def factorial_34(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_34(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 428. multiply (validation)

Buggy:
```python
def multiply_35(a, b):
    return a + b
```

Reference fix:
```python
def multiply_35(a, b):
    return a * b
```

### 429. average (training)

Buggy:
```python
def average_27(xs):
    return sum(xs)
```

Reference fix:
```python
def average_27(xs):
    return sum(xs) / len(xs)
```

### 430. average (training)

Buggy:
```python
def average_18(xs):
    return sum(xs)
```

Reference fix:
```python
def average_18(xs):
    return sum(xs) / len(xs)
```

### 431. add (training)

Buggy:
```python
def add_2(a, b):
    return a - b
```

Reference fix:
```python
def add_2(a, b):
    return a + b
```

### 432. factorial (training)

Buggy:
```python
def factorial_36(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_36(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 433. positive (training)

Buggy:
```python
def positive_36(n):
    return n < 0
```

Reference fix:
```python
def positive_36(n):
    return n > 0
```

### 434. empty (training)

Buggy:
```python
def empty_14(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_14(xs):
    return len(xs) == 0
```

### 435. multiply (validation)

Buggy:
```python
def multiply_37(a, b):
    return a + b
```

Reference fix:
```python
def multiply_37(a, b):
    return a * b
```

### 436. empty (training)

Buggy:
```python
def empty_4(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_4(xs):
    return len(xs) == 0
```

### 437. maximum (training)

Buggy:
```python
def maximum_3(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_3(a, b):
    return a if a > b else b
```

### 438. maximum (training)

Buggy:
```python
def maximum_21(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_21(a, b):
    return a if a > b else b
```

### 439. factorial (training)

Buggy:
```python
def factorial_10(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_10(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 440. reverse (training)

Buggy:
```python
def reverse_13(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_13(xs):
    return xs[::-1]
```

### 441. empty (training)

Buggy:
```python
def empty_39(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_39(xs):
    return len(xs) == 0
```

### 442. last (validation)

Buggy:
```python
def last_35(xs):
    return xs[0]
```

Reference fix:
```python
def last_35(xs):
    return xs[-1]
```

### 443. first (training)

Buggy:
```python
def first_6(xs):
    return xs[1]
```

Reference fix:
```python
def first_6(xs):
    return xs[0]
```

### 444. sum_range (training)

Buggy:
```python
def sum_range_21(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_21(n):
    return sum(range(n + 1))
```

### 445. even (training)

Buggy:
```python
def even_31(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_31(n):
    return n % 2 == 0
```

### 446. odd (training)

Buggy:
```python
def odd_24(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_24(n):
    return n % 2 == 1
```

### 447. empty (training)

Buggy:
```python
def empty_20(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_20(xs):
    return len(xs) == 0
```

### 448. positive (training)

Buggy:
```python
def positive_24(n):
    return n < 0
```

Reference fix:
```python
def positive_24(n):
    return n > 0
```

### 449. contains (validation)

Buggy:
```python
def contains_2(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_2(xs, n):
    return n in xs
```

### 450. minimum (training)

Buggy:
```python
def minimum_23(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_23(a, b):
    return a if a < b else b
```

### 451. even (training)

Buggy:
```python
def even_4(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_4(n):
    return n % 2 == 0
```

### 452. average (training)

Buggy:
```python
def average_7(xs):
    return sum(xs)
```

Reference fix:
```python
def average_7(xs):
    return sum(xs) / len(xs)
```

### 453. add (training)

Buggy:
```python
def add_5(a, b):
    return a - b
```

Reference fix:
```python
def add_5(a, b):
    return a + b
```

### 454. last (validation)

Buggy:
```python
def last_22(xs):
    return xs[0]
```

Reference fix:
```python
def last_22(xs):
    return xs[-1]
```

### 455. minimum (training)

Buggy:
```python
def minimum_33(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_33(a, b):
    return a if a < b else b
```

### 456. contains (validation)

Buggy:
```python
def contains_17(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_17(xs, n):
    return n in xs
```

### 457. add (training)

Buggy:
```python
def add_26(a, b):
    return a - b
```

Reference fix:
```python
def add_26(a, b):
    return a + b
```

### 458. positive (training)

Buggy:
```python
def positive_21(n):
    return n < 0
```

Reference fix:
```python
def positive_21(n):
    return n > 0
```

### 459. positive (training)

Buggy:
```python
def positive_34(n):
    return n < 0
```

Reference fix:
```python
def positive_34(n):
    return n > 0
```

### 460. empty (training)

Buggy:
```python
def empty_34(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_34(xs):
    return len(xs) == 0
```

### 461. maximum (training)

Buggy:
```python
def maximum_30(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_30(a, b):
    return a if a > b else b
```

### 462. square (training)

Buggy:
```python
def square_6(n):
    return n * 2
```

Reference fix:
```python
def square_6(n):
    return n ** 2
```

### 463. contains (validation)

Buggy:
```python
def contains_38(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_38(xs, n):
    return n in xs
```

### 464. first (training)

Buggy:
```python
def first_15(xs):
    return xs[1]
```

Reference fix:
```python
def first_15(xs):
    return xs[0]
```

### 465. last (validation)

Buggy:
```python
def last_4(xs):
    return xs[0]
```

Reference fix:
```python
def last_4(xs):
    return xs[-1]
```

### 466. average (training)

Buggy:
```python
def average_37(xs):
    return sum(xs)
```

Reference fix:
```python
def average_37(xs):
    return sum(xs) / len(xs)
```

### 467. odd (training)

Buggy:
```python
def odd_3(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_3(n):
    return n % 2 == 1
```

### 468. minimum (training)

Buggy:
```python
def minimum_17(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_17(a, b):
    return a if a < b else b
```

### 469. empty (training)

Buggy:
```python
def empty_10(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_10(xs):
    return len(xs) == 0
```

### 470. sum_range (training)

Buggy:
```python
def sum_range_24(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_24(n):
    return sum(range(n + 1))
```

### 471. add (training)

Buggy:
```python
def add_32(a, b):
    return a - b
```

Reference fix:
```python
def add_32(a, b):
    return a + b
```

### 472. odd (training)

Buggy:
```python
def odd_32(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_32(n):
    return n % 2 == 1
```

### 473. contains (validation)

Buggy:
```python
def contains_20(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_20(xs, n):
    return n in xs
```

### 474. factorial (training)

Buggy:
```python
def factorial_35(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_35(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 475. multiply (validation)

Buggy:
```python
def multiply_17(a, b):
    return a + b
```

Reference fix:
```python
def multiply_17(a, b):
    return a * b
```

### 476. odd (training)

Buggy:
```python
def odd_8(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_8(n):
    return n % 2 == 1
```

### 477. add (training)

Buggy:
```python
def add_14(a, b):
    return a - b
```

Reference fix:
```python
def add_14(a, b):
    return a + b
```

### 478. sum_range (training)

Buggy:
```python
def sum_range_3(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_3(n):
    return sum(range(n + 1))
```

### 479. sum_range (training)

Buggy:
```python
def sum_range(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range(n):
    return sum(range(n + 1))
```

### 480. maximum (training)

Buggy:
```python
def maximum_10(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_10(a, b):
    return a if a > b else b
```

### 481. positive (training)

Buggy:
```python
def positive_7(n):
    return n < 0
```

Reference fix:
```python
def positive_7(n):
    return n > 0
```

### 482. first (training)

Buggy:
```python
def first_3(xs):
    return xs[1]
```

Reference fix:
```python
def first_3(xs):
    return xs[0]
```

### 483. factorial (training)

Buggy:
```python
def factorial_29(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_29(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 484. first (training)

Buggy:
```python
def first_2(xs):
    return xs[1]
```

Reference fix:
```python
def first_2(xs):
    return xs[0]
```

### 485. first (training)

Buggy:
```python
def first_37(xs):
    return xs[1]
```

Reference fix:
```python
def first_37(xs):
    return xs[0]
```

### 486. multiply (validation)

Buggy:
```python
def multiply_13(a, b):
    return a + b
```

Reference fix:
```python
def multiply_13(a, b):
    return a * b
```

### 487. multiply (validation)

Buggy:
```python
def multiply_25(a, b):
    return a + b
```

Reference fix:
```python
def multiply_25(a, b):
    return a * b
```

### 488. square (training)

Buggy:
```python
def square_9(n):
    return n * 2
```

Reference fix:
```python
def square_9(n):
    return n ** 2
```

### 489. reverse (training)

Buggy:
```python
def reverse_25(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_25(xs):
    return xs[::-1]
```

### 490. square (training)

Buggy:
```python
def square_35(n):
    return n * 2
```

Reference fix:
```python
def square_35(n):
    return n ** 2
```

### 491. contains (validation)

Buggy:
```python
def contains_1(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_1(xs, n):
    return n in xs
```

### 492. factorial (training)

Buggy:
```python
def factorial_9(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_9(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 493. contains (validation)

Buggy:
```python
def contains_8(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_8(xs, n):
    return n in xs
```

### 494. reverse (training)

Buggy:
```python
def reverse_5(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_5(xs):
    return xs[::-1]
```

### 495. maximum (training)

Buggy:
```python
def maximum_26(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_26(a, b):
    return a if a > b else b
```

### 496. factorial (training)

Buggy:
```python
def factorial(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 497. contains (validation)

Buggy:
```python
def contains_36(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_36(xs, n):
    return n in xs
```

### 498. empty (training)

Buggy:
```python
def empty_27(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_27(xs):
    return len(xs) == 0
```

### 499. square (training)

Buggy:
```python
def square_32(n):
    return n * 2
```

Reference fix:
```python
def square_32(n):
    return n ** 2
```

### 500. add (training)

Buggy:
```python
def add_38(a, b):
    return a - b
```

Reference fix:
```python
def add_38(a, b):
    return a + b
```

### 501. positive (training)

Buggy:
```python
def positive_8(n):
    return n < 0
```

Reference fix:
```python
def positive_8(n):
    return n > 0
```

### 502. contains (validation)

Buggy:
```python
def contains_4(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_4(xs, n):
    return n in xs
```

### 503. last (validation)

Buggy:
```python
def last_2(xs):
    return xs[0]
```

Reference fix:
```python
def last_2(xs):
    return xs[-1]
```

### 504. empty (training)

Buggy:
```python
def empty_8(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_8(xs):
    return len(xs) == 0
```

### 505. reverse (training)

Buggy:
```python
def reverse_31(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_31(xs):
    return xs[::-1]
```

### 506. sum_range (training)

Buggy:
```python
def sum_range_38(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_38(n):
    return sum(range(n + 1))
```

### 507. last (validation)

Buggy:
```python
def last_21(xs):
    return xs[0]
```

Reference fix:
```python
def last_21(xs):
    return xs[-1]
```

### 508. maximum (training)

Buggy:
```python
def maximum_31(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_31(a, b):
    return a if a > b else b
```

### 509. odd (training)

Buggy:
```python
def odd_19(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_19(n):
    return n % 2 == 1
```

### 510. minimum (training)

Buggy:
```python
def minimum_7(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_7(a, b):
    return a if a < b else b
```

### 511. multiply (validation)

Buggy:
```python
def multiply_30(a, b):
    return a + b
```

Reference fix:
```python
def multiply_30(a, b):
    return a * b
```

### 512. maximum (training)

Buggy:
```python
def maximum_25(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_25(a, b):
    return a if a > b else b
```

### 513. last (validation)

Buggy:
```python
def last_28(xs):
    return xs[0]
```

Reference fix:
```python
def last_28(xs):
    return xs[-1]
```

### 514. reverse (training)

Buggy:
```python
def reverse_22(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_22(xs):
    return xs[::-1]
```

### 515. positive (training)

Buggy:
```python
def positive_32(n):
    return n < 0
```

Reference fix:
```python
def positive_32(n):
    return n > 0
```

### 516. empty (training)

Buggy:
```python
def empty_24(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_24(xs):
    return len(xs) == 0
```

### 517. factorial (training)

Buggy:
```python
def factorial_3(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_3(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 518. contains (validation)

Buggy:
```python
def contains_18(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_18(xs, n):
    return n in xs
```

### 519. reverse (training)

Buggy:
```python
def reverse_1(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_1(xs):
    return xs[::-1]
```

### 520. square (training)

Buggy:
```python
def square_5(n):
    return n * 2
```

Reference fix:
```python
def square_5(n):
    return n ** 2
```

### 521. contains (validation)

Buggy:
```python
def contains_30(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_30(xs, n):
    return n in xs
```

### 522. average (training)

Buggy:
```python
def average_39(xs):
    return sum(xs)
```

Reference fix:
```python
def average_39(xs):
    return sum(xs) / len(xs)
```

### 523. average (training)

Buggy:
```python
def average_9(xs):
    return sum(xs)
```

Reference fix:
```python
def average_9(xs):
    return sum(xs) / len(xs)
```

### 524. contains (validation)

Buggy:
```python
def contains_12(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_12(xs, n):
    return n in xs
```

### 525. last (validation)

Buggy:
```python
def last_37(xs):
    return xs[0]
```

Reference fix:
```python
def last_37(xs):
    return xs[-1]
```

### 526. factorial (training)

Buggy:
```python
def factorial_11(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_11(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 527. odd (training)

Buggy:
```python
def odd_17(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_17(n):
    return n % 2 == 1
```

### 528. square (training)

Buggy:
```python
def square_39(n):
    return n * 2
```

Reference fix:
```python
def square_39(n):
    return n ** 2
```

### 529. add (training)

Buggy:
```python
def add_30(a, b):
    return a - b
```

Reference fix:
```python
def add_30(a, b):
    return a + b
```

### 530. add (training)

Buggy:
```python
def add_34(a, b):
    return a - b
```

Reference fix:
```python
def add_34(a, b):
    return a + b
```

### 531. sum_range (training)

Buggy:
```python
def sum_range_25(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_25(n):
    return sum(range(n + 1))
```

### 532. average (training)

Buggy:
```python
def average_10(xs):
    return sum(xs)
```

Reference fix:
```python
def average_10(xs):
    return sum(xs) / len(xs)
```

### 533. average (training)

Buggy:
```python
def average_11(xs):
    return sum(xs)
```

Reference fix:
```python
def average_11(xs):
    return sum(xs) / len(xs)
```

### 534. first (training)

Buggy:
```python
def first_36(xs):
    return xs[1]
```

Reference fix:
```python
def first_36(xs):
    return xs[0]
```

### 535. last (validation)

Buggy:
```python
def last_32(xs):
    return xs[0]
```

Reference fix:
```python
def last_32(xs):
    return xs[-1]
```

### 536. sum_range (training)

Buggy:
```python
def sum_range_8(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_8(n):
    return sum(range(n + 1))
```

### 537. last (validation)

Buggy:
```python
def last_13(xs):
    return xs[0]
```

Reference fix:
```python
def last_13(xs):
    return xs[-1]
```

### 538. positive (training)

Buggy:
```python
def positive_25(n):
    return n < 0
```

Reference fix:
```python
def positive_25(n):
    return n > 0
```

### 539. empty (training)

Buggy:
```python
def empty_1(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_1(xs):
    return len(xs) == 0
```

### 540. first (training)

Buggy:
```python
def first_21(xs):
    return xs[1]
```

Reference fix:
```python
def first_21(xs):
    return xs[0]
```

### 541. contains (validation)

Buggy:
```python
def contains_3(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_3(xs, n):
    return n in xs
```

### 542. contains (validation)

Buggy:
```python
def contains_29(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_29(xs, n):
    return n in xs
```

### 543. multiply (validation)

Buggy:
```python
def multiply_8(a, b):
    return a + b
```

Reference fix:
```python
def multiply_8(a, b):
    return a * b
```

### 544. multiply (validation)

Buggy:
```python
def multiply_38(a, b):
    return a + b
```

Reference fix:
```python
def multiply_38(a, b):
    return a * b
```

### 545. odd (training)

Buggy:
```python
def odd_29(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_29(n):
    return n % 2 == 1
```

### 546. odd (training)

Buggy:
```python
def odd_12(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_12(n):
    return n % 2 == 1
```

### 547. average (training)

Buggy:
```python
def average_34(xs):
    return sum(xs)
```

Reference fix:
```python
def average_34(xs):
    return sum(xs) / len(xs)
```

### 548. odd (training)

Buggy:
```python
def odd_31(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_31(n):
    return n % 2 == 1
```

### 549. first (training)

Buggy:
```python
def first_26(xs):
    return xs[1]
```

Reference fix:
```python
def first_26(xs):
    return xs[0]
```

### 550. square (training)

Buggy:
```python
def square_29(n):
    return n * 2
```

Reference fix:
```python
def square_29(n):
    return n ** 2
```

### 551. multiply (validation)

Buggy:
```python
def multiply_5(a, b):
    return a + b
```

Reference fix:
```python
def multiply_5(a, b):
    return a * b
```

### 552. positive (training)

Buggy:
```python
def positive_31(n):
    return n < 0
```

Reference fix:
```python
def positive_31(n):
    return n > 0
```

### 553. even (training)

Buggy:
```python
def even_17(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_17(n):
    return n % 2 == 0
```

### 554. minimum (training)

Buggy:
```python
def minimum_2(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_2(a, b):
    return a if a < b else b
```

### 555. even (training)

Buggy:
```python
def even_16(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_16(n):
    return n % 2 == 0
```

### 556. sum_range (training)

Buggy:
```python
def sum_range_27(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_27(n):
    return sum(range(n + 1))
```

### 557. odd (training)

Buggy:
```python
def odd_34(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_34(n):
    return n % 2 == 1
```

### 558. multiply (validation)

Buggy:
```python
def multiply_10(a, b):
    return a + b
```

Reference fix:
```python
def multiply_10(a, b):
    return a * b
```

### 559. minimum (training)

Buggy:
```python
def minimum_3(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_3(a, b):
    return a if a < b else b
```

### 560. average (training)

Buggy:
```python
def average_28(xs):
    return sum(xs)
```

Reference fix:
```python
def average_28(xs):
    return sum(xs) / len(xs)
```

### 561. even (training)

Buggy:
```python
def even_34(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_34(n):
    return n % 2 == 0
```

### 562. sum_range (training)

Buggy:
```python
def sum_range_17(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_17(n):
    return sum(range(n + 1))
```

### 563. minimum (training)

Buggy:
```python
def minimum_12(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_12(a, b):
    return a if a < b else b
```

### 564. even (training)

Buggy:
```python
def even_24(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_24(n):
    return n % 2 == 0
```

### 565. odd (training)

Buggy:
```python
def odd_36(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_36(n):
    return n % 2 == 1
```

### 566. add (training)

Buggy:
```python
def add_28(a, b):
    return a - b
```

Reference fix:
```python
def add_28(a, b):
    return a + b
```

### 567. square (training)

Buggy:
```python
def square_33(n):
    return n * 2
```

Reference fix:
```python
def square_33(n):
    return n ** 2
```

### 568. reverse (training)

Buggy:
```python
def reverse_7(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_7(xs):
    return xs[::-1]
```

### 569. average (training)

Buggy:
```python
def average_36(xs):
    return sum(xs)
```

Reference fix:
```python
def average_36(xs):
    return sum(xs) / len(xs)
```

### 570. empty (training)

Buggy:
```python
def empty_26(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_26(xs):
    return len(xs) == 0
```

### 571. reverse (training)

Buggy:
```python
def reverse_15(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_15(xs):
    return xs[::-1]
```

### 572. sum_range (training)

Buggy:
```python
def sum_range_33(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_33(n):
    return sum(range(n + 1))
```

### 573. odd (training)

Buggy:
```python
def odd_33(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_33(n):
    return n % 2 == 1
```

### 574. even (training)

Buggy:
```python
def even_14(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_14(n):
    return n % 2 == 0
```

### 575. add (training)

Buggy:
```python
def add_3(a, b):
    return a - b
```

Reference fix:
```python
def add_3(a, b):
    return a + b
```

### 576. add (training)

Buggy:
```python
def add_19(a, b):
    return a - b
```

Reference fix:
```python
def add_19(a, b):
    return a + b
```

### 577. reverse (training)

Buggy:
```python
def reverse_6(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_6(xs):
    return xs[::-1]
```

### 578. add (training)

Buggy:
```python
def add_13(a, b):
    return a - b
```

Reference fix:
```python
def add_13(a, b):
    return a + b
```

### 579. square (training)

Buggy:
```python
def square_24(n):
    return n * 2
```

Reference fix:
```python
def square_24(n):
    return n ** 2
```

### 580. average (training)

Buggy:
```python
def average_8(xs):
    return sum(xs)
```

Reference fix:
```python
def average_8(xs):
    return sum(xs) / len(xs)
```

### 581. average (training)

Buggy:
```python
def average_2(xs):
    return sum(xs)
```

Reference fix:
```python
def average_2(xs):
    return sum(xs) / len(xs)
```

### 582. last (validation)

Buggy:
```python
def last_23(xs):
    return xs[0]
```

Reference fix:
```python
def last_23(xs):
    return xs[-1]
```

### 583. average (training)

Buggy:
```python
def average_24(xs):
    return sum(xs)
```

Reference fix:
```python
def average_24(xs):
    return sum(xs) / len(xs)
```

### 584. last (validation)

Buggy:
```python
def last_1(xs):
    return xs[0]
```

Reference fix:
```python
def last_1(xs):
    return xs[-1]
```

### 585. maximum (training)

Buggy:
```python
def maximum_16(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_16(a, b):
    return a if a > b else b
```

### 586. even (training)

Buggy:
```python
def even_33(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_33(n):
    return n % 2 == 0
```

### 587. sum_range (training)

Buggy:
```python
def sum_range_6(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_6(n):
    return sum(range(n + 1))
```

### 588. sum_range (training)

Buggy:
```python
def sum_range_31(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_31(n):
    return sum(range(n + 1))
```

### 589. reverse (training)

Buggy:
```python
def reverse_36(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_36(xs):
    return xs[::-1]
```

### 590. add (training)

Buggy:
```python
def add_10(a, b):
    return a - b
```

Reference fix:
```python
def add_10(a, b):
    return a + b
```

### 591. maximum (training)

Buggy:
```python
def maximum_20(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_20(a, b):
    return a if a > b else b
```

### 592. contains (validation)

Buggy:
```python
def contains_5(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_5(xs, n):
    return n in xs
```

### 593. last (validation)

Buggy:
```python
def last(xs):
    return xs[0]
```

Reference fix:
```python
def last(xs):
    return xs[-1]
```

### 594. add (training)

Buggy:
```python
def add_27(a, b):
    return a - b
```

Reference fix:
```python
def add_27(a, b):
    return a + b
```

### 595. first (training)

Buggy:
```python
def first_7(xs):
    return xs[1]
```

Reference fix:
```python
def first_7(xs):
    return xs[0]
```

### 596. empty (training)

Buggy:
```python
def empty_29(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_29(xs):
    return len(xs) == 0
```

### 597. square (training)

Buggy:
```python
def square_30(n):
    return n * 2
```

Reference fix:
```python
def square_30(n):
    return n ** 2
```

### 598. sum_range (training)

Buggy:
```python
def sum_range_4(n):
    return sum(range(n))
```

Reference fix:
```python
def sum_range_4(n):
    return sum(range(n + 1))
```

### 599. odd (training)

Buggy:
```python
def odd_30(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_30(n):
    return n % 2 == 1
```

### 600. minimum (training)

Buggy:
```python
def minimum_32(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_32(a, b):
    return a if a < b else b
```

### 601. add (training)

Buggy:
```python
def add_7(a, b):
    return a - b
```

Reference fix:
```python
def add_7(a, b):
    return a + b
```

### 602. last (validation)

Buggy:
```python
def last_19(xs):
    return xs[0]
```

Reference fix:
```python
def last_19(xs):
    return xs[-1]
```

### 603. add (training)

Buggy:
```python
def add_29(a, b):
    return a - b
```

Reference fix:
```python
def add_29(a, b):
    return a + b
```

### 604. last (validation)

Buggy:
```python
def last_14(xs):
    return xs[0]
```

Reference fix:
```python
def last_14(xs):
    return xs[-1]
```

### 605. average (training)

Buggy:
```python
def average_33(xs):
    return sum(xs)
```

Reference fix:
```python
def average_33(xs):
    return sum(xs) / len(xs)
```

### 606. minimum (training)

Buggy:
```python
def minimum_24(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_24(a, b):
    return a if a < b else b
```

### 607. even (training)

Buggy:
```python
def even_20(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_20(n):
    return n % 2 == 0
```

### 608. first (training)

Buggy:
```python
def first_39(xs):
    return xs[1]
```

Reference fix:
```python
def first_39(xs):
    return xs[0]
```

### 609. average (training)

Buggy:
```python
def average_13(xs):
    return sum(xs)
```

Reference fix:
```python
def average_13(xs):
    return sum(xs) / len(xs)
```

### 610. minimum (training)

Buggy:
```python
def minimum_28(a, b):
    return a if a > b else b
```

Reference fix:
```python
def minimum_28(a, b):
    return a if a < b else b
```

### 611. average (training)

Buggy:
```python
def average_29(xs):
    return sum(xs)
```

Reference fix:
```python
def average_29(xs):
    return sum(xs) / len(xs)
```

### 612. reverse (training)

Buggy:
```python
def reverse_3(xs):
    return xs[::1]
```

Reference fix:
```python
def reverse_3(xs):
    return xs[::-1]
```

### 613. factorial (training)

Buggy:
```python
def factorial_6(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_6(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 614. maximum (training)

Buggy:
```python
def maximum_4(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_4(a, b):
    return a if a > b else b
```

### 615. average (training)

Buggy:
```python
def average_3(xs):
    return sum(xs)
```

Reference fix:
```python
def average_3(xs):
    return sum(xs) / len(xs)
```

### 616. square (training)

Buggy:
```python
def square_19(n):
    return n * 2
```

Reference fix:
```python
def square_19(n):
    return n ** 2
```

### 617. even (training)

Buggy:
```python
def even_25(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_25(n):
    return n % 2 == 0
```

### 618. multiply (validation)

Buggy:
```python
def multiply_29(a, b):
    return a + b
```

Reference fix:
```python
def multiply_29(a, b):
    return a * b
```

### 619. average (training)

Buggy:
```python
def average_32(xs):
    return sum(xs)
```

Reference fix:
```python
def average_32(xs):
    return sum(xs) / len(xs)
```

### 620. even (training)

Buggy:
```python
def even_3(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_3(n):
    return n % 2 == 0
```

### 621. contains (validation)

Buggy:
```python
def contains_14(xs, n):
    return n not in xs
```

Reference fix:
```python
def contains_14(xs, n):
    return n in xs
```

### 622. factorial (training)

Buggy:
```python
def factorial_27(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_27(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 623. average (training)

Buggy:
```python
def average_16(xs):
    return sum(xs)
```

Reference fix:
```python
def average_16(xs):
    return sum(xs) / len(xs)
```

### 624. positive (training)

Buggy:
```python
def positive_37(n):
    return n < 0
```

Reference fix:
```python
def positive_37(n):
    return n > 0
```

### 625. even (training)

Buggy:
```python
def even_38(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_38(n):
    return n % 2 == 0
```

### 626. even (training)

Buggy:
```python
def even_23(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_23(n):
    return n % 2 == 0
```

### 627. last (validation)

Buggy:
```python
def last_15(xs):
    return xs[0]
```

Reference fix:
```python
def last_15(xs):
    return xs[-1]
```

### 628. factorial (training)

Buggy:
```python
def factorial_30(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_30(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 629. factorial (training)

Buggy:
```python
def factorial_32(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_32(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 630. multiply (validation)

Buggy:
```python
def multiply_32(a, b):
    return a + b
```

Reference fix:
```python
def multiply_32(a, b):
    return a * b
```

### 631. average (training)

Buggy:
```python
def average_4(xs):
    return sum(xs)
```

Reference fix:
```python
def average_4(xs):
    return sum(xs) / len(xs)
```

### 632. last (validation)

Buggy:
```python
def last_9(xs):
    return xs[0]
```

Reference fix:
```python
def last_9(xs):
    return xs[-1]
```

### 633. empty (training)

Buggy:
```python
def empty_38(xs):
    return len(xs) > 0
```

Reference fix:
```python
def empty_38(xs):
    return len(xs) == 0
```

### 634. last (validation)

Buggy:
```python
def last_24(xs):
    return xs[0]
```

Reference fix:
```python
def last_24(xs):
    return xs[-1]
```

### 635. first (training)

Buggy:
```python
def first_22(xs):
    return xs[1]
```

Reference fix:
```python
def first_22(xs):
    return xs[0]
```

### 636. even (training)

Buggy:
```python
def even_28(n):
    return n % 2 == 1
```

Reference fix:
```python
def even_28(n):
    return n % 2 == 0
```

### 637. odd (training)

Buggy:
```python
def odd_10(n):
    return n % 2 == 0
```

Reference fix:
```python
def odd_10(n):
    return n % 2 == 1
```

### 638. maximum (training)

Buggy:
```python
def maximum_1(a, b):
    return a if a < b else b
```

Reference fix:
```python
def maximum_1(a, b):
    return a if a > b else b
```

### 639. factorial (training)

Buggy:
```python
def factorial_25(n):
    r = 1
    for i in range(1, n):
        r *= i
    return r
```

Reference fix:
```python
def factorial_25(n):
    r = 1
    for i in range(1, n + 1):
        r *= i
    return r
```

### 640. last (validation)

Buggy:
```python
def last_34(xs):
    return xs[0]
```

Reference fix:
```python
def last_34(xs):
    return xs[-1]
```
