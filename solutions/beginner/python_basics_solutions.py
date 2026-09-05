# ============================================================
# SOLUTIONS — exercises/beginner/python_basics.py
# Run: python python_basics_solutions.py
# ============================================================

# ------------------------------------------------------------
# EXERCISE 1: Hello World
# ------------------------------------------------------------
print("Hello, World!")

# ------------------------------------------------------------
# EXERCISE 2: Variables and Types
# ------------------------------------------------------------
my_int = 42
my_float = 3.14
my_string = "AI/ML"
my_bool = True
my_list = [1, 2, 3, 4, 5]
print(f"int: {my_int}, type: {type(my_int)}")
print(f"float: {my_float}, type: {type(my_float)}")
print(f"string: {my_string}, type: {type(my_string)}")
print(f"bool: {my_bool}, type: {type(my_bool)}")
print(f"list: {my_list}, type: {type(my_list)}")

# ------------------------------------------------------------
# EXERCISE 3: String Operations
# ------------------------------------------------------------
text = "Hello, Python World!"
print(text.upper())                        # HELLO, PYTHON WORLD!
print(text.replace("Python", "AI/ML"))     # Hello, AI/ML World!
print(len(text))                           # 21
print(text.startswith("Hello"))            # True
print(text.split())                        # ['Hello,', 'Python', 'World!']

# ------------------------------------------------------------
# EXERCISE 4: Lists
# ------------------------------------------------------------
numbers = [5, 2, 8, 1, 9, 3]

numbers.sort()
print(f"Sorted: {numbers}")

print(f"Max: {max(numbers)}, Min: {min(numbers)}")

numbers.reverse()
print(f"Reversed: {numbers}")

# Remove duplicates while keeping order
unique_numbers = list(dict.fromkeys(numbers))
print(f"Unique: {unique_numbers}")

print(f"Sum: {sum(numbers)}, Average: {sum(numbers) / len(numbers)}")

# ------------------------------------------------------------
# EXERCISE 5: Dictionary
# ------------------------------------------------------------
grades = {"Alice": 85, "Bob": 92, "Charlie": 78}
grades["Diana"] = 95                       # Add a new student
grades["Alice"] = 88                       # Update a grade
highest = max(grades.values())
average = sum(grades.values()) / len(grades)
print(f"Grades: {grades}")
print(f"Highest grade: {highest}")
print(f"Average grade: {average:.2f}")

# ------------------------------------------------------------
# EXERCISE 6: Loops
# ------------------------------------------------------------
print("1. Numbers 1-10:")
for i in range(1, 11):
    print(i, end=" ")
print()

print("2. Even numbers 1-20:")
for i in range(2, 21, 2):
    print(i, end=" ")
print()

print("3. Multiplication table for 5:")
for i in range(1, 11):
    print(f"5 x {i} = {5 * i}")

factorial = 1
for i in range(1, 11):
    factorial *= i
print(f"4. Factorial of 10: {factorial}")

print("5. Fibonacci (first 10):")
a, b = 0, 1
for _ in range(10):
    print(a, end=" ")
    a, b = b, a + b
print()

# ------------------------------------------------------------
# EXERCISE 7: Functions
# ------------------------------------------------------------
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def count_vowels(s):
    return sum(1 for c in s.lower() if c in "aeiou")


def second_largest(lst):
    unique = sorted(set(lst))
    return unique[-2] if len(unique) >= 2 else None


print(f"is_prime(29): {is_prime(29)}")
print(f"is_prime(1): {is_prime(1)}")
print(f"count_vowels('Artificial Intelligence'): {count_vowels('Artificial Intelligence')}")
print(f"second_largest([5, 2, 8, 1, 9, 3]): {second_largest([5, 2, 8, 1, 9, 3])}")

# ------------------------------------------------------------
# EXERCISE 8: List Comprehension
# ------------------------------------------------------------
squares_even = [x ** 2 for x in range(1, 21) if x % 2 == 0]
print(f"Squares of evens 1-20: {squares_even}")

words = ["cat", "elephant", "dog", "giraffe", "bat", "hippopotamus"]
long_words = [w for w in words if len(w) > 3]
print(f"Words longer than 3 chars: {long_words}")

celsius = [0, 10, 20, 30, 40]
fahrenheit = [(c * 9 / 5) + 32 for c in celsius]
print(f"Celsius {celsius} -> Fahrenheit {fahrenheit}")

# ------------------------------------------------------------
# EXERCISE 9: File Handling
# ------------------------------------------------------------
items = ["numpy", "pandas", "matplotlib", "scikit-learn", "pytorch"]

with open("_tmp_items.txt", "w") as f:
    for item in items:
        f.write(item + "\n")

with open("_tmp_items.txt", "r") as f:
    lines = f.readlines()

print("File contents:")
for line in lines:
    print(f"  {line.rstrip()}")
print(f"Number of lines: {len(lines)}")

import os
os.remove("_tmp_items.txt")  # cleanup

# ------------------------------------------------------------
# EXERCISE 10: Error Handling
# ------------------------------------------------------------
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        print("Error: division by zero")
        return None


def safe_get(lst, index):
    try:
        return lst[index]
    except IndexError:
        print(f"Error: index {index} out of range")
        return None


def safe_get_value(d, key):
    try:
        return d[key]
    except KeyError:
        print(f"Error: key '{key}' not found")
        return None


print(f"safe_divide(10, 2): {safe_divide(10, 2)}")
print(f"safe_divide(10, 0): {safe_divide(10, 0)}")
print(f"safe_get([1,2,3], 5): {safe_get([1, 2, 3], 5)}")
print(f"safe_get_value(grades, 'Eve'): {safe_get_value(grades, 'Eve')}")
