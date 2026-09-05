# BEGINNER PYTHON EXERCISES
# Complete these exercises to test your basic Python knowledge

# ============================================================
# EXERCISE 1: Hello World
# Print "Hello, World!" to the console
# ============================================================

# Your code here:


# ============================================================
# EXERCISE 2: Variables and Types
# Create variables of different types and print them
# ============================================================

# Create an integer, float, string, boolean, and list
# Print each one with its type


# ============================================================
# EXERCISE 3: String Operations
# Given a string, perform the following operations
# ============================================================

text = "Hello, Python World!"

# 1. Convert to uppercase
# 2. Replace "Python" with "AI/ML"
# 3. Find the length
# 4. Check if it starts with "Hello"
# 5. Split into words


# ============================================================
# EXERCISE 4: Lists
# Perform list operations
# ============================================================

numbers = [5, 2, 8, 1, 9, 3]

# 1. Sort the list
# 2. Find the maximum and minimum
# 3. Reverse the list
# 4. Remove duplicates
# 5. Find the sum and average


# ============================================================
# EXERCISE 5: Dictionary
# Create and manipulate dictionaries
# ============================================================

# Create a dictionary of student grades
# Add a new student
# Update a grade
# Find the highest grade
# Calculate the average grade


# ============================================================
# EXERCISE 6: Loops
# Write loops to solve these problems
# ============================================================

# 1. Print numbers 1 to 10
# 2. Print even numbers from 1 to 20
# 3. Print multiplication table for 5
# 4. Find factorial of 10
# 5. Print Fibonacci sequence (first 10 numbers)


# ============================================================
# EXERCISE 7: Functions
# Write functions for these tasks
# ============================================================

# 1. Function to check if a number is prime
def is_prime(n):
    # Your code here
    pass

# 2. Function to count vowels in a string
def count_vowels(s):
    # Your code here
    pass

# 3. Function to find the second largest element in a list
def second_largest(lst):
    # Your code here
    pass


# ============================================================
# EXERCISE 8: List Comprehension
# Rewrite these using list comprehensions
# ============================================================

# 1. Square of even numbers from 1 to 20
# 2. Filter words longer than 3 characters
# 3. Convert temperatures from Celsius to Fahrenheit


# ============================================================
# EXERCISE 9: File Handling
# Write code to handle files
# ============================================================

# 1. Write a list of items to a file
# 2. Read the file and print each line
# 3. Count the number of lines


# ============================================================
# EXERCISE 10: Error Handling
# Add proper error handling to these functions
# ============================================================

# 1. Safe division (handle division by zero)
def safe_divide(a, b):
    # Your code here
    pass

# 2. Safe list access (handle index error)
def safe_get(lst, index):
    # Your code here
    pass

# 3. Safe dictionary access (handle key error)
def safe_get_value(d, key):
    # Your code here
    pass


# ============================================================
# SOLUTIONS (Unfold to check your answers)
# ============================================================

# EXERCISE 1 SOLUTION:
# print("Hello, World!")

# EXERCISE 2 SOLUTION:
# my_int = 42
# my_float = 3.14
# my_string = "AI/ML"
# my_bool = True
# my_list = [1, 2, 3, 4, 5]
# print(f"int: {my_int}, type: {type(my_int)}")
# print(f"float: {my_float}, type: {type(my_float)}")
# print(f"string: {my_string}, type: {type(my_string)}")
# print(f"bool: {my_bool}, type: {type(my_bool)}")
# print(f"list: {my_list}, type: {type(my_list)}")

# EXERCISE 3 SOLUTION:
# print(text.upper())  # HELLO, PYTHON WORLD!
# print(text.replace("Python", "AI/ML"))  # Hello, AI/ML World!
# print(len(text))  # 21
# print(text.startswith("Hello"))  # True
# print(text.split())  # ['Hello,', 'Python', 'World!']

# EXERCISE 4 SOLUTION:
# numbers.sort()
# print(f"Max: {max(numbers)}, Min: {min(numbers)}")
# numbers.reverse()
# unique_numbers = list(set(numbers))
# print(f"Sum: {sum(numbers)}, Average: {sum(numbers)/len(numbers)}")

# EXERCISE 5 SOLUTION:
# grades = {"Alice": 85, "Bob": 92, "Charlie": 78}
# grades["Diana"] = 95  # Add
# grades["Alice"] = 88  # Update
# highest = max(grades.values())
# average = sum(grades.values()) / len(grades)

# EXERCISE 6 SOLUTION:
# 1. for i in range(1, 11): print(i)
# 2. for i in range(2, 21, 2): print(i)
# 3. for i in range(1, 11): print(f"5 × {i} = {5*i}")
# 4. factorial = 1; for i in range(1, 11): factorial *= i
# 5. a, b = 0, 1; for _ in range(10): print(a); a, b = b, a + b

# EXERCISE 7 SOLUTION:
# def is_prime(n):
#     if n < 2: return False
#     for i in range(2, int(n**0.5) + 1):
#         if n % i == 0: return False
#     return True

# def count_vowels(s):
#     return sum(1 for c in s.lower() if c in 'aeiou')

# def second_largest(lst):
#     unique = list(set(lst))
#     unique.sort()
#     return unique[-2] if len(unique) >= 2 else None
