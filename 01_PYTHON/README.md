# Module 01: Python for AI/ML

## What You Will Learn

- Python fundamentals: variables, types, operators
- Control flow: if/else, loops, comprehensions
- Functions: arguments, return values, lambdas, closures, decorators
- Object-Oriented Programming: classes, inheritance, magic methods
- File handling: reading, writing, CSV, JSON
- Error handling: try/except, custom exceptions
- Modules and packages: importing, creating packages
- Virtual environments and dependency management
- Python idioms and best practices for data science

## Why You Need It

Python is the foundation of the entire AI/ML ecosystem. Every library you will use — NumPy, Pandas, scikit-learn, PyTorch, TensorFlow, FastAPI — is built on Python. Without solid Python skills, you will struggle with every subsequent module.

This module is NOT about learning "basic programming." It is about learning the Python that AI/ML engineers actually use daily: list comprehensions for data manipulation, decorators for ML pipelines, context managers for resource handling, and generators for processing large datasets.

## Prerequisites

- Basic computer literacy
- Completed the diagnostic assessment (00_ORIENTATION/diagnostic_assessment.txt)
- Text editor installed (VS Code recommended)

## What You Will Be Able to Do After This Module

1. Write Python functions, classes, and modules
2. Manipulate data using lists, dictionaries, and comprehensions
3. Handle errors gracefully with try/except
4. Read and write files (CSV, JSON, text)
5. Use virtual environments for project isolation
6. Write clean, readable, Pythonic code
7. Understand the Python code used in all subsequent modules

## Module Files

| File | Topic | Duration |
|------|-------|----------|
| 01_variables_types.txt | Variables, Types, Operators | 2-3 hours |
| 02_control_flow.txt | If/Else, Loops, Comprehensions | 2-3 hours |
| 03_functions.txt | Functions, Args, Lambdas, Decorators | 3-4 hours |
| 04_classes_oop.txt | Classes, OOP, Magic Methods | 3-4 hours |
| 05_file_handling.txt | File I/O, CSV, JSON | 2-3 hours |
| 06_error_handling.txt | Exceptions, Error Handling | 2-3 hours |
| 07_modules_packages.txt | Modules, Packages, Imports | 1-2 hours |
| 08_virtual_environments.txt | venv, pip, requirements.txt | 1-2 hours |

## Total Estimated Time: 16-24 hours

## Learning Path

```
Variables & Types
    ↓
Control Flow
    ↓
Functions
    ↓
Classes & OOP
    ↓
File Handling
    ↓
Error Handling
    ↓
Modules & Packages
    ↓
Virtual Environments
    ↓
EXERCISES → LABS → PROJECTS
```

## Knowledge Checkpoint

After completing this module, you should be able to:

- [ ] Explain the difference between mutable and immutable types
- [ ] Write a list comprehension that filters and transforms data
- [ ] Create a function with *args and **kwargs
- [ ] Define a class with inheritance
- [ ] Read a CSV file and parse it into a dictionary
- [ ] Handle file-not-found errors gracefully
- [ ] Create and activate a virtual environment
- [ ] Write clean, well-structured Python code

## Common Mistakes

1. **Mutable default arguments** — Using `def f(x=[])` instead of `def f(x=None)`
2. **Not using context managers** — Using `open()` without `with`
3. **Ignoring exceptions** — Using bare `except:` without specifying the exception type
4. **Global variable abuse** — Using `global` when you should pass parameters
5. **Not using virtual environments** — Installing packages globally

## Interview Questions

1. What is the difference between a list and a tuple?
2. What are *args and **kwargs?
3. What is a decorator? Give an example.
4. What is the difference between `__str__` and `__repr__`?
5. What is a generator? When would you use one?
6. What is the GIL? How does it affect Python?
7. What is the difference between deep copy and shallow copy?
8. What are context managers? How do you create one?
