# SOLUTIONS DIRECTORY

This directory contains worked solutions for the exercise files in `exercises/`.

## How to use

1. **Attempt every exercise yourself first** — solutions exist to verify, not to copy.
2. After attempting, run the solution file to confirm the expected output.
3. If your answer differs, find *why* — do not just replace your code.

## Structure

```
solutions/
├── beginner/
│   ├── python_basics_solutions.py       # Solutions for exercises/beginner/python_basics.py
│   └── data_structures_solutions.py     # Solutions for exercises/beginner/data_structures.py
├── intermediate/
│   └── ml_basics_solutions.py           # Solutions for exercises/intermediate/ml_basics.py
└── advanced/
    └── system_design_solutions.txt      # Written architecture answers (no single right answer)
```

## Running the code solutions

```bash
python solutions/beginner/python_basics_solutions.py
python solutions/beginner/data_structures_solutions.py
python solutions/intermediate/ml_basics_solutions.py
```

The intermediate solution prints the exact expected values noted in the
exercise file (MSE, BCE, accuracy, confusion matrix, precision/recall/F1,
scaled means, split sizes, etc.).

## Note on system design solutions

System design has no single correct answer. `system_design_solutions.txt`
gives one defensible, industry-typical design per exercise with the key
trade-offs explained. Use it as a benchmark to argue against, not a
memorized answer.

## Full-answer coverage

- Beginner Python exercises → runnable Python solution with printed output.
- Beginner data-structure exercises → full implementations of Stack, Queue,
  LinkedList, HashMap + classic problems (balanced parentheses, queue with
  two stacks, cycle detection, middle node).
- Intermediate ML exercises → NumPy implementations of MSE, cross-entropy,
  accuracy, confusion matrix, precision/recall/F1, StandardScaler,
  train/test split, K-fold CV, variance threshold, and a full pipeline.
- Advanced system design exercises → architecture, components, technology
  choices, and trade-offs for 6 production AI systems.
