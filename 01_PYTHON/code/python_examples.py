"""
Module 01: Python for AI/ML — Code Examples
=============================================
Run these examples to see Python concepts in action.

Usage:
    python python_examples.py
    OR run cells individually in Jupyter
"""

# =============================================================================
# 1. VARIABLES AND TYPES
# =============================================================================

print("=" * 60)
print("1. VARIABLES AND TYPES")
print("=" * 60)

# Dataset metadata
dataset_info = {
    "name": "Customer Churn",
    "rows": 7043,
    "columns": 21,
    "features": ["gender", "SeniorCitizen", "tenure", "MonthlyCharges"],
    "target": "Churn",
    "has_missing": True
}

print(f"Dataset: {dataset_info['name']}")
print(f"Shape: ({dataset_info['rows']}, {dataset_info['columns']})")
print(f"Features: {len(dataset_info['features'])}")

# Type checking
values = [42, 3.14, "hello", True, None, [1, 2, 3], {"a": 1}]
for v in values:
    print(f"  {str(v):>20} → {type(v).__name__}")

# Mutable default argument trap
def bad_append(item, lst=[]):    # DANGER!
    lst.append(item)
    return lst

def good_append(item, lst=None):  # SAFE
    if lst is None:
        lst = []
    lst.append(item)
    return lst

print("\nMutable default trap:")
print(f"  bad_append(1): {bad_append(1)}")
print(f"  bad_append(2): {bad_append(2)}")    # [1, 2] — wrong!
print(f"  good_append(1): {good_append(1)}")
print(f"  good_append(2): {good_append(2)}")  # [2] — correct!

# =============================================================================
# 2. LIST COMPREHENSIONS
# =============================================================================

print("\n" + "=" * 60)
print("2. LIST COMPREHENSIONS")
print("=" * 60)

# Basic
squares = [x**2 for x in range(10)]
print(f"Squares: {squares}")

# With filter
evens = [x for x in range(20) if x % 2 == 0]
print(f"Even numbers: {evens}")

# String processing
names = ["  Alice ", " BOB", "charlie  "]
cleaned = [name.strip().lower() for name in names]
print(f"Cleaned names: {cleaned}")

# Nested: flatten 2D list
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [num for row in matrix for num in row]
print(f"Flattened: {flat}")

# ML use: Feature selection
importance = {"age": 0.15, "income": 0.25, "tenure": 0.20,
              "contract": 0.10, "charges": 0.30}
threshold = 0.20
important_features = [feat for feat, imp in importance.items() if imp >= threshold]
print(f"Important features (>= {threshold}): {important_features}")

# Dict comprehension
score_to_grade = {f"student_{i}": 60 + i * 5 for i in range(5)}
print(f"Scores: {score_to_grade}")

# =============================================================================
# 3. FUNCTIONS
# =============================================================================

print("\n" + "=" * 60)
print("3. FUNCTIONS")
print("=" * 60)

# Basic function with type hints
def calculate_mse(y_true: list, y_pred: list) -> float:
    """Calculate Mean Squared Error from scratch."""
    n = len(y_true)
    return sum((t - p)**2 for t, p in zip(y_true, y_pred)) / n

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mse = calculate_mse(y_true, y_pred)
print(f"MSE: {mse:.4f}")

# Function with *args and **kwargs
def create_config(model_name, *features, **params):
    """Create a model configuration."""
    config = {
        "model": model_name,
        "features": list(features),
        "params": params
    }
    return config

config = create_config("RandomForest", "age", "income", "tenure",
                       n_estimators=100, max_depth=5)
print(f"Config: {config}")

# Lambda + sorted
models = [
    {"name": "LR", "f1": 0.82},
    {"name": "RF", "f1": 0.87},
    {"name": "XGB", "f1": 0.89},
    {"name": "SVM", "f1": 0.85}
]
sorted_models = sorted(models, key=lambda m: m["f1"], reverse=True)
print(f"Best model: {sorted_models[0]['name']} (F1: {sorted_models[0]['f1']})")

# Decorator
import time
from functools import wraps

def timer(func):
    """Decorator that times function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  {func.__name__}: {elapsed:.4f}s")
        return result
    return wrapper

@timer
def slow_sum(n):
    """Sum numbers from 0 to n."""
    return sum(range(n))

result = slow_sum(1000000)
print(f"Sum: {result}")

# =============================================================================
# 4. CLASSES AND OOP
# =============================================================================

print("\n" + "=" * 60)
print("4. CLASSES AND OOP")
print("=" * 60)

class ModelResult:
    """Store and compare ML model results."""

    def __init__(self, name, accuracy, f1, precision, recall):
        self.name = name
        self.accuracy = accuracy
        self.f1 = f1
        self.precision = precision
        self.recall = recall

    def __str__(self):
        return (f"{self.name}: acc={self.accuracy:.3f}, "
                f"f1={self.f1:.3f}, prec={self.precision:.3f}, "
                f"rec={self.recall:.3f}")

    def __repr__(self):
        return f"ModelResult('{self.name}', {self.accuracy}, {self.f1}, ...)"

    def __eq__(self, other):
        return self.f1 == other.f1

    def __lt__(self, other):
        return self.f1 < other.f1

    def summary_dict(self):
        return {
            "name": self.name,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall
        }

# Create and compare results
lr = ModelResult("Logistic Regression", 0.82, 0.79, 0.80, 0.78)
rf = ModelResult("Random Forest", 0.87, 0.85, 0.86, 0.84)
xgb = ModelResult("XGBoost", 0.89, 0.87, 0.88, 0.86)

print(f"LR:  {lr}")
print(f"RF:  {rf}")
print(f"XGB: {xgb}")
print(f"XGB > RF: {xgb > rf}")
print(f"Best: {max([lr, rf, xgb])}")

# Dataclass example
from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    model_name: str
    dataset: str
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    early_stopping: bool = True
    patience: int = 10

config = TrainingConfig(model_name="ResNet", dataset="cifar10")
print(f"\nTraining config: {config}")

# =============================================================================
# 5. ERROR HANDLING
# =============================================================================

print("\n" + "=" * 60)
print("5. ERROR HANDLING")
print("=" * 60)

class ValidationError(Exception):
    """Custom validation error."""
    def __init__(self, field, message):
        self.field = field
        self.message = message
        super().__init__(f"Validation error in '{field}': {message}")

def validate_model_config(config):
    """Validate model configuration."""
    required = ["model_name", "dataset"]
    for field in required:
        if field not in config:
            raise ValidationError(field, "is required")

    if config.get("learning_rate", 0) <= 0:
        raise ValidationError("learning_rate", "must be positive")

    if config.get("epochs", 0) < 1:
        raise ValidationError("epochs", "must be at least 1")

    return True

# Test validation
try:
    validate_model_config({"model_name": "RF"})
except ValidationError as e:
    print(f"Caught: {e}")

try:
    validate_model_config({"model_name": "RF", "dataset": "churn",
                           "learning_rate": -0.01})
except ValidationError as e:
    print(f"Caught: {e}")

# Safe data processing
def safe_process(values):
    """Process values safely with error handling."""
    results = []
    for v in values:
        try:
            result = float(v) ** 2
            results.append(result)
        except (ValueError, TypeError):
            print(f"  Skipping invalid value: {v}")
            results.append(None)
    return results

data = [1, 2, "abc", 4, None, 6]
processed = safe_process(data)
print(f"Processed: {processed}")

# =============================================================================
# 6. FILE HANDLING
# =============================================================================

print("\n" + "=" * 60)
print("6. FILE HANDLING")
print("=" * 60)

import json
import csv
from pathlib import Path

# JSON example
config = {
    "experiment": "churn_prediction",
    "models": {
        "logistic_regression": {"C": 1.0, "penalty": "l2"},
        "random_forest": {"n_estimators": 100, "max_depth": 5}
    },
    "results": [
        {"model": "LR", "accuracy": 0.82},
        {"model": "RF", "accuracy": 0.87}
    ]
}

# Save JSON
config_path = Path("experiment_config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"Saved config to {config_path}")

# Load JSON
with open(config_path) as f:
    loaded = json.load(f)
print(f"Loaded experiment: {loaded['experiment']}")

# CSV example
csv_path = Path("sample_data.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "age", "score"])
    writer.writerow(["Alice", 25, 95])
    writer.writerow(["Bob", 30, 87])
    writer.writerow(["Charlie", 35, 92])
print(f"Saved CSV to {csv_path}")

# Read CSV
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"  {row['name']}: age={row['age']}, score={row['score']}")

# Cleanup
config_path.unlink(missing_ok=True)
csv_path.unlink(missing_ok=True)
print("Cleaned up temp files")

# =============================================================================
# 7. ML PATTERNS
# =============================================================================

print("\n" + "=" * 60)
print("7. ML PATTERNS")
print("=" * 60)

# Pattern 1: Model comparison
models_config = {
    "LogisticRegression": {"C": 1.0},
    "RandomForest": {"n_estimators": 100},
    "XGBoost": {"learning_rate": 0.1}
}

# Simulated results (in real code, these come from training)
simulated_results = {
    "LogisticRegression": 0.82,
    "RandomForest": 0.87,
    "XGBoost": 0.89
}

best_model = max(simulated_results, key=simulated_results.get)
print(f"Best model: {best_model} (score: {simulated_results[best_model]})")

# Pattern 2: Feature importance ranking
features = ["age", "income", "tenure", "contract", "monthly_charges"]
importances = [0.15, 0.25, 0.20, 0.10, 0.30]

ranked = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
print("\nFeature importance ranking:")
for rank, (feat, imp) in enumerate(ranked, 1):
    bar = "█" * int(imp * 40)
    print(f"  {rank}. {feat:<20} {imp:.2f} {bar}")

# Pattern 3: Batch processing
def process_batch(data, batch_size=3):
    """Process data in batches."""
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        yield batch

data = list(range(10))
print("\nBatch processing:")
for batch_num, batch in enumerate(process_batch(data, batch_size=3)):
    print(f"  Batch {batch_num + 1}: {batch}")

print("\n" + "=" * 60)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 60)
