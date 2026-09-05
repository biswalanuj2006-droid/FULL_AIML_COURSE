# INTERMEDIATE ML EXERCISES
# Complete these exercises to practice ML concepts

import numpy as np
from sklearn.datasets import make_classification, make_regression

# ============================================================
# EXERCISE 1: Implement MSE Loss
# ============================================================

def mse_loss(y_true, y_pred):
    """Compute Mean Squared Error"""
    # Your code here
    pass

# Test:
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
print(f"MSE: {mse_loss(y_true, y_pred)}")

# ============================================================
# EXERCISE 2: Implement Cross-Entropy Loss
# ============================================================

def cross_entropy_loss(y_true, y_pred):
    """Compute Binary Cross-Entropy Loss"""
    # Your code here
    pass

# Test:
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.3])
print(f"BCE: {cross_entropy_loss(y_true, y_pred)}")

# ============================================================
# EXERCISE 3: Implement Accuracy Metric
# ============================================================

def accuracy(y_true, y_pred):
    """Compute classification accuracy"""
    # Your code here
    pass

# Test:
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([1, 0, 0, 1, 1])
print(f"Accuracy: {accuracy(y_true, y_pred)}")

# ============================================================
# EXERCISE 4: Implement Confusion Matrix
# ============================================================

def confusion_matrix(y_true, y_pred):
    """Compute confusion matrix"""
    # Your code here
    pass

# Test:
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

# ============================================================
# EXERCISE 5: Implement Precision, Recall, F1
# ============================================================

def precision(y_true, y_pred):
    """Compute precision"""
    # Your code here
    pass

def recall(y_true, y_pred):
    """Compute recall"""
    # Your code here
    pass

def f1_score(y_true, y_pred):
    """Compute F1 score"""
    # Your code here
    pass

# Test:
print(f"Precision: {precision(y_true, y_pred)}")
print(f"Recall: {recall(y_true, y_pred)}")
print(f"F1: {f1_score(y_true, y_pred)}")

# ============================================================
# EXERCISE 6: Implement Standard Scaler
# ============================================================

class StandardScaler:
    def fit(self, X):
        """Compute mean and std"""
        # Your code here
        pass
    
    def transform(self, X):
        """Standardize features"""
        # Your code here
        pass
    
    def fit_transform(self, X):
        """Fit and transform"""
        self.fit(X)
        return self.transform(X)

# Test:
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Original mean: {X.mean(axis=0)}, Std: {X.std(axis=0)}")
print(f"Scaled mean: {X_scaled.mean(axis=0)}, Std: {X_scaled.std(axis=0)}")

# ============================================================
# EXERCISE 7: Implement Train-Test Split
# ============================================================

def train_test_split(X, y, test_size=0.2, random_state=None):
    """Split data into train and test sets"""
    # Your code here
    pass

# Test:
X, y = make_classification(n_samples=100, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ============================================================
# EXERCISE 8: Implement K-Fold Cross Validation
# ============================================================

def k_fold_cv(X, y, model_class, k=5):
    """Perform K-Fold Cross Validation"""
    # Your code here
    pass

# Test:
from sklearn.linear_model import LogisticRegression
scores = k_fold_cv(X, y, LogisticRegression, k=5)
print(f"CV Scores: {scores}")
print(f"Mean: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

# ============================================================
# EXERCISE 9: Implement Feature Selection (Variance Threshold)
# ============================================================

class VarianceThreshold:
    def __init__(self, threshold=0.0):
        self.threshold = threshold
    
    def fit(self, X):
        """Compute variances"""
        # Your code here
        pass
    
    def transform(self, X):
        """Select features above threshold"""
        # Your code here
        pass

# Test:
X = np.array([[0, 2, 0.5], [1, 2, 0.5], [2, 2, 0.5], [3, 2, 0.5]])
selector = VarianceThreshold(threshold=0.1)
X_selected = selector.fit_transform(X)
print(f"Original shape: {X.shape}, Selected shape: {X_selected.shape}")

# ============================================================
# EXERCISE 10: Complete ML Pipeline
# ============================================================

def ml_pipeline(X, y):
    """Complete ML pipeline: preprocess, split, train, evaluate"""
    # Your code here
    # 1. Standardize features
    # 2. Split data
    # 3. Train model
    # 4. Evaluate
    # 5. Return results
    pass

# Test:
X, y = make_classification(n_samples=200, n_features=10, random_state=42)
results = ml_pipeline(X, y)
print(f"Results: {results}")
