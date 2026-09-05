# K-NEAREST NEIGHBORS FROM SCRATCH
# Complete implementation for classification and regression

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class KNNClassifier:
    """
    K-Nearest Neighbors Classifier.
    
    Algorithm:
    1. Store all training data
    2. For a new point, compute distances to all training points
    3. Find K nearest neighbors
    4. Majority vote for classification
    """
    
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _predict_single(self, x):
        # Compute distances to all training points (brute-force loop:
        # the clearest statement of the algorithm, slow for big batches)
        distances = [self._euclidean_distance(x, x_train) 
                     for x_train in self.X_train]
        
        # Get K nearest neighbor indices
        k_indices = np.argsort(distances)[:self.k]
        
        # Get labels of K nearest neighbors
        k_nearest_labels = self.y_train[k_indices]
        
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X):
        # Vectorized over the whole batch: identical math to
        # _predict_single, but no Python loop (how KNN is actually
        # implemented at scale). dists[i, j] = ||X[i] - X_train[j]||.
        dists = np.sqrt(((X[:, None, :] - self.X_train[None, :, :]) ** 2).sum(-1))
        k_idx = np.argsort(dists, axis=1)[:, :self.k]   # (n_test, k)
        k_labels = self.y_train[k_idx]                  # (n_test, k)
        # Majority vote per row (ties broken by lowest label)
        return np.array([Counter(row).most_common(1)[0][0] for row in k_labels])
    
    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


class KNNRegressor:
    """
    K-Nearest Neighbors Regressor.
    
    Predicts the mean of K nearest neighbors' values.
    """
    
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x):
        distances = [np.sqrt(np.sum((x - x_train) ** 2)) 
                     for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_values = self.y_train[k_indices]
        return np.mean(k_nearest_values)


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    
    # Classification demo
    X_cls, y_cls = make_classification(n_samples=200, n_features=2,
                                        n_redundant=0, random_state=42)
    
    # Split data
    split = 160
    X_train, X_test = X_cls[:split], X_cls[split:]
    y_train, y_test = y_cls[:split], y_cls[split:]
    
    # Try different K values
    for k in [1, 3, 5, 7]:
        model = KNNClassifier(k=k)
        model.fit(X_train, y_train)
        acc = model.accuracy(X_test, y_test)
        print(f"K={k}: Accuracy = {acc:.4f}")
    
    # Visualization with K=5
    model = KNNClassifier(k=5)
    model.fit(X_train, y_train)
    
    x_min, x_max = X_cls[:, 0].min() - 1, X_cls[:, 0].max() + 1
    y_min, y_max = X_cls[:, 1].min() - 1, X_cls[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 120),
                          np.linspace(y_min, y_max, 120))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], c='red', label='Class 0')
    plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c='blue', label='Class 1')
    plt.title('KNN Classification (K=5)')
    plt.legend()
    plt.savefig('knn_demo.png', dpi=150)
    plt.show()
    
    print("✅ KNN from scratch complete!")
