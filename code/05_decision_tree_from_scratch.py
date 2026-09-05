# DECISION TREE FROM SCRATCH
# CART algorithm with Gini impurity

import numpy as np
import matplotlib.pyplot as plt

class Node:
    """Decision Tree Node."""
    def __init__(self, feature=None, threshold=None, left=None, 
                 right=None, value=None):
        self.feature = feature      # Feature index to split on
        self.threshold = threshold  # Threshold value for split
        self.left = left            # Left subtree (≤ threshold)
        self.right = right          # Right subtree (> threshold)
        self.value = value          # Leaf class prediction


class DecisionTreeClassifier:
    """
    Decision Tree Classifier using CART algorithm.
    
    Split criterion: Gini Impurity
    Gini = 1 - Σ(p_i²) where p_i is probability of class i
    
    Algorithm:
    1. Find best feature and threshold to split
    2. Create left (≤) and right (>) child nodes
    3. Recursively build subtrees
    4. Stop when: max_depth reached, min_samples, or pure node
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
    
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.root = self._grow_tree(X, y, depth=0)
        return self
    
    def _gini(self, y):
        """Compute Gini impurity."""
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _best_split(self, X, y):
        """Find the best feature and threshold to split on."""
        n_samples, n_features = X.shape
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if (np.sum(left_mask) < self.min_samples_leaf or 
                    np.sum(right_mask) < self.min_samples_leaf):
                    continue
                
                # Weighted Gini
                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                n_left, n_right = np.sum(left_mask), np.sum(right_mask)
                weighted_gini = (n_left * gini_left + n_right * gini_right) / n_samples
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _grow_tree(self, X, y, depth):
        """Recursively build the decision tree."""
        n_samples = len(y)
        n_labels = len(np.unique(y))
        
        # Stopping conditions
        if (depth >= self.max_depth or n_labels == 1 or 
            n_samples < self.min_samples_split):
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)
        
        # Find best split
        feature, threshold = self._best_split(X, y)
        
        if feature is None:
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)
        
        # Split data
        left_mask = X[:, feature] <= threshold
        left_subtree = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)
        
        return Node(feature=feature, threshold=threshold, 
                   left=left_subtree, right=right_subtree)
    
    def _predict_one(self, x, node):
        """Predict class for a single sample."""
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)
    
    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])
    
    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=200, n_features=2, 
                               n_redundant=0, random_state=42)
    
    split = 160
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train
    tree = DecisionTreeClassifier(max_depth=5)
    tree.fit(X_train, y_train)
    
    train_acc = tree.accuracy(X_train, y_train)
    test_acc = tree.accuracy(X_test, y_test)
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    
    # Decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                          np.linspace(y_min, y_max, 200))
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], c='red', label='Class 0')
    plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c='blue', label='Class 1')
    plt.title(f'Decision Tree (Accuracy: {test_acc:.2%})')
    plt.legend()
    plt.savefig('decision_tree_demo.png', dpi=150)
    plt.show()
    
    print("✅ Decision Tree from scratch complete!")
