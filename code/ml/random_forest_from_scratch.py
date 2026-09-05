# RANDOM FOREST FROM SCRATCH
# Complete implementation using bagging + random feature selection

import numpy as np
from collections import Counter

class DecisionTree:
    """Simple decision tree for classification."""
    
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None
        self.n_features = None
    
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, 
                     right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
    
    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y, depth=0)
        return self
    
    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    
    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        # Random feature subset
        if self.max_features and self.max_features < n_features:
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        else:
            feature_indices = range(n_features)
        
        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < 1 or np.sum(right_mask) < 1:
                    continue
                
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
        n_samples = len(y)
        n_labels = len(np.unique(y))
        
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            return self.Node(value=np.argmax(np.bincount(y)))
        
        feature, threshold = self._best_split(X, y)
        
        if feature is None:
            return self.Node(value=np.argmax(np.bincount(y)))
        
        left_mask = X[:, feature] <= threshold
        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)
        
        return self.Node(feature=feature, threshold=threshold, left=left, right=right)
    
    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)
    
    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])


class RandomForestClassifier:
    """
    Random Forest using bagging + random feature selection.
    
    Algorithm:
    1. Create bootstrap samples
    2. Train decision trees on each sample
    3. Each tree uses random feature subset
    4. Majority vote for final prediction
    """
    
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2,
                 max_features='sqrt', bootstrap=True):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []
    
    def _get_max_features(self, n_features):
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return self.max_features
        return n_features
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        max_feat = self._get_max_features(n_features)
        
        self.trees = []
        
        for i in range(self.n_trees):
            # Bootstrap sample
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
            else:
                indices = np.arange(n_samples)
            
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Train tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_feat
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Majority vote
        result = []
        for i in range(X.shape[0]):
            votes = tree_predictions[:, i]
            result.append(Counter(votes).most_common(1)[0][0])
        
        return np.array(result)
    
    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate data. (Smaller defaults keep the pure-Python demo fast;
    # raise n_samples/n_trees/depth for the full lesson.)
    X, y = make_classification(n_samples=300, n_features=10, 
                               n_informative=5, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_trees=25, max_depth=8)
    rf.fit(X_train, y_train)
    
    train_acc = rf.accuracy(X_train, y_train)
    test_acc = rf.accuracy(X_test, y_test)
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    
    print("[OK] Random Forest from scratch complete!")
