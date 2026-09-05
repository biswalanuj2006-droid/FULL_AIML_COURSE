# GRADIENT BOOSTING FROM SCRATCH
# Complete implementation for classification

import numpy as np

class DecisionStump:
    """Simple decision stump (1-level tree) for gradient boosting."""
    
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left_value = None
        self.right_value = None
    
    def fit(self, X, residuals):
        n_samples, n_features = X.shape
        best_loss = float('inf')
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                left_value = np.mean(residuals[left_mask])
                right_value = np.mean(residuals[right_mask])
                
                predictions = np.where(left_mask, left_value, right_value)
                loss = np.mean((residuals - predictions) ** 2)
                
                if loss < best_loss:
                    best_loss = loss
                    self.feature = feature
                    self.threshold = threshold
                    self.left_value = left_value
                    self.right_value = right_value
    
    def predict(self, X):
        return np.where(X[:, self.feature] <= self.threshold,
                       self.left_value, self.right_value)


class GradientBoostingClassifier:
    """
    Gradient Boosting for Classification.
    
    Algorithm:
    1. Start with initial prediction (log-odds for classification)
    2. Compute pseudo-residuals
    3. Fit a weak learner to residuals
    4. Update predictions
    5. Repeat
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        # Initial prediction: log-odds
        p = np.clip(np.mean(y), 1e-15, 1 - 1e-15)
        self.initial_prediction = np.log(p / (1 - p))
        
        # Current predictions
        F = np.full(len(y), self.initial_prediction)
        
        self.trees = []
        
        for i in range(self.n_estimators):
            # Compute pseudo-residuals
            predictions = self._sigmoid(F)
            residuals = y - predictions
            
            # Fit tree to residuals
            tree = DecisionStump()
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # Update predictions
            F += self.lr * tree.predict(X)
        
        return self
    
    def predict_proba(self, X):
        F = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            F += self.lr * tree.predict(X)
        return self._sigmoid(F)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate data. (Smaller defaults keep the pure-Python demo fast;
    # raise n_samples/n_estimators for the full lesson.)
    X, y = make_classification(n_samples=300, n_features=10, 
                               n_informative=5, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    gb = GradientBoostingClassifier(n_estimators=60, learning_rate=0.1)
    gb.fit(X_train, y_train)
    
    train_acc = gb.accuracy(X_train, y_train)
    test_acc = gb.accuracy(X_test, y_test)
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    
    print("[OK] Gradient Boosting from scratch complete!")
