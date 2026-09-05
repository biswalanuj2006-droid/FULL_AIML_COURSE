# LOGISTIC REGRESSION FROM SCRATCH
# Complete implementation for binary classification

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

class LogisticRegression:
    """
    Logistic Regression using Gradient Descent.
    
    Model:  z = Xw + b
            ŷ = sigmoid(z)
    Loss:   Binary Cross-Entropy = -(1/n) Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
    Gradient: dw = (1/n) Xᵀ(ŷ - y)
              db = (1/n) Σ(ŷ - y)
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []
        
        for i in range(self.n_iter):
            # Forward pass
            z = X.dot(self.weights) + self.bias
            y_pred = sigmoid(z)
            
            # Compute loss (binary cross-entropy)
            epsilon = 1e-15  # Avoid log(0)
            loss = -np.mean(y * np.log(y_pred + epsilon) + 
                          (1 - y) * np.log(1 - y_pred + epsilon))
            self.losses.append(loss)
            
            # Compute gradients
            dw = (1 / n_samples) * X.T.dot(y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            if (i + 1) % 200 == 0:
                print(f"Iteration {i+1}/{self.n_iter}, Loss: {loss:.6f}")
    
    def predict_proba(self, X):
        return sigmoid(X.dot(self.weights) + self.bias)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


# ============================================================
# DEMO: Binary classification
# ============================================================

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X, y = make_classification(n_samples=200, n_features=2, 
                               n_redundant=0, n_informative=2,
                               random_state=42, n_clusters_per_class=1)
    
    # Fit model
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    
    print(f"\nAccuracy: {model.accuracy(X, y):.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                          np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[0].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    axes[0].scatter(X[y==0, 0], X[y==0, 1], c='red', label='Class 0', alpha=0.6)
    axes[0].scatter(X[y==1, 0], X[y==1, 1], c='blue', label='Class 1', alpha=0.6)
    axes[0].set_title('Logistic Regression Decision Boundary')
    axes[0].legend()
    
    # Loss curve
    axes[1].plot(model.losses)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Binary Cross-Entropy Loss')
    axes[1].set_title('Training Loss')
    
    plt.tight_layout()
    plt.savefig('logistic_regression_demo.png', dpi=150)
    plt.show()
    
    print("✅ Logistic Regression from scratch complete!")
