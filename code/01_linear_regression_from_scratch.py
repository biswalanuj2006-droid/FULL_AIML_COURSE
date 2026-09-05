# LINEAR REGRESSION FROM SCRATCH
# Complete implementation with gradient descent

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """
    Linear Regression using Gradient Descent.
    
    Model: y = Xw + b
    Loss:  MSE = (1/n) * Σ(y_pred - y_true)²
    Gradient: dw = (2/n) * Xᵀ(Xw + b - y)
              db = (2/n) * Σ(Xw + b - y)
    Update:  w = w - lr * dw
             b = b - lr * db
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []
        
        # Gradient descent
        for i in range(self.n_iter):
            # Forward pass
            y_pred = X.dot(self.weights) + self.bias
            
            # Compute loss (MSE)
            loss = np.mean((y_pred - y) ** 2)
            self.losses.append(loss)
            
            # Compute gradients
            dw = (2 / n_samples) * X.T.dot(y_pred - y)
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Print progress
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.n_iter}, Loss: {loss:.6f}")
    
    def predict(self, X):
        return X.dot(self.weights) + self.bias
    
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)  # R² score


# ============================================================
# DEMO: Generate synthetic data and fit model
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate synthetic data: y = 3x + 7 + noise
    X = 2 * np.random.rand(100, 1)
    y = 3 * X.squeeze() + 7 + np.random.randn(100) * 0.5
    
    # Fit model
    model = LinearRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    
    print(f"\nLearned weights: {model.weights[0]:.4f} (true: 3.0)")
    print(f"Learned bias: {model.bias:.4f} (true: 7.0)")
    print(f"R² score: {model.score(X, y):.4f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Data + Regression Line
    axes[0].scatter(X, y, alpha=0.5, label='Data')
    axes[0].plot(X, model.predict(X), color='red', label='Regression Line')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('y')
    axes[0].set_title('Linear Regression Fit')
    axes[0].legend()
    
    # Plot 2: Loss Curve
    axes[1].plot(model.losses)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('MSE Loss')
    axes[1].set_title('Training Loss Curve')
    
    plt.tight_layout()
    plt.savefig('linear_regression_demo.png', dpi=150)
    plt.show()
    
    print("\n✅ Linear Regression from scratch complete!")
