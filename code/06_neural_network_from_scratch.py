# NEURAL NETWORK FROM SCRATCH
# Simple 2-layer neural network for classification

import numpy as np
import matplotlib.pyplot as plt

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    n = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-15)) / n


class NeuralNetwork:
    """
    Simple 2-layer neural network.
    
    Architecture: Input → Hidden (ReLU) → Output (Softmax)
    
    Forward:
        Z1 = X · W1 + b1
        A1 = relu(Z1)
        Z2 = A1 · W2 + b2
        A2 = softmax(Z2)
    
    Backward:
        dZ2 = A2 - y
        dW2 = (1/n) A1ᵀ · dZ2
        db2 = (1/n) Σ dZ2
        dZ1 = (dZ2 · W2ᵀ) ⊙ relu'(Z1)
        dW1 = (1/n) Xᵀ · dZ1
        db1 = (1/n) Σ dZ1
    """
    
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.lr = lr
        # He initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.losses = []
    
    def forward(self, X):
        self.Z1 = X.dot(self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1.dot(self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2
    
    def backward(self, X, y):
        n = X.shape[0]
        
        # Output layer gradients
        dZ2 = self.A2 - y
        dW2 = (1 / n) * self.A1.T.dot(dZ2)
        db2 = (1 / n) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dZ1 = dZ2.dot(self.W2.T) * relu_derivative(self.Z1)
        dW1 = (1 / n) * X.T.dot(dZ1)
        db1 = (1 / n) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def fit(self, X, y, epochs=500, batch_size=32, verbose=True):
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Forward
                y_pred = self.forward(X_batch)
                
                # Loss
                loss = cross_entropy_loss(y_pred, y_batch)
                epoch_loss += loss
                
                # Backward
                self.backward(X_batch, y_batch)
            
            avg_loss = epoch_loss / (n_samples / batch_size)
            self.losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 100 == 0:
                acc = self.accuracy(X, y)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
    
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def accuracy(self, X, y):
        predictions = self.predict(X)
        labels = np.argmax(y, axis=1)
        return np.mean(predictions == labels)


# ============================================================
# DEMO: Simple classification
# ============================================================

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import OneHotEncoder
    
    # Generate data
    X, y = make_classification(n_samples=500, n_features=10, 
                               n_informative=5, n_classes=3,
                               random_state=42)
    
    # One-hot encode labels
    y_onehot = np.zeros((len(y), 3))
    y_onehot[np.arange(len(y)), y] = 1
    
    # Normalize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Split
    split = 400
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_onehot[:split], y_onehot[split:]
    y_test_labels = y[split:]
    
    # Train
    nn = NeuralNetwork(input_size=10, hidden_size=32, output_size=3, lr=0.05)
    nn.fit(X_train, y_train, epochs=500, batch_size=32)
    
    # Evaluate
    train_acc = nn.accuracy(X_train, y_train)
    test_acc = nn.accuracy(X_test, y_test)
    
    print(f"\nFinal Train Accuracy: {train_acc:.4f}")
    print(f"Final Test Accuracy:  {test_acc:.4f}")
    
    # Plot loss
    plt.figure(figsize=(8, 5))
    plt.plot(nn.losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Neural Network Training Loss')
    plt.savefig('nn_training_loss.png', dpi=150)
    plt.show()
    
    print("✅ Neural Network from scratch complete!")
