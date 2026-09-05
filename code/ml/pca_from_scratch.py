# PCA FROM SCRATCH
# Principal Component Analysis implementation

import numpy as np
import matplotlib.pyplot as plt

class PCA:
    """
    Principal Component Analysis.
    
    Algorithm:
    1. Center data (subtract mean)
    2. Compute covariance matrix
    3. Compute eigenvalues and eigenvectors
    4. Sort by eigenvalue (largest first)
    5. Project data onto top-k eigenvectors
    """
    
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.explained_variance_ratio = None
    
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Step 1: Center data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Step 2: Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Step 3: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Step 4: Sort by eigenvalue (descending)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        
        # Step 5: Select top-k components
        self.eigenvalues = eigenvalues[:self.n_components]
        self.components = eigenvectors[:, :self.n_components].T
        
        # Compute explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues[:self.n_components] / total_variance
        
        return self
    
    def transform(self, X):
        X_centered = X - self.mean
        return X_centered.dot(self.components.T)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_reduced):
        return X_reduced.dot(self.components) + self.mean


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print(f"Original shape: {X.shape}")
    print(f"Reduced shape: {X_pca.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio):.4f}")
    
    # Plot
    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green']
    for i in range(3):
        mask = y == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], 
                   label=iris.target_names[i], alpha=0.7)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio[1]:.2%} variance)')
    plt.title('PCA - Iris Dataset')
    plt.legend()
    plt.savefig('pca_demo.png', dpi=150)
    plt.show()
    
    print("[OK] PCA from scratch complete!")
