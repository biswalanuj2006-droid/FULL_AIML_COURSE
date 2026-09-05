# K-MEANS CLUSTERING FROM SCRATCH
# Complete implementation with visualization

import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    """
    K-Means Clustering Algorithm.
    
    Algorithm:
    1. Initialize K random centroids
    2. Repeat until convergence:
       a. Assign each point to nearest centroid
       b. Update centroids as mean of assigned points
    3. Return final centroids and assignments
    """
    
    def __init__(self, k=3, max_iterations=100, tolerance=1e-4):
        self.k = k
        self.max_iter = max_iterations
        self.tolerance = tolerance
        self.centroids = None
        self.labels = None
        self.inertia_history = []
    
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly from data points
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices].copy()
        
        self.inertia_history = []
        
        for iteration in range(self.max_iter):
            # Step 1: Assign points to nearest centroid
            self.labels = self._assign_clusters(X)
            
            # Step 2: Update centroids
            new_centroids = np.array([
                X[self.labels == i].mean(axis=0) if np.sum(self.labels == i) > 0
                else self.centroids[i]
                for i in range(self.k)
            ])
            
            # Compute inertia (sum of squared distances)
            inertia = sum(
                np.sum((X[self.labels == i] - self.centroids[i]) ** 2)
                for i in range(self.k)
            )
            self.inertia_history.append(inertia)
            
            # Check convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tolerance):
                print(f"Converged at iteration {iteration + 1}")
                break
            
            self.centroids = new_centroids
        
        return self
    
    def _assign_clusters(self, X):
        distances = np.array([
            np.linalg.norm(X - centroid, axis=1)
            for centroid in self.centroids
        ])
        return np.argmin(distances, axis=0)
    
    def predict(self, X):
        return self._assign_clusters(X)
    
    def inertia(self):
        return self.inertia_history[-1] if self.inertia_history else 0


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    
    # Generate synthetic data
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
    
    # Fit KMeans
    kmeans = KMeans(k=4, max_iterations=100)
    kmeans.fit(X)
    
    print(f"Final centroids:\n{kmeans.centroids}")
    print(f"Inertia: {kmeans.inertia():.2f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Clustering result
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    for i in range(kmeans.k):
        mask = kmeans.labels == i
        axes[0].scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.6, label=f'Cluster {i}')
    axes[0].scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1],
                    c='black', marker='X', s=200, label='Centroids')
    axes[0].set_title('K-Means Clustering')
    axes[0].legend()
    
    # Elbow method
    inertias = []
    K_range = range(1, 10)
    for k in K_range:
        km = KMeans(k=k, max_iterations=50)
        km.fit(X)
        inertias.append(km.inertia())
    
    axes[1].plot(K_range, inertias, 'bo-')
    axes[1].set_xlabel('K')
    axes[1].set_ylabel('Inertia')
    axes[1].set_title('Elbow Method for Optimal K')
    axes[1].axvline(x=4, color='red', linestyle='--', label='Optimal K=4')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('kmeans_demo.png', dpi=150)
    plt.show()
    
    print("✅ K-Means from scratch complete!")
