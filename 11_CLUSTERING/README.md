# Module 11: Clustering

Unsupervised grouping — customer segmentation, anomaly discovery, and the
first step of many real-world pipelines.

## What You Will Learn

- The unsupervised problem and similarity/distance measures
- K-Means: algorithm, k-means++ init, choosing k (elbow/silhouette)
- Hierarchical clustering and dendrograms
- DBSCAN: density-based clusters + noise, eps/min_samples intuition
- Gaussian Mixture Models and soft assignments
- Mean Shift overview
- Clustering evaluation: silhouette, Davies-Bouldin, Calinski-Harabasz
- Distance/similarity choice and feature scaling pitfalls

## Module Files

| File | Topic |
|------|-------|
| clustering_complete.txt | Full theory → math → code progression |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

Related code: `code/04_kmeans_from_scratch.py`.

## Prerequisites

- 06_ML_FUNDAMENTALS
- 05_MATHEMATICS (distances, means, some linear algebra)

## Exit Criteria

- [ ] You can explain why K-Means fails on non-convex shapes
- [ ] You can choose k with evidence, not vibes
- [ ] You can compare sklearn K-Means vs your from-scratch version
- [ ] Projects complete, including a real segmentation problem
