# Module 12: Dimensionality Reduction

Fighting the curse of dimensionality: compressing features while keeping the
signal — and visualizing high-dimensional data.

## What You Will Learn

- Why high dimensions hurt: sparsity, distance concentration, compute
- PCA: covariance, eigenvectors, variance explained, projections
- SVD relationship to PCA; TruncatedSVD for sparse data
- Linear vs manifold methods: t-SNE, UMAP (visualization), ICA overview
- Feature extraction vs feature selection (Module 15)
- Choosing n_components; reconstruction error and scree plots
- Pitfalls: scaling before PCA, leakage in pipelines, misreading t-SNE

## Module Files

| File | Topic |
|------|-------|
| dimensionality_reduction_complete.txt | Full theory → math → code |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

Related code: `code/ml/pca_from_scratch.py`.

## Prerequisites

- 05_MATHEMATICS linear algebra (eigenvalues, SVD intuition)
- 06_ML_FUNDAMENTALS

## Exit Criteria

- [ ] You can explain what PCA maximizes and why scaling matters
- [ ] You can read a scree plot and defend your n_components choice
- [ ] You know when to use PCA vs t-SNE vs UMAP
- [ ] PCA project parity-checked against sklearn
