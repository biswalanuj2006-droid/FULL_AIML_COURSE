# Module 16: Machine Learning From Scratch

Implementing core algorithms with pure Python + NumPy. The fastest way to
actually understand what the libraries do under the hood.

## What You Will Learn

- Linear regression via normal equation and gradient descent
- Logistic regression: sigmoid, likelihood, BCE gradients
- KNN and K-Means implemented by hand
- Decision trees (CART with Gini/entropy splits)
- Naive Bayes with Gaussian likelihoods
- PCA via covariance eigendecomposition / SVD
- Random forest and gradient boosting fundamentals
- Neural network forward/backward by hand (bridge to Module 19)
- Parity testing: your implementation vs sklearn, explained gaps

## Module Files

| File | Topic |
|------|-------|
| from_scratch_ml.txt | Walkthrough of every algorithm |
| practice.txt | Exercises |
| project.txt | Level 1-3 implementation projects |

Runnable code (all in `code/` and `code/ml/`):
01 linear regression, 02 logistic, 03 knn, 04 kmeans,
05 decision tree, 06 neural network, ml/pca, ml/random_forest,
ml/gradient_boosting, ml/gradient_descent, ml/softmax_cross_entropy
— each has a matching `*_practice.txt`.

## Prerequisites

- 02_NUMPY solid, 05_MATHEMATICS (gradients, linear algebra)
- Best taken alongside 09-13 as reference

## Exit Criteria

- [ ] You can implement any core algorithm with NumPy unaided
- [ ] Your implementations match sklearn within documented tolerance
- [ ] You can explain the math line-by-line in your own code
