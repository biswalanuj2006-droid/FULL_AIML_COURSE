# Module 05: Mathematics for AI/ML

## What You Will Learn

This is a COMPLETE mathematics course designed specifically for AI/ML. It is NOT a generic school math course. Every concept is taught in the context of where it is used in machine learning.

## Why You Need It

Mathematics is the language of machine learning. Without understanding the math:
- You cannot understand HOW algorithms work
- You cannot debug model failures
- You cannot choose the right algorithm
- You cannot read ML papers
- You cannot innovate beyond applying pre-built libraries

With strong math skills:
- You understand WHY each algorithm exists
- You can implement algorithms from scratch
- You can read and understand research papers
- You can make informed engineering decisions
- You can debug models by understanding the mathematics

## Teaching Philosophy

For every difficult concept:

```
Real-world intuition
    ↓
Simple language explanation
    ↓
Meaning of every symbol
    ↓
Tiny numerical example
    ↓
Manual calculation
    ↓
Formula
    ↓
Graph/visualization
    ↓
Python implementation
    ↓
ML application
```

## Prerequisites

- Module 01: Python for AI/ML
- Module 02: NumPy

## Module Structure

| Sub-Module | Topic | Duration |
|------------|-------|----------|
| 01_ALGEBRA | Variables, equations, functions, exponentials, logarithms | 3-4 hours |
| 02_LINEAR_ALGEBRA | Vectors, matrices, dot products, eigenvalues, SVD | 5-6 hours |
| 03_CALCULUS | Derivatives, gradients, chain rule, optimization | 4-5 hours |
| 04_PROBABILITY | Sample space, Bayes theorem, distributions | 4-5 hours |
| 05_STATISTICS | Descriptive stats, inference, hypothesis testing | 3-4 hours |
| 06_OPTIMIZATION | Gradient descent, loss functions, convergence | 3-4 hours |

## Total Estimated Time: 22-28 hours

## Mathematics → ML Map

```
ALGEBRA
├── Variables ────────────→ Feature representation
├── Equations ────────────→ Model equations (y = wx + b)
├── Exponentials ─────────→ Sigmoid, Softmax, Learning rates
├── Logarithms ───────────→ Log loss, Cross entropy
└── Functions ────────────→ Loss functions, activation functions

LINEAR ALGEBRA
├── Vectors ──────────────→ Feature vectors, word embeddings
├── Matrices ─────────────→ Datasets (X), weights (W)
├── Dot Product ──────────→ Linear combination (w·x + b)
├── Matrix Multiply ──────→ Neural network layers
├── Eigenvalues ──────────→ PCA, dimensionality reduction
├── SVD ──────────────────→ Recommender systems, embeddings
└── Norms ────────────────→ Regularization (L1, L2)

CALCULUS
├── Derivatives ──────────→ Slope of loss function
├── Partial Derivatives ──→ Gradient (multi-variable)
├── Chain Rule ───────────→ Backpropagation
├── Gradient ─────────────→ Direction of steepest decrease
├── Hessian ──────────────→ Second-order optimization
└── Integration ──────────→ Probability (area under curve)

PROBABILITY
├── Conditional Prob. ────→ P(class | features)
├── Bayes Theorem ────────→ Naive Bayes classifier
├── Distributions ────────→ Gaussian Naive Bayes, sampling
├── Maximum Likelihood ───→ Logistic Regression training
└── Information Theory ───→ Decision trees, transformers

STATISTICS
├── Mean, Median ─────────→ Feature centering
├── Variance, Std ────────→ Feature scaling
├── Correlation ──────────→ Feature selection
├── Sampling ─────────────→ Train/test split
├── Hypothesis Testing ───→ Model comparison
└── Confidence Intervals ─→ Uncertainty quantification

OPTIMIZATION
├── Loss Functions ───────→ How wrong is the model?
├── Gradient Descent ─────→ How to improve the model
├── Learning Rate ────────→ How fast to improve
├── Momentum ─────────────→ Accelerate convergence
└── Adam ─────────────────→ Adaptive learning rates
```

## Knowledge Checkpoint

After completing this module, you should be able to:

- [ ] Solve linear and quadratic equations
- [ ] Explain what a matrix represents in ML
- [ ] Compute dot products and matrix multiplication
- [ ] Explain eigenvalues and eigenvectors
- [ ] Take derivatives of common functions
- [ ] Explain the chain rule and backpropagation
- [ ] Compute conditional probability and apply Bayes theorem
- [ ] Explain common probability distributions
- [ ] Compute mean, variance, standard deviation
- [ ] Explain gradient descent and convergence
- [ ] Connect every math concept to its ML application
