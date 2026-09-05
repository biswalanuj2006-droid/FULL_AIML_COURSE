# Module 09: Classification

Predicting categories — spam/not-spam, fraud/not-fraud, disease present/absent.
The most common supervised task in industry.

## What You Will Learn

- The classification problem and decision boundaries
- Logistic Regression: sigmoid, odds, log-loss, decision rule
- KNN, Naive Bayes, Decision Trees, Random Forest, SVM
- Gradient boosting family (sklearn, XGBoost/LightGBM overview)
- Class imbalance: why accuracy lies, and what to do about it
- Probabilities vs hard labels; threshold choice and calibration
- Evaluation for classification (cross-referenced with Module 14)

## Module Files

| File | Topic |
|------|-------|
| classification_complete.txt | Full theory → math → code progression |
| imbalanced_learning.txt | Imbalance: threshold, class weights, SMOTE, evaluation |
| svm_deep_dive.txt | SVM: primal/dual/KKT derivation, kernels, SVR, complexity |
| failure_lab.txt | Reproduce → diagnose → fix failure exercises |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

Related runnable code: `code/02_logistic_regression_from_scratch.py`,
`code/03_knn_from_scratch.py`, `code/05_decision_tree_from_scratch.py`.

## Prerequisites

- 06_ML_FUNDAMENTALS, 07_DATA_PREPROCESSING
- Basic 05_MATHEMATICS (sigmoid, log, probability basics)

## Exit Criteria

- [ ] You can pick a classifier given data size, noise, and interpretability needs
- [ ] You know when accuracy is the wrong metric
- [ ] You have completed the 3 projects, comparing sklearn vs from-scratch
