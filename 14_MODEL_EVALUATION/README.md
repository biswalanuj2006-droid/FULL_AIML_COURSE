# Module 14: Model Evaluation & Selection

The discipline that stops you from deploying a model that only looks good:
splits, cross-validation, metrics, and honest comparison.

## What You Will Learn

- Train/validation/test and why the test set must stay untouched
- K-fold / stratified / grouped cross-validation; leakage in CV
- Classification metrics: accuracy, precision, recall, F1, specificity
- Confusion matrix, ROC/PR curves, AUC, log loss, calibration
- Regression metrics: MAE, MSE, RMSE, R², MAPE — and their traps
- Clustering metrics: silhouette, Davies-Bouldin, Calinski-Harabasz
- Threshold selection and cost-sensitive evaluation
- Comparing models: paired tests, error analysis, learning curves
- How bad evaluation silently ruins ML projects

## Module Files

| File | Topic |
|------|-------|
| model_evaluation_complete.txt | Full theory → math → code |
| hyperparameter_optimization.txt | Grid/random/Bayesian/HPO discipline + Optuna |
| xai_interpretability.txt | SHAP/LIME/PDP/ICE/counterfactuals + why explanations can lie |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

## Prerequisites

- 09_CLASSIFICATION / 10_REGRESSION (you need models to evaluate)

## Exit Criteria

- [ ] You can choose the right metric for a business problem
- [ ] You can explain ROC-AUC and PR-AUC and when PR wins
- [ ] You can design a leakage-free CV setup
- [ ] Every project in later modules reports metrics honestly using this module
