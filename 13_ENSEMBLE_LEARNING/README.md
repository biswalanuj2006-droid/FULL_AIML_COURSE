# Module 13: Ensemble Learning

Why a crowd of weak models beats a single strong one — and the three ways to
build the crowd.

## What You Will Learn

- Why ensembles reduce error: bias/variance decomposition intuition
- Bagging: bootstrap sampling, Random Forest, Extra Trees
- Boosting: AdaBoost → Gradient Boosting → XGBoost/LightGBM/CatBoost
- Stacking and voting
- Random Forest vs XGBoost vs LightGBM vs CatBoost: when each wins
- Hyperparameters that matter for each family
- Feature importance and interpretability of ensembles

## Module Files

| File | Topic |
|------|-------|
| ensemble_complete.txt | Theory → math → code progression |
| gradient_boosting_libraries.txt | XGBoost/LightGBM/CatBoost deep dive |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

Related code: `code/05_decision_tree_from_scratch.py`,
`code/ml/random_forest_from_scratch.py`, `code/ml/gradient_boosting_from_scratch.py`.

## Prerequisites

- 09_CLASSIFICATION and 10_REGRESSION (base learners)
- 14_MODEL_EVALUATION recommended in parallel

## Exit Criteria

- [ ] You can explain bagging vs boosting with variance/bias language
- [ ] You can name the key hyperparameter of each major ensemble
- [ ] You can choose Random Forest vs XGBoost for a given dataset
- [ ] All projects complete with honest benchmark comparisons
