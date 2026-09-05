# Module 07: Data Preprocessing

## What You Will Learn

- Handling missing values (detection, imputation strategies)
- Removing duplicates
- Outlier detection and treatment
- Encoding categorical variables (one-hot, ordinal, target)
- Feature scaling (standardization, normalization)
- Train/test splitting
- Cross-validation
- Building preprocessing pipelines
- Data leakage prevention

## Why You Need It

Raw data is NEVER ready for ML models. It has missing values, inconsistent types, categorical text, different scales, and outliers. Data preprocessing is 80% of the ML workflow. The quality of your preprocessing directly determines model performance.

**The most common ML mistake is NOT bad model choice — it's bad preprocessing.**

## Prerequisites

- Module 01: Python for AI/ML
- Module 02: NumPy
- Module 03: Pandas
- Module 06: ML Fundamentals (sklearn_deep_dive.txt for preprocessing reference)

## What You Will Be Able to Do After This Module

1. Detect and handle every type of data quality issue
2. Choose the right imputation strategy for each situation
3. Encode categorical variables correctly
4. Scale features appropriately for each algorithm
5. Build sklearn Pipelines that prevent data leakage
6. Split data using stratification and cross-validation

## Module Files

| File | Topic | Duration |
|------|-------|----------|
| 01_missing_values.txt | Detection, imputation strategies | 2-3 hours |
| 02_duplicates.txt | Detection, removal, impact | 1 hour |
| 03_outliers.txt | Detection methods, treatment | 2-3 hours |
| 04_encoding.txt | Categorical encoding methods | 2-3 hours |
| 05_scaling.txt | Feature scaling and normalization | 2-3 hours |
| 06_train_test_split.txt | Train/test split and cross-validation | 2-3 hours |
| 07_pipelines.txt | sklearn Pipeline, ColumnTransformer | 3-4 hours |
| 08_data_leakage.txt | What it is, how to prevent it | 2-3 hours |

## Total Estimated Time: 17-23 hours

## The Preprocessing Pipeline

```
Raw Data
    ↓
1. Inspect (shape, dtypes, missing, duplicates)
    ↓
2. Handle Missing Values
    ↓
3. Remove Duplicates
    ↓
4. Handle Outliers
    ↓
5. Encode Categoricals
    ↓
6. Scale Features
    ↓
7. Split Data
    ↓
Clean Data Ready for Modeling
```

**CRITICAL RULE:** Steps 1-6 should be fitted on TRAINING data only, then applied to test data.

## Knowledge Checkpoint

After completing this module, you should be able to:

- [ ] Choose the right imputation strategy for different data types
- [ ] Explain why mean imputation can be problematic
- [ ] One-hot encode categorical variables correctly
- [ ] Choose StandardScaler vs MinMaxScaler vs RobustScaler
- [ ] Build a ColumnTransformer for mixed data types
- [ ] Prevent data leakage in preprocessing
- [ ] Explain what data leakage is and why it matters
