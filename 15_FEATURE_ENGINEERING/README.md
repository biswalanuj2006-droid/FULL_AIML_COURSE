# Module 15: Feature Engineering

Where most real-world model performance actually comes from — turning raw
messy data into features models can learn from.

## What You Will Learn

- Numerical transforms: log/Box-Cox, clipping, scaling
- Polynomial and interaction features
- Binning, encoding (one-hot, ordinal, target-aware — and leakage risks)
- Date/time features: lags, rolling stats, cyclical encoding
- Text features into ML: TF-IDF vectors, embedding features
- Aggregation features over groups (per-user means, counts)
- Domain features and when features beat better models
- Feature selection: filter, wrapper, embedded (SelectKBest, RFE, importance)
- Pipelines/ColumnTransformer to keep it leakage-free and repeatable

## Module Files

| File | Topic |
|------|-------|
| feature_engineering_complete.txt | Full guide with real examples |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

## Prerequisites

- 07_DATA_PREPROCESSING, 09/10 (to measure feature impact)

## Exit Criteria

- [ ] You can improve a baseline measurably with features — and prove it
- [ ] You know which encodings leak and how to avoid it
- [ ] You build features inside a Pipeline, never by hand on full data
