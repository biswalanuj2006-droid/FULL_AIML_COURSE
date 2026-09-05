# Module 55: Anomaly Detection

_(Renumbered from Module 11 on 2026-09-03 to resolve the numbering
collision with 11_CLUSTERING.)_

Finding the rare, the strange, and the broken - fraud, faults, intrusions,
and outliers that deserve attention. Unsupervised first, supervised when
labels exist.

## What You Will Learn

- Anomaly types: point, contextual, collective
- Statistical methods: z-score, IQR, robust z (MAD), Grubbs
- Distance/density: kNN distance, Local Outlier Factor (LOF)
- Parametric: Mahalanobis distance, Elliptic Envelope (robust
  covariance), Gaussian Mixture density threshold
- Isolation Forest: random partitioning and path-length scoring
- One-class SVM (nu-SVM)
- Autoencoder reconstruction-error approach (concept bridge to DL)
- Time-series anomaly detection (bridge to Module 16)
- Threshold selection and evaluation WITHOUT labels
- Choosing a method by data shape, scale, and contamination
- Failure modes and debugging anomaly systems

## Module Files

| File | Topic |
|------|-------|
| anomaly_detection_complete.txt | Full theory → math → code progression |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |
| think.txt | Deep reasoning problems |

## Prerequisites

- 07_DATA_PREPROCESSING (outliers lesson), 10_REGRESSION
- 05_MATHEMATICS statistics (Gaussian, covariance) and distances
- 18_ANN optional but helpful for the autoencoder section

## Exit Criteria

- [ ] You can classify an anomaly type and pick a matching method
- [ ] You can explain isolation forest's scoring formula
- [ ] You can evaluate an unsupervised detector honestly (no labels)
- [ ] All three projects complete (fraud, monitoring, series anomalies)
