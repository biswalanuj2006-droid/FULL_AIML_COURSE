# Module 56: Time Series

_(Renumbered from Module 16 on 2026-09-03 to resolve the numbering
collision with 16_ML_FROM_SCRATCH.)_

Data with order — forecasting, anomaly detection, and why "random split"
does not apply when rows are dates.

## What You Will Learn

- Time series structure: trend, seasonality, noise; stationarity
- Decomposition and autocorrelation (ACF/PACF intuition)
- Classical models: AR, MA, ARMA/ARIMA/SARIMA, exponential smoothing
- ML forecasting: feature engineering with lags/rolling windows
- Gradient-boosting forecasters and why they win tabular forecasting
- Deep forecasting: LSTM and Transformer-based approaches overview
- Evaluation: time-based splits, walk-forward validation, backtesting
- Metrics: MAE/RMSE vs MAPE; forecasting-specific scoring
- Anomaly detection on series (thresholds, residuals, isolation)

## Module Files

| File | Topic |
|------|-------|
| time_series_complete.txt | Full theory → math → code |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

Related code: `code/rnn/01_keras_lstm_sine.py` (windowing pattern).

## Prerequisites

- 09/10 (supervised models reused with lag features)
- 05_MATHEMATICS statistics basics

## Exit Criteria

- [ ] You never shuffle time data — and can say why
- [ ] You can build a walk-forward evaluation loop
- [ ] You can beat a naive baseline with a real method (and prove it)
