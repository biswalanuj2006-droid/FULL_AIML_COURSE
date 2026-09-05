# Module 59: Recommender Systems

Ranking what a user should see next - the economics engine of
e-commerce, media, and social platforms - from baselines to
two-stage production recommenders.

## What You Will Learn

- Explicit vs implicit feedback; why absence is ambiguous
- Popularity / co-occurrence baselines (the bar every model must beat)
- Matrix factorization: Funk SGD and ALS with full math
- Implicit-feedback ALS (Hu-Koren confidence weighting)
- Ranking evaluation: HitRate@k, NDCG, MAP; time-based splits
- Cold start for new users and items; hybrid + content bootstrap
- Two-stage architecture: candidate generation + ranking
- Two-tower embeddings and ANN serving (modern retrieval)
- Failure modes: feedback loops, position bias, leakage, drift

## Module Files

| File | Topic |
|------|-------|
| recommender_systems_complete.txt | Full course: math -> code -> production |
| practice.txt | Exercises (12 items, math + code + design) |
| project.txt | Level 1-3 projects |
| think.txt | Hard reasoning incl. OPEN PROBLEM items |

## Code Examples (verified by execution)

| File | What it shows | Result |
|------|---------------|--------|
| code/recommenders/01_matrix_factorization_sgd.py | Funk-SGD on explicit ratings | test RMSE 0.56 vs 0.87 global-mean baseline |
| code/recommenders/02_als_vs_sgd.py | ALS vs SGD on the same ratings (reg, passes tuned) | ALS test RMSE 0.476 vs SGD 0.486; ALS ~6x faster to converge here |

Both run with numpy only: `python code/recommenders/0X_*.py`. The comments
document two real bugs the runs exposed (ratings clipped onto the scale
floor, and an ordered split that gave test users zero training data).

## Prerequisites

- 06_ML_FUNDAMENTALS, 09_CLASSIFICATION, 10_REGRESSION
- Linear algebra comfort (05_MATHEMATICS/02)
- 30_VECTOR_DATABASES helps for the ANN serving parts

## Exit Criteria

- [ ] You can derive the Funk-SGD and ALS updates
- [ ] You can evaluate a recommender honestly (ranking metrics, time split)
- [ ] You can sketch a two-stage production architecture with cold-start handled
- [ ] You have completed project 1-2 and can diagnose a feedback loop

## Interview Relevance

Matrix factorization math, implicit feedback, cold start, two-tower
retrieval, NDCG vs recall, rich-get-richer - all common FAANG ML
questions.
