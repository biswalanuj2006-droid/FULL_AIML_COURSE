# ml_course - MASTER PROMPT COMPLIANCE AUDIT

_Checked 2026-09-04 against the pasted "ULTRA-DEEP MACHINE LEARNING COURSE
GENERATOR" prompt. Every row cites where the requirement actually lives.
Status: [OK] fully implemented / [PART] partial - drill work listed /
[GAP] missing in the files - fixed or pointed to below._

## Deliverables inventory

| File | Lines | Contents |
|---|---|---|
| COURSE.txt | 2456 | Modules 0-46 (47 module headers), 16-point topic template, comparison set, error-analysis loops, final-exam blueprint, capstone spec, final audit |
| PRACTICE.txt | 1010 | Levels 0-5 (conceptual E/M/H + math + coding + debugging + dataset + interview per level), FULL EXAM SET (78 items, all 9 sections) + verified answer key |
| PROJECT.txt | 321 | TIER E..S+ (10/10/10/10/10/5/3 = 58 projects) + PRODUCTION ML PLATFORM capstone with acceptance criteria |
| EXAMPLE.py | 795 | 26-section runnable lab; verified 26/26 PASS (75 s) by execution |
| MASTER_PROMPT_AUDIT.md | this | the compliance map |

## The master-prompt checklist, mapped

### Content coverage (the FINAL AUDIT list of the prompt)

| # | Requirement | Status | Location (grep-verified) |
|---|---|---|---|
| 1 | Mathematics | OK | COURSE.txt Module 2 (linear algebra, derivations w = (X^T X)^-1 X^T y, SVD) |
| 2 | Probability | OK | Module 3 (distributions, Bayes, LLN/CLT) |
| 3 | Statistics | OK | Module 4 (tests, CI, power, Type I/II, multiple comparisons) |
| 4 | Information theory | OK | Module 5 (entropy/CE/KL/MI, derived) |
| 5 | NumPy | OK | Module 1 (ndarray, broadcasting, linalg, memory layout) + EXAMPLE.py S1 |
| 6 | Pandas | OK | Module 1 (Series/DataFrame, groupby, merge, pivot, datetime) + EXAMPLE.py S2 |
| 7 | Visualization | OK | Module 1 (matplotlib/seaborn) + per-topic "visualize and why" (EXAMPLE S4, S20 SHAP/ICE) |
| 8 | EDA | OK | Module 7 full workflow + EXAMPLE S4 |
| 9 | Preprocessing | OK | Module 8 (imputation, scaling, encodings, ColumnTransformer/Pipeline) |
| 10 | Data leakage | OK | Module 9 (target/contamination/temporal/preprocessing leakage) + PRACTICE D3/D6/B1/B5 |
| 11 | Regression | OK | Module 11 (derivations + from-scratch) + EXAMPLE S6 |
| 12 | Classification | OK | Module 13 (logistic, OvR/multinomial) + EXAMPLE S8 |
| 13 | Metrics | OK | Module 14 (confusion, PR/ROC/AUC, MCC, log loss) + EXAMPLE S15 |
| 14 | Regularization | OK | Module 12 (Ridge/Lasso/EN objectives derived) |
| 15 | Trees | OK | Module 17 (entropy/Gini/gain, pruning) + EXAMPLE S11 |
| 16 | Ensemble learning | OK | Module 18 (voting/bagging/boosting) + EXAMPLE S12 |
| 17 | XGBoost | OK | Module 18 deep-dive + EXAMPLE S19 |
| 18 | LightGBM | OK | Module 18 (leaf-wise, histograms, GOSS/EFB) |
| 19 | CatBoost | OK | Module 18 (ordered boosting, categoricals) + comparison table |
| 20 | SVM | OK | Module 19 (margin, hinge, kernels, dual conceptually) |
| 21 | Clustering | OK | Module 20 (k-means/hierarchical/DBSCAN) + EXAMPLE S13 |
| 22 | PCA | OK | Module 21 + EXAMPLE S14 (from scratch) |
| 23 | Feature engineering | OK | Module 22 (numerical/categorical/date/text, selection filters/wrappers/embedded) + EXAMPLE S17 |
| 24 | Imbalanced learning | OK | Module 23 (SMOTE/Borderline/ADASYN, class weights, threshold tuning) |
| 25 | Anomaly detection | OK | Module 24 (z-score/IQR/IsolationForest/OC-SVM/LOF) + EXAMPLE S22 |
| 26 | Time series | OK | Module 25 (trend/seasonality/stationarity/lags, AR/ARIMA vs ML) + EXAMPLE S21 |
| 27 | Hyperparameter tuning | OK | Module 26 (Grid/Random/Optuna/Bayesian, nested CV) + EXAMPLE S18 |
| 28 | Calibration | OK | Module 27 (Platt/isotonic, Brier, cost-sensitive thresholds) |
| 29 | Interpretability | OK | Module 28 (permutation, PDP/ICE, SHAP, LIME) + EXAMPLE S20 |
| 30 | Probabilistic ML | OK | Module 29 (MLE/MAP/Bayesian LR, GPs, aleatoric/epistemic) |
| 31 | Causal ML introduced | OK | Module 35 (confounding, DAGs, backdoor, ATE/CATE) |
| 32 | Distribution shift | OK | Module 34 (covariate/label shift, concept drift) |
| 33 | ML security | OK | Module 37 (adversarial, poisoning, model stealing, membership) |
| 34 | Responsible ML | OK | Module 36 (bias sources, fairness metrics, tradeoffs) |
| 35 | Experiment design | OK | Module 38 (hypotheses, ablations, seeds, significance, report format) |
| 36 | ML system design | OK | Module 39 (ingestion -> serving -> monitoring architecture) |
| 37 | FastAPI | OK | Module 40 + PROJECT TIER A projects |
| 38 | Docker | OK | Module 40/41 + PROJECT TIER A |
| 39 | MLflow | OK | Module 41 (tracking, registry) + EXAMPLE S25 (guarded) |
| 40 | Monitoring | OK | Module 42 (drift detection, retraining decision loop) |
| 41 | MLOps | OK | Module 41 (Git/DVC/CI-CD/testing) |
| 42 | Cloud concepts | OK | Module 43 (storage/compute/containers, vendor-neutral) |
| 43 | Research methodology | OK | Module 45 (paper reading path, reproduction, gaps) + Level 5 |
| 44 | From-scratch algorithms | OK | Module 16 + EXAMPLE S6-S14 (OLS, GD, logreg, KNN, NB, CART, RF, k-means, PCA; SVM/GBDT conceptual + GBDT in S19) |
| 45 | Practice aligned with course | OK | PRACTICE.txt levels map 1:1 to COURSE modules |
| 46 | Projects aligned with course | OK | PROJECT.txt tier map mirrors module order |
| 47 | Python examples aligned | OK | EXAMPLE.py S1-S26 match module sequence |
| 48 | Final capstone included | OK | PROJECT.txt TIER S+ (full spec, milestones, acceptance criteria) |

### Structural requirements of the prompt

| Requirement | Status | Location / note |
|---|---|---|
| Course roadmap AI/ML/DL/learning paradigms + full ML lifecycle | OK | Module 0 + COURSE.txt:33-98 |
| "For every library: what/why/APIs/examples/when NOT to use" | OK | Module 1 library-mastery section + COURSE.txt:112 research connection template |
| 16-point topic template (intuition -> research) applied to every major topic | OK | 16 numbered points used across Modules 11-28 (e.g. FAILURE MODES at :449/:673, PSEUDOCODE :641, RESEARCH CONNECTION :112/:690) |
| MODEL COMPARISON tables | OK | Module-level tables (RF/XGB/LGBM/CatBoost :Module 18, k-means/DBSCAN/hierarchical, PCA/t-SNE/UMAP, bagging vs boosting, parametric vs non-parametric, generative vs discriminative, online vs batch) |
| ERROR ANALYSIS playbook | OK | COURSE.txt:2360 (workflow embeds it) + PRACTICE D1-D8 |
| Interview preparation by level (Beginner 100+/Intermediate 150+/Advanced 150+/Expert 100+) | OK (added this audit) | INTERVIEW_BANK.md: 500 Q&A rows (B100/I150/A150/X100, grep-verified) + in-level interview blocks (~60) + full-exam essays |
| FINAL EXAM sections 1-9 (math/statistics/theory/algorithms/coding/data/debugging/system/research) + MCQs/derivations/case studies | OK (added this audit) | PRACTICE.txt "FULL EXAM SET": 78 items across all 9 sections with answer key; MCQs in T2/T7/T10/S10 |
| LEETCODE + ML coding (arrays/hashmaps/sorting/graphs/DP ... why DSA matters) | OK (added this audit) | DSA_FOR_ML.md: 9 DSA topics taught through the ML lens (prefix sums -> rolling stats, heaps -> top-k/beam search, tries -> tokenizers, union-find -> clustering, DP -> Viterbi/DTW/edit distance) + 30-45 min/day plan + self-check. Course-root 53_DSA + 49_LEETCODE remain as the parallel deep dive |
| Case studies by industry (finance/health/e-com/cyber/manufacturing/...) | OK | Fraud/medical/spam cases in Module 14; anomaly cases in Module 24; fraud API in exam X1/D2; PROJECT TIER B/C real-domain projects |
| RESUME + internship prep (GitHub structure, READMEs, model cards, portfolio, interview rounds) | OK (added this audit) | PORTFOLIO_KIT.md: GitHub repo template, README/model-card/experiment-report templates, quantified resume bullets, 8-week interview roadmap, application checklist |
| Visualization REQUIREMENT | OK | Module 1 + "what to visualize & why" per algorithm + EXAMPLE S4/S20 |
| From-scratch minimum 10 algorithms | OK | 9 scratch + GBDT scratch (EXAMPLE S6-S14, S19); SVM conceptually per prompt |
| No-shallow-content rule / 6-level depth (child -> research) | OK | The 16-point template is the 6-level standard, applied per topic |
| Cross-file consistency rule | OK | Every major algorithm appears in COURSE theory + PRACTICE Qs + PROJECT tier + EXAMPLE code (spot-checked: OLS, GD, logreg, KNN, NB, CART, RF, k-means, PCA, GBDT, IsolationForest) |
| Teaching depth per concept: 6 levels | OK | Same as 16-point template |

## Remaining drill work (honest list)

1. ~~Interview bank scale-up~~ DONE - INTERVIEW_BANK.md adds 500 compact Q&A
   rows: Beginner 100 / Intermediate 150 / Advanced 150 / Expert 100
   (verified by count: grep -cE '^[BIXA][0-9]+' = 500), on top of the ~60
   in-level interview questions and the 78-item exam.
2. ~~DSA/LeetCode for ML~~ DONE - DSA_FOR_ML.md (9 topics, ML lens, plan +
   self-check).  Course-root 53_DSA + 49_LEETCODE remain as the parallel deep
   dive for anyone who wants the raw algorithm grind.
3. ~~Resume/portfolio kit~~ DONE - PORTFOLIO_KIT.md (GitHub structure,
   README/model-card/experiment-report templates, resume bullets, 8-week
   interview roadmap, application checklist).
4. **More verified maths** - extend the VERIFIED ANSWERS block as you solve new
   level items; the recompute habit is the point.

## Verification evidence

- EXAMPLE.py ran end-to-end: 26/26 sections PASS, 0 FAIL (~75 s) - recorded in
  COURSE_AUDIT.txt section 15 (three real bugs caught and fixed by running:
  RF depth-4 underfit, a false 1-step time-series claim, invalid sklearn
  make_classification defaults).
- Every numerical answer in PRACTICE.txt VERIFIED ANSWERS was independently
  recomputed; four first-draft slips were corrected and annotated (L1-M2,
  L1-M5, L2-M3, L3-M10).
- Final suite: verify_course.py VERDICT HEALTHY - compiled 56, ran 33 examples,
  33 pass, 0 fail (ml_course/EXAMPLE.py wired with a 180 s slot).
