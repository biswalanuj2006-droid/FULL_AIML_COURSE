================================================================================
PORTFOLIO KIT  (ml_course/PORTFOLIO_KIT.md)
================================================================================
Companion to:  COURSE.txt  PRACTICE.txt  PROJECT.txt  EXAMPLE.py
Track     :  ml_course  - the master-prompt "Resume + Internship Preparation"
              requirement: turning ML knowledge into employable proof.
Audience  :  2nd-year B.Tech AI & ML student.

This file is a WORKBOOK: every section has templates you fill in. Do the
work here, then your GitHub + resume practically write themselves.

================================================================================
1. THE ONE-PARAGRAPH THEORY OF A PORTFOLIO
================================================================================
Recruiters and hiring managers look for EVIDENCE, not claims. A portfolio
project earns trust when it shows:

  1. A real problem with a measurable objective.
  2. Data you can describe (size, source, leakage controls).
  3. Baselines BEFORE your model.
  4. A model + why that model (not just "accuracy 0.9").
  5. Error analysis: where it fails and why.
  6. Engineering: how it would run in production (API, monitoring).
  7. Honest limitations.

One deep, well-documented project beats five shallow notebooks. Aim for
3-4 projects at increasing depth: one exploratory analysis, one classical-ML
system, one deep-learning / LLM system, one production-shaped service.

================================================================================
2. GITHUB PROJECT STRUCTURE (template)
================================================================================
Every project repo should look like this:

  project-name/
  ├── README.md                  <- the front page (Section 3)
  ├── requirements.txt           <- pinned, minimal
  ├── LICENSE                   <- MIT unless you have a reason not to
  ├── .gitignore                <- data/, __pycache__/, .env, checkpoints/
  ├── data/
  │   ├── raw/                  <- never edit; document the source
  │   ├── processed/            <- produced by scripts (reproducible)
  │   └── README.md             <- provenance notes for every dataset
  ├── notebooks/
  │   ├── 01_eda.ipynb          <- numbered: chronological reasoning
  │   ├── 02_baselines.ipynb
  │   └── 03_experiments.ipynb
  ├── src/
  │   ├── __init__.py
  │   ├── data.py               <- loading + cleaning (leakage-safe)
  │   ├── features.py           <- feature engineering (fit on train only!)
  │   ├── models.py             <- model definitions
  │   ├── train.py              <- training entry point (seeded)
  │   ├── evaluate.py           <- metrics + error analysis
  │   └── predict.py            <- single-sample prediction for serving
  ├── tests/
  │   ├── test_data.py
  │   ├── test_features.py      <- e.g. no future leakage in rolling features
  │   └── test_models.py
  ├── reports/
  │   ├── experiment_report.md  <- Section 5 template
  │   └── model_card.md         <- Section 6 template
  ├── api/
  │   └── app.py                <- FastAPI wrapper (PROJECT.txt capstone)
  └── docker/
      └── Dockerfile

Rules that make it credible:
  * `python src/train.py` must reproduce the README's headline numbers from
    `data/raw` alone (fixed seed, fixed splits).
  * Never commit data > a few MB or any credentials (.env ignored).
  * Every notebook should run top-to-bottom (or say why it can't).

================================================================================
3. README TEMPLATE (fill the brackets)
================================================================================
# [Project name]

One sentence: what it does and for whom.

## Highlights
- Headline metric with the BASELINE next to it, e.g. "F1 0.87 vs majority-
  class baseline 0.52 on [dataset]".
- 1-3 bullet proof points (what you built, what you measured, what you learned).

## Problem
Why this matters; who has this problem; what a wrong prediction costs.

## Data
- Source, size (rows x features), time span.
- Label definition + how labels were validated.
- Leakage controls used (temporal split? group split? preprocessing fit on
  train only?).

## Approach
1. Baselines (Dummy*, linear).
2. Model family + why (with complexity/sample-size reasoning).
3. Validation design (CV scheme and why it matches deployment).
4. Key features and why (SHAP or ablation).

## Results
| Model | Val metric | Test metric | Notes |
|---|---|---|---|
| Baseline | ... | ... | ... |
| Yours | ... | ... | ... |

## Error analysis
- Where does it fail? (show 3-5 concrete bad cases)
- What would fix it? (more data of type X, feature Y, threshold tuning)

## Reproduce
```bash
pip install -r requirements.txt
python src/train.py          # ~5 min on CPU, writes models/ and reports/
python src/evaluate.py       # prints the test table
```

## Production notes
- Latency/throughput measured; API sketch; monitoring (drift) plan; costs.

## Limitations
Honest list (data bias, domain limits, what was NOT validated).

## License / credits / dataset attribution

================================================================================
4. RESUME BULLET CONSTRUCTION
================================================================================
Formula per project (use numbers everywhere):
  [ACTION] + [WHAT/TOOL] + [MEASURED OUTCOME, baseline vs final] + [SCOPE]

Weak:    "Built a churn prediction model using Random Forest."
Better:  "Built a churn model (XGBoost, 22 features, 480k rows) lifting
          test AUC 0.71 -> 0.84 over a logistic baseline; deployed as a
          FastAPI endpoint with request logging and drift alerts."
Why it works: model+data scale, comparison vs baseline, deployment verbs.

Bullet menu by skill (mix 4-6 on your resume):
  * Data: "Cleaned 1.2M-row sales log: dedup, outlier policy, temporal
           train/val/test split that cut leakage-driven AUC inflation by 0.09."
  * Modeling: "Benchmarked LR/RF/GBDT with 5x2 CV; selected GBDT (0.83 F1 vs
           0.52 majority baseline) using nested CV to control tuning overfit."
  * Engineering: "Served the model via FastAPI (p50 12 ms, 400 req/s on a
           laptop), containerized with Docker, logged inputs + predictions."
  * MLOps: "Tracked 140 runs in MLflow; registered the champion; built a
           drift check comparing train vs live feature distributions (PSI)."
  * Communication: "Wrote a model card + experiment report; presented error
           analysis to a mock stakeholder and turned it into 3 data asks."

Rules: 1 line each, past tense, no unexplained acronyms, numbers you can
defend in an interview.

================================================================================
5. EXPERIMENT REPORT TEMPLATE
================================================================================
# Experiment report: [hypothesis in one line]

## Question
What are you trying to learn? (not: "does X work" - : "does X beat Y by more
than noise on Z?")

## Setup
- Data + split (exact seed, sizes).
- Models/configs compared (all hyperparameters).
- Metric + why (with class imbalance/ cost asymmetry reasoning).
- Compute budget and runtime per run.

## Results
Table with mean +/- std over repeats/CV folds. Include baselines.

## Analysis
1. Is the difference statistically meaningful? (confidence interval /
   paired test; Module 38)
2. Error analysis of the best model: 3 concrete failure cases.
3. Ablations: which component matters? (remove features, revert to simple
   model, change threshold)

## Conclusion + next steps
Decide with evidence; list the 2-3 cheapest experiments that would add the
most information next.

================================================================================
6. MODEL CARD TEMPLATE
================================================================================
# Model card: [name]
- Model details: type, framework, params, trained by, date.
- Intended use: the exact setting + who the users are.
- Out-of-scope uses: what it must NOT be used for.
- Training data: source, size, collection method, known biases.
- Evaluation: metrics, test set construction, how it was kept clean.
- Fairness: results across groups you could measure; known gaps.
- Limitations: failure modes, domains it never saw.
- Deployment: latency, throughput, monitoring plan, rollback plan.
- Contact / maintainer.

(Use this template for the FINAL CAPSTONE model and every PROJECT.txt
"S+" project.)

================================================================================
7. INTERVIEW PREP ROADMAP (tie to PRACTICE.txt + INTERVIEW_BANK.md)
================================================================================
Phase 1 (weeks 1-2): theory fundamentals
  - Explain every algorithm in COURSE.txt in 2 minutes with whiteboard math:
    linear/logistic regression, trees, RF, GBDT, kNN, SVM, k-Means, PCA.
  - Drill INTERVIEW_BANK.md Beginner 100 + Intermediate 150.

Phase 2 (weeks 3-4): ML coding rounds
  - Implement from memory without libraries: linear regression (normal
    equation + GD), logistic regression, kNN, Naive Bayes, decision tree,
    k-Means, PCA (EXAMPLE.py Sections 6-14 are your flashcards).
  - DSA: 2 problems/day from DSA_FOR_ML.md topics most relevant to ML.

Phase 3 (weeks 5-6): system design + debugging
  - Practice the capstone architecture interview: "design a churn alerting
    system" - data -> features -> model -> serving -> monitoring -> cost.
  - Debugging drills: given a broken pipeline (shape mismatch, leakage,
    NaN loss, drift), diagnose in 5 minutes (PRACTICE.txt debugging levels).

Phase 4 (weeks 7-8): mock interviews + storytelling
  - Record yourself explaining your top project end-to-end (90 seconds +
    questions). Use the README + report as your script.
  - Answer "tell me about a time a model failed": pick a real one from your
    error analysis and tell the fix story.

================================================================================
8. WHAT INTERVIEWERS ACTUALLY SCREEN FOR (honest list)
================================================================================
  1. Can you state your metric and baseline from memory? (screens 50%)
  2. Do you know why YOUR model choice fits YOUR data size?
  3. Can you code a from-scratch model under time pressure?
  4. Do you know what leakage is and where it hides?
  5. Can you design an eval that matches deployment?
  6. Can you explain a failure with data, not vibes?
  7. Can you talk about cost/latency/monitoring, not just accuracy?
  8. Do your GitHub repos have a README a stranger could follow?

================================================================================
9. APPLICATION CHECKLIST (before you send any application)
================================================================================
[ ] Resume: 1 page, 4-6 quantified bullets, no unexplained acronyms.
[ ] GitHub: 3-4 repos with Section-3 READMEs; pinned the best one.
[ ] One experiment report + one model card visible per flagship repo.
[ ] Link to the flagship repo in the resume header, not buried.
[ ] Practice answers for: every number on your resume.
[ ] A 90-second "walk me through your best project" you've recorded once.
[ ] 3 questions to ask the interviewer (about data, evaluation, or their
    stack - never salary in round 1).
================================================================================
END PORTFOLIO KIT
================================================================================
