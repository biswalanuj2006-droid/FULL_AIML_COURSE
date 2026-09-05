================================================================================
ULTIMATE PROJECT PLAYBOOK - AI/ML ENGINEERING COURSE
================================================================================
How to attack ANY project or real problem in this course: how to FIGURE
OUT what is wrong, and how to SOLVE it - with depth, not vibes.

Companion files: every key module has real_problem.txt (worked real-world
diagnoses) and failure_lab.txt (reproduce -> diagnose -> fix labs). This
file is the master method those files and all project.txt specs follow.

--------------------------------------------------------------------------------
PART 1 - THE UNIVERSAL METHOD (use on EVERY project)
--------------------------------------------------------------------------------
Do not jump to code. Every problem, from "spam filter too weak" to "RAG
gives wrong answers", follows the same 7 steps:

  STEP 1. DEFINE THE FAILURE PRECISELY.
          "It doesn't work" is not a problem statement. Write the
          symptom as a measurement: "precision@k on 100 hand-labeled
          queries is 0.4" or "validation 0.94 / live 0.61". If you
          cannot measure it, you cannot debug it - build the
          measurement FIRST (see STEP 3).

  STEP 2. FORM THE HYPOTHESIS SPACE (do NOT pick one and code it).
          A wrong output has a bounded number of causes. Before
          touching code, list every stage your system has and the ways
          each stage can fail. Examples:
            pipeline: data -> preprocessing -> features -> model
                      -> evaluation -> deployment -> monitoring
            RAG:      ingest -> chunk -> embed -> store -> retrieve
                      -> rank -> prompt -> generate
          For each stage write 2-4 concrete ways it could fail. Now you
          have a checklist, not a guess.

  STEP 3. MEASURE STAGE BY STAGE (bisect).
          Instrument each stage boundary with a cheap metric:
            - data: row count, missing rate, label balance, checksum
            - preprocessing: range of output vs input (scaler fitted?)
            - retrieval: "does the gold answer appear in top-k chunks?"
            - model: train loss vs val loss vs live score
            - serving: latency, output dtype, NaN rate
          Then bisect: run the metric at the MIDDLE stage. If fine,
          the bug is downstream; if broken, it is upstream. Repeat.
          Every hard bug dies in 3-4 bisections. This is the single
          most valuable habit in this course.

  STEP 4. REPRODUCE ON THE SMALLEST INPUT THAT FAILS.
          Minimize: one batch, 10 questions, 5 documents, 1 feature.
          A bug that only reproduces on the full dataset is a bug you
          cannot see. Shrink until you can hold the failure in your
          head, then fix it there.

  STEP 5. FIX THE ROOT CAUSE, THEN PROVE THE FIX.
          Fix the stage that measurement (STEP 3) implicated - not the
          stage you feel is guilty. Then re-run the SAME measurement
          and show the number moved. No measurement change = no fix.

  STEP 6. ADD THE REGRESSION GUARD.
          Turn the failure into a permanent test: golden questions,
          a unit test on the preprocessing function, a drift alert.
          An unfixed-by-design system will relapse.

  STEP 7. DOCUMENT THE POST-MORTEM.
          Symptom -> hypothesis space -> measurement that caught it ->
          root cause -> fix -> guard. Two lines each. This is what
          turns experience into expertise (and interview stories).

--------------------------------------------------------------------------------
PART 2 - THE FAILURE TAXONOMY (symptom -> how to figure out -> how to solve)
--------------------------------------------------------------------------------
These 12 failure classes cover ~95% of everything that goes wrong in
the projects in this course. Match your symptom, then follow the row.

1. THE PHANTOM SCORE
   Symptom:  accuracy > 95% and it feels too good to be true. It is.
   How to figure out: check label leakage first - does any feature
     contain the answer (a joined future column, an ID shared with the
     label, the label itself in a "predicted_" field)? Then check
     duplicate rows between train and test (dedupe by content hash).
   How to solve: remove the leaked field; split BEFORE any join/encode/
     scale; hold out by GROUP (user, session, time) not by row.
   Where it lives: 09/real_problem.txt, 09/failure_lab.txt,
     07_DATA_PREPROCESSING pipelines+leakage lessons.

2. TRAIN/VAL PERFECT, PRODUCTION GARBAGE
   Symptom: notebook hero, live zero.
   How to figure out: score a logged sample of live requests against
     delayed ground truth; compare live feature ranges to training
     ranges (drift); diff the notebook preprocessing against the
     serving code line by line.
   How to solve: deploy the whole pipeline object, not the bare model;
     freeze preprocessing; schema-validate at the API edge; shadow-deploy
     new versions.
   Where it lives: 34_MODEL_DEPLOYMENT/real_problem.txt,
     33_MLOPS/real_problem.txt, 46_PRODUCTION_AI, 43_TESTING.

3. THE CONFIDENT LIAR (hallucination / ungrounded output)
   Symptom: fluent answer, wrong facts (LLM/RAG systems).
   How to figure out: retrieval hit-rate test - is the answer in the
     top-k chunks at all? If yes and the model still errs, it is
     generation (memory overriding context); if no, it is retrieval.
   How to solve: see 29_RAG/real_problem.txt - hybrid search, chunk
     overlap + titles, grounded prompt that permits "not in context",
     faithfulness self-check, eval harness of 100 golden questions.
   Where it lives: 29_RAG, 28_LLM_FUNDAMENTALS, 26_TRANSFORMERS,
     31_PROMPT_ENGINEERING.

4. DATA THAT LIES (quality)
   Symptom: model trains fine, learns the wrong thing, or plateaus low.
   How to figure out: EDA the ERRORS, not the data: cluster 50 wrong
     predictions and read the inputs. Wrong labels? Duplicates?
     Missingness that correlates with the label? Annotator drift?
   How to solve: fix or filter labels, dedupe, impute with missingness
     as a feature, re-audit a label sample per batch.
   Where it lives: 07_DATA_PREPROCESSING, 08_EDA, 06_ML_FUNDAMENTALS,
     10_REGRESSION/failure_lab.txt.

5. THE IMBALANCE TRAP
   Symptom: 98% accuracy, and the rare class is entirely missed
     (fraud, churn, defects, disease).
   How to figure out: print the confusion matrix and per-class
     precision/recall/F1. 98% accuracy with 0 recall on the positive
     class = the model predicts "all negative".
   How to solve: 09_CLASSIFICATION/imbalanced_learning.txt - class
     weights, resampling (SMOTE family), threshold tuning on PR-AUC,
     and metrics that punish the majority guess (MCC, balanced
     accuracy, PR-AUC).
   Where it lives: 09_CLASSIFICATION, 14_MODEL_EVALUATION.

6. OVERFIT (memorizer) / UNDERFIT (lazy mean-predictor)
   Symptom: train >> val (overfit) or train ~= val but both bad
     (underfit).
   How to figure out: learning curves - plot train and val error vs
     dataset size and vs epochs. Divergence = variance (regularize);
     parallel flat lines high up = bias (more features/complexity).
   How to solve: overfit -> regularization, more data, simpler model,
     early stopping, CV. Underfit -> stronger model, better features,
     less regularization.
   Where it lives: 06_ML_FUNDAMENTALS, 14_MODEL_EVALUATION,
     19_DEEP_LEARNING/real_problem.txt, 18_ANN/failure_lab.txt.

7. THE DEGENERATE FORECAST
   Symptom: time-series model has tiny loss and predicts "same as
     last value" (or the seasonal copy) - useless for decisions.
   How to figure out: compare against the persistence baseline
     (predict last value). If your model only ties it, it learned the
     identity shortcut. Model the DIFFERENCE, not the level.
   How to solve: 22_LSTM_GRU/real_problem.txt, 56_TIME_SERIES/
     real_problem.txt - differencing, chronological split, direction
     accuracy, beat-the-baseline as the acceptance bar.
   Where it lives: 56_TIME_SERIES, 22_LSTM_GRU, 21_RNN.

8. NUMERICAL MELTDOWN (NaN / inf / divergent loss)
   Symptom: loss goes NaN, or blows up after N epochs, or never moves.
   How to figure out: detect_anomaly / print loss right after forward
     BEFORE backward; check input for NaN; check label range vs loss
     function; halve the LR - if it trains, it was LR.
   How to solve: gradient clipping, LR schedule + warmup, correct
     scaling (fit on train), correct dtypes, log every epoch.
   Where it lives: 18_ANN/failure_lab.txt, 19_DEEP_LEARNING/
     failure_lab.txt, 20_CNN/real_problem.txt.

9. THE LATENCY/SIZE CRISIS
   Symptom: works but too slow / too big for the requirement.
   How to figure out: profile stage by stage (transform vs predict vs
     serialize); measure with the target hardware; find the 90/10.
   How to solve: load model once, batch, quantize, distill, ONNX,
     cache, smaller max tokens / shorter contexts.
   Where it lives: 26_TRANSFORMERS/real_problem.txt, 34_MODEL_
     DEPLOYMENT/real_problem.txt, 41_DOCKER, 35_FASTAPI.

10. SILENT DRIFT (worked for a month, then decayed)
    Symptom: no crash, no error - metrics just slide or flip.
    How to figure out: you CANNOT debug it without logs. Version +
    log model/preprocessing/data version + input hash per prediction;
    track feature drift on the SHAP-top features, prediction rate
    drift, and delayed ground truth.
    How to solve: 33_MLOPS/real_problem.txt - canary + shadow + alert
      + retraining triggers; registry for rollback.
    Where it lives: 33_MLOPS, 46_PRODUCTION_AI, 34_MODEL_DEPLOYMENT.

11. THE DOMAIN MISMATCH
    Symptom: works on benchmark data, fails on YOUR data (legal text,
      phone photos, your jargon).
    How to figure out: hold out a source-shifted test set and report
      on it; inspect what the model keys on (spurious cues via SHAP);
      try simple baselines that match domain patterns (BM25, character
      n-grams, dictionary pass).
    How to solve: domain adaptation - small labeled sample + fine-tune,
      hybrid rules+ML, robust features, augmentation matched to real
      conditions.
    Where it lives: 23_NLP/real_problem.txt, 20_CNN/real_problem.txt,
      26_TRANSFORMERS/real_problem.txt, 32_FINE_TUNING.

12. THE REPRODUCIBILITY GREMLIN
    Symptom: same script, different results day to day.
    How to figure out: check seed coverage, dependency drift, dataset
      file replaced, nondeterministic GPU ops, split not cached.
    How to solve: 33_MLOPS/real_problem.txt - manifest (commit, deps,
      dataset checksum, seed, hyperparams), deterministic split,
      versioned data.
    Where it lives: 33_MLOPS, 42_GIT_GITHUB, 43_TESTING.

--------------------------------------------------------------------------------
PART 3 - WHICH PART OF THE PLAYBOOK EACH PROJECT FAMILY EXERCISES
--------------------------------------------------------------------------------
Every project.txt in this course is a practice ground for a subset of
the taxonomy above. When you pick a project, first ask "which failure
classes could this produce?" and pre-build the measurement for them.

  Foundations (01-05):  data pipeline CLI, gradebook, scrapers,
    matrix calculator...        -> classes 4, 12 (data quality,
                                   reproducibility from day one)
  Classic ML (06-16):    spam/fraud/churn/house-price/segmentation,
    forecasting, ensembles...   -> classes 1, 2, 4, 5, 6, 7
  Anomaly (55_ANOMALY_DETECTION): screener, bake-off, streaming
    monitor                    -> classes 5, 10 + label-free eval
  Deep learning (17-22):  MNIST/CIFAR/CNN/LSTM projects... -> 6, 8, 9
  NLP/GenAI (23-32):      spam, sentiment, BERT fine-tune, RAG,
    vector search, LoRA...      -> 3, 4, 11, 9
  Engineering (33-46):    MLflow, DVC, Docker, APIs, CI/CD, cloud,
    production platforms       -> 2, 9, 10, 12
  Capstone (48) + Research (51): end-to-end system, paper reproduction
                               -> ALL classes, end to end

The rule: for LEVEL-2 projects onward, the deliverable is not "it
runs" - it is "it runs AND I can measure why it would fail before it
does." Add the STEP-3 measurement harness to every project you ship.

--------------------------------------------------------------------------------
PART 4 - THE 5-MINUTE DEBUGGING WORKFLOW (code skeleton)
--------------------------------------------------------------------------------
When a project misbehaves, copy this shape. It forces bisection and
prevents guess-and-check:

  def debug_pipeline(problem: str, stages: list):
      """stages = [(name, metric_fn, passes: bool -> bool), ...]
      Run the middle metric; if it passes, recurse downstream,
      else recurse upstream. Classic binary search on your own system.
      """
      def bisect(lo, hi):
          if hi - lo <= 1:
              print(f"ROOT CAUSE in stage: {stages[lo][0]}")
              return stages[lo][0]
          mid = (lo + hi) // 2
          name, metric, ok = stages[mid]
          verdict = metric()
          print(f"stage[{mid}] {name}: {'PASS' if verdict else 'FAIL'}")
          return bisect(mid, hi) if verdict else bisect(lo, mid)
      print(f"DEBUGGING: {problem}")
      return bisect(0, len(stages))

  # example: RAG wrong answer -> stages between question and answer
  # [("chunking", chunk_has_answer, ...),
  #  ("embedding", similar_is_close, ...),
  #  ("retrieval", gold_in_topk, ...),
  #  ("prompt", prompt_forces_context, ...),
  #  ("generation", answer_grounded, ...)]
  # root = debug_pipeline("wrong answer", stages)

--------------------------------------------------------------------------------
PART 5 - THE REAL PROBLEM INDEX (which file has which scenario)
--------------------------------------------------------------------------------
  real_problem.txt exists in these modules (worked diagnoses with
  WHY / HOW TO SOLVE / CODE):
    09_CLASSIFICATION   spam/fraud accuracy traps, imbalance economics
    10_REGRESSION       leakage, multicollinearity, RMSE blindness
    56_TIME_SERIES      degenerate forecasts, backtest traps
    19_DEEP_LEARNING    overfit, LR destruction, vanishing gradients
    20_CNN              99.9% web-bias trap, LR explosion, NaN, batch
    22_LSTM_GRU         identity-shortcut forecasts, stuck loss, BPTT
    23_NLP              distribution shift, domain NER, summarization
    26_TRANSFORMERS     worse-than-TFIDF fine-tunes, leakage, latency
    29_RAG              WRONG ANSWERS end-to-end: diagnostic tree,
                        debug script, fixed agent code (read this one
                        first if you build RAG)
    33_MLOPS            reproducibility, wrong monitoring metrics,
                        retrain regressions, env drift
    34_MODEL_DEPLOYMENT notebook-vs-live gap, latency, silent failure,
                        rollback

  Rule of the course: whenever a project fails, name the failure class
  from PART 2, open the matching real_problem.txt, and follow its
  diagnostic tree BEFORE changing code.
