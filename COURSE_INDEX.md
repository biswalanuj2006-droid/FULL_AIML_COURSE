# AI/ML Engineering Course - Complete Index

_Regenerated from the real file tree so every entry exists on disk.
Numbering resolved 2026-09-03: no directory shares a number prefix
(04_DSA->53_DSA, 05_ADVANCED_PYTHON->54_ADVANCED_PYTHON,
11_ANOMALY_DETECTION->55_ANOMALY_DETECTION, 16_TIME_SERIES->56_TIME_SERIES,
32_KAFKA->57_KAFKA, 33_SPARK->58_SPARK)._

Legend:  `think` = think.txt (hard reasoning + OPEN PROBLEM items)
         `flab` = failure_lab.txt (reproduce -> diagnose -> fix)
         `rp`   = real_problem.txt (real-world diagnosis playbooks)

---

## QUESTIONS_BANK.txt - AI ENGINEERING ULTRA-MASSIVE QUESTION BANK
**Root-level companion:** the cross-domain question bank from the master
prompt. 46 domains (Python -> NumPy/Pandas -> math/statistics -> data
science -> ML -> DL -> PyTorch -> CNN/NLP -> transformers -> LLMs -> LLM
engineering -> fine-tuning/PEFT -> quantization/inference -> embeddings/IR
-> RAG -> vector DBs -> RAG eval -> LangChain -> LangGraph -> agents ->
multi-agent -> multimodal -> generative -> recommenders -> time series ->
MLOps -> production -> cloud -> security -> eval -> observability ->
research -> system design) + Production Incidents (12), Master Interview
(100 done of 500 target, rest sourced from ml_course/INTERVIEW_BANK.md's
500 Q&A), Rapid Fire (100 of 500 target), Whiteboard (10), Master
Capstone (10), Final Exam structure, +47 Data Leakage, +48 Failure
Analysis. Verified: 46/46 domains, 1,711 unique question IDs, 0
duplicates, 4,485 lines. Extend any domain by appending new IDs.

## 00. ORIENTATION (`00_ORIENTATION`)
**Purpose:** Course overview, setup, and how-to-use
- README.md, course_overview.txt, setup_guide.txt, how_to_use.txt
- diagnostic_assessment.txt, diagnostic_answers.txt, practice.txt

## 01. PYTHON FOR AI/ML (`01_PYTHON`)
**Purpose:** Python fundamentals for AI/ML work
- 01_variables_types.txt, 02_control_flow.txt, 03_functions.txt
- 04_classes_oop.txt, 05_file_handling.txt, 06_error_handling.txt
- 07_modules_packages.txt, 08_virtual_environments.txt, 09_standard_library.txt
- README.md, practice.txt, project.txt, code/, exercises/

## 02. NUMPY (`02_NUMPY`)
**Purpose:** Numerical computing foundation
- 01_array_basics.txt, 02_indexing_slicing.txt, 03_broadcasting.txt
- 04_linear_algebra.txt, 05_random.txt, 06_statistics.txt, 07_advanced_numpy.txt
- README.md, practice.txt, project.txt, code/

## 03. PANDAS (`03_PANDAS`)
**Purpose:** Data manipulation and analysis
- 01_series_dataframe.txt, 02_data_reading.txt, 03_data_cleaning.txt
- 04_filtering_sorting.txt, 05_groupby_aggregation.txt, 06_merge_concat.txt
- 07_advanced_operations.txt
- README.md, practice.txt, project.txt

## 04. DATA VISUALIZATION (`04_VISUALIZATION`)
**Purpose:** Matplotlib, Seaborn, Plotly + ML visualizations
- 01_matplotlib_basics.txt, 02_matplotlib_advanced.txt, 03_seaborn.txt
- 04_plotly.txt, 05_ml_visualizations.txt
- README.md, practice.txt, project.txt

## 05. MATHEMATICS FOR ML (`05_MATHEMATICS`)
**Purpose:** Algebra, linear algebra, calculus, probability, statistics, optimization
- Folders: 01_ALGEBRA, 02_LINEAR_ALGEBRA, 03_CALCULUS, 04_PROBABILITY,
  05_STATISTICS, 06_OPTIMIZATION
- formulas/ (master formula sheet), visualizations/ (generated PNGs),
  math_cards/ (9 files: 8 card groups + index covering linear models,
  trees/ensembles, distance/probability, unsupervised, NN optimizers,
  sequences/attention, anomaly/time-series, graph/RL)
- README.md, practice.txt, project.txt, code/, exercises/

## 06. ML FUNDAMENTALS (`06_ML_FUNDAMENTALS`)
**Purpose:** Core ML concepts + scikit-learn mastery + learning paradigms
- sklearn_deep_dive.txt, advanced_learning_paradigms.txt
- README.md, practice.txt, project.txt, think

## 07. DATA PREPROCESSING (`07_DATA_PREPROCESSING`)
**Purpose:** Clean data correctly; avoid leakage
- 01_missing_values.txt, 02_duplicates.txt, 03_outliers.txt
- 04_encoding.txt, 05_scaling.txt, 06_train_test_split.txt
- 07_pipelines.txt, 08_data_leakage.txt
- README.md, practice.txt, project.txt, think

## 08. EDA (`08_EDA`)
**Purpose:** Exploratory data analysis reports
- 01_eda_overview.txt, 02_univariate_analysis.txt, 03_bivariate_analysis.txt
- README.md, practice.txt, project.txt, think

## 09. CLASSIFICATION (`09_CLASSIFICATION`)
**Purpose:** Logistic regression, KNN, naive Bayes, trees, SVM, imbalance
- classification_complete.txt, svm_deep_dive.txt, imbalanced_learning.txt
- README.md, practice.txt, project.txt, think, flab, rp

## 10. REGRESSION (`10_REGRESSION`)
**Purpose:** Linear through Bayesian/GP regression, diagnostics
- regression_complete.txt, bayesian_and_gp_regression.txt
- README.md, practice.txt, project.txt, think, flab, rp

## 11. CLUSTERING (`11_CLUSTERING`)
**Purpose:** K-Means, hierarchical, DBSCAN, GMM, spectral
- clustering_complete.txt
- README.md, practice.txt, project.txt, think

## 12. DIMENSIONALITY REDUCTION (`12_DIMENSIONALITY_REDUCTION`)
**Purpose:** PCA, t-SNE, UMAP, NMF, SVD connections
- dimensionality_reduction_complete.txt
- README.md, practice.txt, project.txt, think

## 13. ENSEMBLE LEARNING (`13_ENSEMBLE_LEARNING`)
**Purpose:** Bagging, RF, boosting, XGBoost/LightGBM/CatBoost, stacking
- ensemble_complete.txt, gradient_boosting_libraries.txt
- README.md, practice.txt, project.txt, think

## 14. MODEL EVALUATION (`14_MODEL_EVALUATION`)
**Purpose:** Metrics, CV, calibration, HPO, XAI
- model_evaluation_complete.txt, hyperparameter_optimization.txt,
  xai_interpretability.txt
- README.md, practice.txt, project.txt, think

## 15. FEATURE ENGINEERING (`15_FEATURE_ENGINEERING`)
**Purpose:** Feature creation, transforms, selection
- feature_engineering_complete.txt
- README.md, practice.txt, project.txt, think

## 16. ML FROM SCRATCH (`16_ML_FROM_SCRATCH`)
**Purpose:** Implement core algorithms with NumPy only
- from_scratch_ml.txt
- README.md, practice.txt, project.txt, think

## 17. CV FOUNDATIONS (`17_CV_FOUNDATIONS`)
**Purpose:** Image processing, OpenCV fundamentals
- cv_foundations_complete.txt
- README.md, practice.txt, project.txt

## 18. ANN (`18_ANN`)
**Purpose:** Neural network math, backprop, training dynamics
- ann_complete.txt
- README.md, practice.txt, project.txt, think, flab

## 19. DEEP LEARNING (`19_DEEP_LEARNING`)
**Purpose:** Modern DL practice, PyTorch deep dive, debugging
- pytorch_deep_dive.txt
- README.md, practice.txt, project.txt, think, flab, rp

## 20. CNN (`20_CNN`)
**Purpose:** Convolutional architectures, training, transfer learning
- cnn_complete.txt
- README.md, practice.txt, project.txt, think, flab, rp

## 21. RNN (`21_RNN`)
**Purpose:** Recurrent networks, sequence modeling, vanishing gradients
- rnn_complete.txt
- README.md, practice.txt, project.txt, think, flab

## 22. LSTM & GRU (`22_LSTM_GRU`)
**Purpose:** LSTM/GRU internals, sequence forecasting, gating math
- lstm_gru_complete.txt
- README.md, practice.txt, project.txt, think, flab, rp

## 23. NLP (`23_NLP`)
**Purpose:** Classical NLP pipeline through modern text models
- nlp_complete.txt
- README.md, practice.txt, project.txt, rp

## 24. WORD EMBEDDINGS (`24_WORD_EMBEDDINGS`)
**Purpose:** Word2Vec, GloVe, contextual embeddings
- embeddings_complete.txt
- README.md, practice.txt, project.txt

## 25. ATTENTION (`25_ATTENTION`)
**Purpose:** Attention mechanism, self-attention, multi-head math
- attention_complete.txt
- README.md, practice.txt, project.txt

## 26. TRANSFORMERS (`26_TRANSFORMERS`)
**Purpose:** Transformer architecture, Hugging Face, fine-tuning
- transformers_complete.txt, huggingface_deep_dive.txt
- README.md, practice.txt, project.txt, rp

## 27. GENAI FOUNDATIONS (`27_GENAI_FOUNDATIONS`)
**Purpose:** Generative AI concepts and applications
- genai_foundations.txt
- README.md, practice.txt, project.txt

## 28. LLM FUNDAMENTALS (`28_LLM_FUNDAMENTALS`)
**Purpose:** LLM training, prompting, evaluation
- llm_fundamentals.txt
- README.md, practice.txt, project.txt

## 29. RAG (`29_RAG`)
**Purpose:** Retrieval-augmented generation, retrieval quality
- rag_complete.txt
- README.md, practice.txt, project.txt, rp (full wrong-answer debug walkthrough)

## 30. VECTOR DATABASES (`30_VECTOR_DATABASES`)
**Purpose:** Embedding storage, ANN search, vector stores
- vector_databases_complete.txt
- README.md, practice.txt, project.txt

## 31. PROMPT ENGINEERING (`31_PROMPT_ENGINEERING`)
**Purpose:** Prompt patterns, evaluation, optimization
- prompt_engineering_complete.txt
- README.md, practice.txt, project.txt

## 32. FINE-TUNING (`32_FINE_TUNING`)
**Purpose:** Full and parameter-efficient fine-tuning (LoRA etc.)
- fine_tuning_complete.txt
- README.md, practice.txt, project.txt

## 33. MLOPS (`33_MLOPS`)
**Purpose:** Experiment tracking, pipelines, registries, monitoring
- mlops_libraries.txt
- README.md, practice.txt, project.txt, rp

## 34. MODEL DEPLOYMENT (`34_MODEL_DEPLOYMENT`)
**Purpose:** Serving models, APIs, scaling inference
- model_deployment_complete.txt
- README.md, practice.txt, project.txt, rp

## 35. FASTAPI (`35_FASTAPI`)
**Purpose:** Modern Python API framework for ML services
- fastapi_deep_dive.txt
- README.md, practice.txt, project.txt

## 36. FLASK (`36_FLASK`)
**Purpose:** Flask web framework and ML serving
- flask_complete.txt
- README.md, practice.txt, project.txt

## 37. DJANGO (`37_DJANGO`)
**Purpose:** Django full apps and REST APIs
- django_complete.txt
- README.md, practice.txt, project.txt

## 38. REST APIs (`38_REST_APIS`)
**Purpose:** API design, versioning, GraphQL, best practices
- rest_apis_complete.txt, graphql.txt
- README.md, practice.txt, project.txt

## 39. AUTH (`39_AUTH`)
**Purpose:** Authentication and authorization for apps/APIs
- auth_complete.txt
- README.md, practice.txt, project.txt

## 40. DATABASES (`40_DATABASES`)
**Purpose:** SQL, PostgreSQL, ORMs, database design
- databases_complete.txt
- README.md, practice.txt, project.txt

## 41. DOCKER (`41_DOCKER`)
**Purpose:** Containers, images, compose for ML apps
- docker_complete.txt
- README.md, practice.txt, project.txt

## 42. GIT & GITHUB (`42_GIT_GITHUB`)
**Purpose:** Version control, branching, CI/CD basics
- git_github_complete.txt
- README.md, practice.txt, project.txt

## 43. TESTING (`43_TESTING`)
**Purpose:** Unit, integration, API testing for ML systems
- testing_complete.txt
- README.md, practice.txt, project.txt

## 44. CLOUD (`44_CLOUD`)
**Purpose:** Cloud platforms, deployment, managed ML services
- cloud_complete.txt
- README.md, practice.txt, project.txt

## 45. SYSTEM DESIGN (`45_SYSTEM_DESIGN`)
**Purpose:** Designing scalable ML/backend systems
- system_design_complete.txt
- README.md, practice.txt, project.txt

## 46. PRODUCTION AI (`46_PRODUCTION_AI`)
**Purpose:** Monitoring, drift detection, ML security, production platforms
- production_ai_complete.txt, ml_security.txt
- README.md, practice.txt, project.txt

## 47. PROJECTS (`47_PROJECTS`)
**Purpose:** Capstone-style project work
- projects_overview.txt, README.md, LEVEL_1/, LEVEL_2/, LEVEL_3/

## 48. CAPSTONE (`48_CAPSTONE`)
**Purpose:** End-to-end final projects
- capstone_projects.txt, README.md

## 49. LEETCODE (`49_LEETCODE`)
**Purpose:** Pattern-based coding practice
- Folders: 01_two_pointer ... 16_linked_list (16 patterns)
- README.md

## 50. INTERVIEW PREP (`50_INTERVIEW_PREP`)
**Purpose:** ML interview question banks
- interview_prep_complete.txt
- README.md, practice.txt, project.txt

## 51. RESEARCH ENGINEERING (`51_RESEARCH_ENGINEERING`)
**Purpose:** Reading/reproducing papers, running experiments
- research_engineering_complete.txt, research_paper_track.txt
- README.md, practice.txt, project.txt

## 52. EXAM (`52_EXAM`)
**Purpose:** Full module examinations
- python_dsa_exam.txt, math_statistics_exam.txt, machine_learning_exam.txt
- deep_learning_nlp_exam.txt, systems_engineering_exam.txt, practical_exam.txt
- README.md

## 53. DATA STRUCTURES & ALGORITHMS (`53_DSA`)  _(renumbered from 04_DSA)_
**Purpose:** Interview-grade DSA foundations (complements 49_LEETCODE)
- Folders: 01_complexity, 02_arrays, 03_strings, 04_linked_lists,
  05_stacks, 06_queues, 07_hashing, 08_recursion, 09_binary_search,
  10_sorting, 11_trees, 12_heaps, 13_graphs, 14_dp, 15_greedy,
  16_tries, 17_advanced_ds
- README.md, project.txt

## 54. ADVANCED PYTHON (`54_ADVANCED_PYTHON`)  _(renumbered from 05_ADVANCED_PYTHON)_
**Purpose:** OOP depth, decorators/generators, async, internals
- Folders: 01_oop_deep, 02_decorators_generators, 03_async_concurrency,
  04_internals, 05_design_patterns, 06_profiling
- README.md, practice.txt, project.txt

## 55. ANOMALY DETECTION (`55_ANOMALY_DETECTION`)  _(renumbered from 11_ANOMALY_DETECTION)_
**Purpose:** Isolation Forest, LOF, one-class SVM, autoencoders, TS methods
- anomaly_detection_complete.txt
- README.md, practice.txt, project.txt, think

## 56. TIME SERIES (`56_TIME_SERIES`)  _(renumbered from 16_TIME_SERIES)_
**Purpose:** ARIMA/SARIMA, exponential smoothing, ML forecasting
- time_series_complete.txt
- README.md, practice.txt, project.txt, think, rp

## 57. KAFKA (`57_KAFKA`)  _(renumbered from 32_KAFKA)_
**Purpose:** Streaming, event pipelines, real-time ML features
- kafka_complete.txt
- README.md, practice.txt, project.txt

## 58. SPARK (`58_SPARK`)  _(renumbered from 33_SPARK)_
**Purpose:** Distributed data processing with PySpark
- spark_complete.txt
- README.md, practice.txt, project.txt

## 59. RECOMMENDER SYSTEMS (`59_RECOMMENDER_SYSTEMS`)
**Purpose:** Ranking the next item to show (CF -> two-stage production)
- recommender_systems_complete.txt
- README.md, practice.txt, project.txt, think

## 60. GRAPH MACHINE LEARNING (`60_GRAPH_MACHINE_LEARNING`)
**Purpose:** GNNs, node/link prediction, knowledge graphs, graph RAG
- graph_ml_complete.txt
- README.md, practice.txt, project.txt, think

## 61. REINFORCEMENT LEARNING (`61_REINFORCEMENT_LEARNING`)
**Purpose:** MDPs, Bellman, Q-learning -> PPO, RLHF connections
- rl_foundations_complete.txt
- README.md, practice.txt, project.txt, think

---

## Support Areas

**cheat_sheets/** - algorithm_selection_guide, docker, fastapi, python,
pytorch, sklearn cheat sheets

**code/** - runnable examples + practice per topic (each topic folder has
at least one runnable example AND a practice.txt)
- From-scratch algorithms (01-06 at root + ml/ subfolder with PCA, RF,
  gradient boosting, GD, softmax/CE + practice sets)
- Topic folders with example + practice: python, numpy, pandas,
  visualization, dl, cnn, rnn, transformers, nlp, rag, fastapi, flask,
  docker, mlops, recommenders, graph, rl, projects
- New-module examples verified by execution: MF-SGD recommender +
  ALS-vs-SGD comparison, numpy GCN + link prediction, Q-learning
  gridworld (each PASSes its baseline check)

**datasets/** - registry of recommended datasets with sources/licenses

**diagrams/** - generated diagrams (backend, dl, math, ml, nlp, rag,
  transformers, graphs, llm, agents) + diagram-generation scripts
  (generate_all_diagrams.py, generate_module_diagrams.py,
  generate_course_diagrams.py)
  - diagrams/llm/ (10): llm_architecture, pretraining_pipeline, kv_cache,
    kv_cache_memory, lora, sampling, scaling_laws (REAL sweep numbers
    2.723/2.617/2.421 vs bigram 2.763), prefill_decode, quantization,
    speculative_decoding
  - diagrams/agents/ (5): agent_loop, multi_agent, rag_agent,
    prod_rag_server, embedding_bench (REAL lab numbers: 9/9 vs 6/9
    hit@1, paraphrases 6/6 vs 4/6)
  - diagrams/ml/ (8 new): ml_lifecycle, learning_curves, kmeans, pca,
    pr_curve, imbalance, feature_importance, time_series
  - GALLERY: diagrams/DIAGRAM_GALLERY.md (markdown index of all 64
    images, renders on GitHub) + diagrams/gallery/index.html (browsable
    light/dark gallery with thumbnails + captions) + diagrams/gallery/
    REVIEW.txt (integrity review: 64/64 decode, 0 blank). Regenerate:
    python diagrams/generate_gallery.py + python diagrams/verify_diagrams.py

**exercises/** + **solutions/** - beginner / intermediate / advanced
  practice sets with verified solutions

**reference/** - third-party Python PDF notes (Django, Flask, REST APIs,
  PostgreSQL, data analysis, OpenCV...) extracted for supplementary reading

**genai_agents_course/** - visual reference: diagrams/agents/ (5 PNGs:
agent loop, multi-agent supervisor, RAG agent, production RAG server,
embedding bench with real lab numbers).

**genai_agents_course/** - self-contained ULTRA-DEEP GENERATIVE AI + LLM +
AI AGENTS sub-course (COURSE.txt Parts 0-50 + dependency graph + gates +
roadmaps + comparisons + final exam; PRACTICE.txt levels 0-8 with verified
math answers; PROJECT.txt 3 levels + production capstone; EXAMPLE.py
62-section runnable lab - verified 24/24 sections OK; agent_lab.py runnable
AI Agent lab - tool registry/schemas, safe AST calculator, doc retrieval,
memory, ReAct loop, reflection/replan, indirect prompt-injection guard,
evaluation - verified 7/7 tasks PASS; multi_agent_lab.py - supervisor +
writer + reviewer + researcher specialists, 5/5 PASS; rag_agent_server.py
- FastAPI RAG-agent server exercised through TestClient (real HTTP, no
sockets): auth 401, per-key token-bucket rate limit 429, grounded answers
with citations, calculator tool, streaming, upstream retry, response
cache, honest fallback, prompt-injection guard - verified 8/8 checks PASS;
rag_agent_server_prod.py - the PRODUCTION layers on top of that server:
SQL request log (SQLite file, PostgreSQL-compatible SQL) with per-user
GROUP BY usage reports, Redis-style TTL/LRU response cache with provable
expiry, multi-user roles (admin/free/trial) with rate limits and token
quotas (401/403/429/402 all exercised) - 10/10 checks PASS;
embedding_rag_lab.py - RAG with REAL dense embeddings trained in-file
(PPMI + truncated SVD), cosine retrieval beating lexical search on
paraphrases (emb 9/9 vs lex 6/9; paraphrases 6/6 vs 4/6) - all checks PASS;
embedding_hf_bench.py - benchmark of the local PPMI-SVD embeddings vs a
real transformer sentence-embedding model WHEN one is cached on disk
(offline-only; scans the HF hub cache by file, imports HF libraries only
if a model exists, else prints cache instructions and exits PASS - no
downloads ever)

**llm_course/** - visual reference: diagrams/llm/ (10 PNGs: LLM
architecture, KV cache + memory, LoRA, sampling, scaling curve with real
lab numbers, prefill/decode, quantization, speculative decoding).

**llm_course/** - self-contained ULTRA-DEEP LLM sub-course (COURSE.txt
Parts 0-66: math -> attention/transformers -> tokenization -> language
modeling -> pretraining/scaling -> LoRA/QLoRA/quantization -> KV cache &
inference -> serving/eval -> RAG/agents + dependency graph, 10 gates,
comparison set, final exam plan; PRACTICE.txt levels 1-8 with verified
answers; PROJECT.txt 3 levels (24 projects) + Production LLM capstone;
EXAMPLE.py 54-section runnable lab with torch-guarded training - verified
22/22 sections OK; mini_gpt_lab.py real end-to-end mini-GPT - trains a
~250k-param GPT on real lesson prose with AdamW, samples during training,
beats the bigram baseline (1500 steps: val ppl 12.8 vs bigram 15.9);
kv_cache_lab.py - proves cached decoding == full forward (max diff
9.5e-07), measures decode speedup (1.8x at 60 tokens), computes KV
memory incl. MHA/GQA/MQA for a 7B-class config (10.0/1.25/0.16 GB);
speculative_decoding_lab.py - draft/verify/reject loop: identical greedy
outputs, ~2.6x fewer forwards, gamma sweep up to 3x;
lora_finetune_lab.py - end-to-end LoRA fine-tune: pretrains a tiny GPT on
prose, adapts it to a math-lecture style with rank-8 LoRA vs full
fine-tune from the same checkpoint (LoRA: 18,432 of 141,120 trainable,
val-B 3.54->2.50 with domain A protected; full FT memorizes the 4.2k-char
corpus and forgets A by +0.71 nats - the implicit-regularizer result),
NOW WITH A QLORA EXTENSION: the frozen base is stored int8/int4 (4x/6x
memory cut, mean-abs quant error 7.2e-4/4.0e-3) and the same adapters
recover the domain within 0.01-0.02 nats of the fp32-base LoRA
(int8 2.508 / int4 2.501 vs 2.498) - 9/9 checks PASS;
scale_sweep_lab.py - model-size sweep: S/M/L (119k/253k/435k params) x
450 steps on the same 200k-char corpus and seed; ALL sizes beat the
bigram baseline and bigger is strictly better (val loss 2.723 / 2.617 /
2.421 vs bigram 2.763; ppl 15.2/13.7/11.3) - the empirical scaling curve
of Part 27, ALL CHECKS PASS monitoring_lab.py - observability for Part 57:
spans with latency/tokens/cost/errors, p50/p95, caching that measurably
cuts latency + cost, retries (2 injected failures, 0 lost), JS-divergence
drift detection over query topics + alerts + dashboard - 5/5 PASS)

**ml_course/** - visual reference: diagrams/ml/ (existing jpgs + 8 new
PNGs: ml_lifecycle, learning_curves, kmeans, pca, pr_curve, imbalance,
feature_importance, time_series).

**ml_course/** - self-contained ULTRA-DEEP MACHINE LEARNING sub-course
for a 2nd-year B.Tech AI&ML student (university + industry + research
level). COURSE.txt Modules 0-46: roadmap -> Python/NumPy/Pandas/Matplotlib
-> linear algebra/probability/statistics/info theory -> data quality/EDA/
preprocessing -> leakage-safe splits + baselines -> linear & logistic
regression/regularization -> metrics -> KNN/Naive Bayes/decision trees ->
ensembles (RF/XGB/LightGBM/CatBoost) -> SVM -> clustering/PCA -> feature
engineering -> imbalance/anomaly/time series -> tuning/calibration -> SHAP/
interpretability -> probabilistic/causal ML -> security/fairness/experiment
design -> ML systems/MLOps/monitoring/cloud/research skills, with 16-point
deep-dive per topic, comparison tables, final exam + capstone; EXAMPLE.py
26-section runnable lab - scratch OLS/GD/logreg/KNN/Naive Bayes/CART/RF/
K-Means/PCA/metrics/CV/GBDT + pipelines/joblib serving - verified 26/26
sections PASS (0 fail); PRACTICE.txt levels 0-5 with verified math answers
+ FULL EXAM SET (78 items, all 9 sections, with answer key); PROJECT.txt
tiers E-S+ (58 projects) + Production ML Platform capstone;
MASTER_PROMPT_AUDIT.md maps every requirement of the ultra-deep ML master
prompt to its home with status + remaining drill work; INTERVIEW_BANK.md
- 500 compact Q&A rows scaled to the master-prompt targets (Beginner 100 /
Intermediate 150 / Advanced 150 / Expert 100))

**Root docs:** README.md, ROADMAP.md, COURSE_AUDIT.txt (full audit),
ML_MASTER_AUDIT.txt (ML-module audit + scores), QUALITY_STANDARDS.md,
library_map.md, reference_reading.md (PDF mapping), ultimate_project.md
(master problem-solving playbook), report.txt
