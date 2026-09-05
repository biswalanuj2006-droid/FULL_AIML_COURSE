# ml_course - INTERVIEW BANK (scaled to master-prompt targets)

Compact Q&A drill bank. Targets: Beginner 100+ / Intermediate 150+ /
Advanced 150+ / Expert 100+. Answers are deliberately terse - expand each
into a 30-second structured answer when drilling. Cross-references: COURSE.txt
modules, PRACTICE.txt VERIFIED ANSWERS, EXAMPLE.py sections.

================================================================================
BEGINNER (B001-B100) - Python, data, EDA, statistics, linear models, metrics
================================================================================
B001 | What is the difference between a list, a tuple and a set? | list: ordered mutable; tuple: ordered immutable; set: unordered unique hashable
B002 | How does a dict work under the hood? | hash table: key -> hash -> bucket; O(1) average lookups
B003 | Explain list comprehension and when you would NOT use it | compact builder [f(x) for x in xs]; avoid when the logic spans branches -> use a loop
B004 | What is broadcasting in NumPy? Give one example | aligning shapes by expanding trailing dims; (3,1)+(1,4) -> (3,4)
B005 | Why vectorize instead of looping in NumPy? | loops run Python per element; vector ops run C loops, 10-100x faster
B006 | Difference between np.dot, np.matmul and * | dot/matmul = matrix product; * = elementwise
B007 | What is a view vs a copy in NumPy/Pandas? | view shares memory (slicing); copy owns data; mutation surprises
B008 | How do you handle NaN in a DataFrame? | isnull() -> dropna / fillna(mean/median/ffill) after checking the pattern
B009 | What does groupby do and what must you aggregate? | split-apply-combine by key; every other column needs agg (mean/sum/count)
B010 | Merge vs concat vs join? | merge on keys (SQL join); concat stacks rows/cols; join = merge on index
B011 | What is EDA and what do you check first? | inspect shape, dtypes, missing, duplicates, target distribution, ranges
B012 | Why log-transform a skewed feature? | compresses long right tail; makes relationships more linear and robust
B013 | Mean vs median: when is median the right summary? | skewed or outlier-heavy distributions
B014 | What does a boxplot show? | min/Q1/median/Q3/max + outliers beyond 1.5*IQR
B015 | Correlation 0.8: what can you conclude? | linear association; not causation; may be spurious/confounded
B016 | Why inspect class balance first? | decides the metric (accuracy can mislead) and the split strategy
B017 | What is a histogram vs a KDE? | histogram bins counts; KDE = smooth density estimate
B018 | What is data leakage in one sentence? | information from outside the training fold (future/test) influences the model
B019 | Train/validation/test: three roles? | train fits; val selects (hyperparams/models); test estimates final performance once
B020 | Why must test be touched last? | repeated peeking overfits the test estimate
B021 | P(A|B) meaning in words | probability of A given B happened
B022 | Bayes theorem: write it | P(A|B)=P(B|A)P(A)/P(B)
B023 | Independence vs mutually exclusive? | independent: P(AB)=P(A)P(B); exclusive: cannot both happen
B024 | Expected value of a die | 3.5
B025 | Variance formula intuition | average squared deviation from the mean
B026 | CLT: what converges and how fast? | sample mean -> Normal; std shrinks as sigma/sqrt(n)
B027 | p-value definition (plain words) | probability of data this extreme IF the null were true
B028 | Type I vs Type II error with a medical example | false positive: healthy labeled sick; false negative: sick labeled healthy
B029 | Confidence interval meaning | across repeated samples, ~95% of such intervals contain the true value
B030 | Why does a bigger sample give tighter CIs? | SE = sigma/sqrt(n)
B031 | OLS assumption violated by outliers? | least squares is not robust: squared errors amplify outliers
B032 | Interpret slope 2.5 in y = 1 + 2.5x | x up 1 unit -> y up 2.5 on average
B033 | R2 = 0.7 meaning | 70% of variance in y explained by the model
B034 | RMSE vs MAE choice? | RMSE punishes big errors (normal noise); MAE robust (heavy tails)
B035 | Why add an intercept? | otherwise the line is forced through the origin and residuals bias
B036 | Closed-form vs gradient descent for OLS? | normal equation exact for small p; GD scales to huge p
B037 | Multicollinearity symptom? | unstable, inflated coefficients; X^T X near singular
B038 | What is homoscedasticity? | constant residual variance across fitted values
B039 | When is linear regression the wrong tool? | nonlinear/heterogeneous effects, bounded targets, outliers
B040 | What do residuals tell you after a fit? | pattern => missed structure; fan shape => variance misspecified
B041 | Why regularize? | constrain capacity to lower variance; useful when p ~ n or correlated features
B042 | L1 vs L2 effect on coefficients | L1 shrinks some to exactly 0 (selection); L2 shrinks proportionally
B043 | Ridge closed form | (X^T X + lambda I)^-1 X^T y
B044 | What does lambda do in Ridge? | larger -> coefficients closer to 0, more bias, less variance
B045 | When would Lasso fail? | correlated groups: picks one arbitrarily; prefer elastic net
B046 | Why scale before L1/L2 penalties? | penalty is scale-dependent; unscaled features get penalized unevenly
B047 | Bias-variance tradeoff in one line | too-simple = high bias; too-flexible = high variance; total error = bias^2 + variance + noise
B048 | Train error low, val high: diagnosis | overfitting (variance) - reduce capacity/regularize/more data
B049 | Both train and val error high | underfitting (bias) - increase capacity/features
B050 | How does more data help overfitting? | averages out noise; variance of the fit falls
B051 | Sigmoid output range and why logistic uses it | (0,1) so it can model a probability
B052 | Decision boundary of logistic regression | linear in feature space: w^T x + b = 0
B053 | Log loss for perfect prediction | 0
B054 | Why log loss instead of squared error for classification? | CE gradient is well-scaled; squared error saturates on sigmoid
B055 | Confusion matrix: define TP/FP/FN/TN | predicted-vs-actual counts; FP = predicted 1 actual 0 etc
B056 | Precision in words | of predicted positives, fraction actually positive
B057 | Recall in words | of actual positives, fraction found
B058 | F1 when precision=recall=0.8 | harmonic mean = 0.8
B059 | Accuracy on 99:1 data - why misleading? | predicting all-majority gives 99% accuracy with 0 recall
B060 | When to prefer PR over ROC | imbalanced classes: PR highlights the rare class
B061 | AUC 0.5 vs 0.9 meaning | random ranking vs strong ranking ability
B062 | What is a ROC curve? | TPR vs FPR across thresholds
B063 | Log loss vs accuracy | log loss penalizes confidence errors; accuracy only the final label
B064 | Multiclass: one-vs-rest vs multinomial? | OvR trains K binary models; multinomial fits one softmax model
B065 | What metric for fraud where FP is expensive? | precision (or F_beta with beta < 1); threshold on expected cost
B066 | KNN: prediction rule | majority vote (classification) / mean (regression) of k nearest
B067 | Why scale features for KNN? | distance is dominated by large-magnitude features otherwise
B068 | Choosing k: small vs large? | small k = low bias/high variance; large k smoother
B069 | Curse of dimensionality effect on KNN | distances converge; neighborhoods empty; needs exponential data
B070 | Naive Bayes assumption | features independent given the class (usually false, often still useful)
B071 | Why is Naive Bayes fast and sample-efficient? | each feature estimated separately; few parameters
B072 | Gaussian vs Multinomial vs Bernoulli NB use | Gaussian numeric; Multinomial counts (text); Bernoulli binary presence
B073 | When does NB beat logistic? | very small data, high dimensions (text); otherwise logistic usually better
B074 | Distance metrics: Euclidean vs Manhattan | L2 vs L1; Manhattan robust in high dims/sparse
B075 | Why standardize before PCA | PCA is variance-based; unscaled features dominate
B076 | Dummy baseline: why build one first? | sets the bar; any model must beat majority/mean prediction
B077 | What does the mean regressor predict? | the training mean for every row
B078 | How do you know a model actually learned? | beats the baseline on a holdout, not just the training set
B079 | Leakage via scaling - fix | fit scaler on train folds only; transform val/test with it
B080 | Why shuffle before splitting? | avoids ordered/group structure leaking into folds
B081 | What is a duplicate row and why care? | same sample twice inflates its weight and leaks between splits
B082 | Describe a time-series-safe split | chronological: train on past, validate/test on future only
B083 | Feature that uses the target (aggregation) is called | target leakage
B084 | What is an outlier and two detection heuristics | extreme value; z-score > 3 or outside 1.5*IQR
B085 | Missing data: mean fill when NOT ok | MNAR or high missing rate; distorts distribution; consider indicator
B086 | What is one-hot encoding and its cost | K binary columns; cost: dimensionality (drop one or use categorical)
B087 | Ordinal vs nominal encoding | ordinal keeps order (label); nominal should not use arbitrary integers
B088 | Why drop the first one-hot column | avoids perfect multicollinearity (dummy trap)
B089 | What is a pipeline and why? | bundles preprocessing + model; prevents leakage and keeps transforms consistent
B090 | How do you report a model's expected performance? | cross-validated score + CI, never the training score
B091 | 5-fold CV: how many models trained? | 5 (each fold held out once)
B092 | What does stratified CV protect? | class proportions in every fold (rare classes)
B093 | Grid search pitfall | tuning on the test set leaks; use nested CV or a final holdout
B094 | What is a confusion-matrix-derived accuracy formula | (TP+TN)/(TP+TN+FP+FN)
B095 | Positive class imbalance handling order (3 steps) | metric -> class weights/resample -> threshold
B096 | Explain to a non-technical person what a model does | maps inputs to predictions learned from labeled examples
B097 | When would you not use ML at all? | a simple rule/query is accurate, cheap, auditable; tiny data
B098 | How do you debug a model that predicts one class only? | check imbalance, threshold, feature scale, class weights, loss
B099 | What does a learning curve show? | error vs training size; gap = variance, level = bias
B100 | Your project: 30 seconds | problem -> data -> baseline -> model -> metric -> error analysis -> deployment lesson

================================================================================
INTERMEDIATE (I001-I150) - trees, ensembles, SVM, clustering, PCA,
features, imbalance, tuning, calibration, interpretability
================================================================================
I001 | Decision tree: what does a split optimize? | impurity drop: Gini/entropy (class) or variance (regression)
I002 | Gini of a pure node | 0
I003 | Information gain formula sketch | H(parent) - weighted avg H(children)
I004 | Why are trees prone to overfitting? | they can partition until every leaf is pure; depth/leaf constraints needed
I005 | Pruning: pre vs post | pre: depth/min_samples limits; post: grow fully then collapse weakest branches
I006 | How does a tree choose between splits of similar gain? | greedily, largest immediate gain; no lookahead
I007 | Feature importance from a tree | total impurity reduction attributed to each feature
I008 | Regression tree leaf prediction | mean of the samples in the leaf
I009 | Why binary splits (CART) instead of multiway? | binary + depth is flexible; multiway fragments data fast
I010 | Tree stability problem | small data changes flip splits -> high variance (why ensembles)
I011 | Bagging in one line | train many models on bootstrap samples and average votes
I012 | Why does averaging reduce variance? | errors are ~independent; average variance ~ sigma^2/B
I013 | Random forest: two randomness sources | bootstrap rows + random feature subset per split
I014 | Why random features help? | decorrelates trees so the average has less variance
I015 | RF out-of-bag score | evaluate each tree on samples it never saw; free validation
I016 | RF bias-variance position | low-ish variance, but not lower bias than a single tree
I017 | When is RF weak? | extrapolation, high-cardinality categoricals, sparse linear structure
I018 | Extra Trees difference | random thresholds too -> even lower variance, slightly more bias
I019 | Boosting idea | sequentially fit weak learners to the errors of the ensemble
I020 | AdaBoost weight update | misclassified samples get higher weight; alpha ~ log((1-err)/err)
I021 | GBDT fits what exactly | a tree to the negative gradient (pseudo-residual) of the loss
I022 | Why learning rate in boosting? | each tree is a small step; smaller lr -> smoother, needs more trees
I023 | GBDT with 10k trees and small lr = ? | low bias low variance if val-selected; risk is cost and overfit without early stop
I024 | Early stopping in boosting | monitor val loss and keep the best iteration count
I025 | XGBoost vs plain GBDT (3 improvements) | regularization (gamma/lambda), second-order (Hessian), weighted quantile sketching + pruning
I026 | What is gamma in XGBoost? | minimum loss reduction required to make a further partition
I027 | XGBoost handles missing values how | learns default direction per split
I028 | Why is XGBoost fast? | pre-sorted/quantile histograms, column blocks, cache-aware, parallel
I029 | XGBoost overfit symptoms | very high train vs val; shrink lr, add gamma/lambda, fewer depth
I030 | What is a DMatrix? | XGBoost's internal pre-processed data format (sparse-aware, cached)
I031 | LightGBM vs XGBoost core difference | leaf-wise growth with max-depth limit vs level-wise
I032 | Why can leaf-wise growth overfit? | it grows the best leaf -> deeper unbalanced trees; constrain num_leaves
I033 | Histogram binning does what? | buckets continuous features -> faster split search, some precision loss
I034 | GOSS idea | keep high-gradient samples, sample low-gradient ones -> speed at small accuracy cost
I035 | EFB (Exclusive Feature Bundling) | bundle mutually exclusive sparse features to cut dimension
I036 | LightGBM categorical handling | native: sort by target statistic per category, no one-hot
I037 | LightGBM pitfalls | small data overfit, num_leaves too large, leakage with high-cardinality category stats
I038 | CatBoost ordering principle | ordered boosting + ordered target statistics avoid target leakage in categoricals
I039 | Why does CatBoost handle categoricals better? | target encoding done online per sample with priors -> no leak
I040 | CatBoost vs LightGBM when? | CatBoost for heavy categoricals; LightGBM often faster on big numeric data
I041 | When is boosting better than RF? | structured/tabular with nonlinear interactions; RF better for wide noise robustness
I042 | Hyperparameters that most control GBDT overfit (3) | depth/num_leaves, lr, min_child/leaf samples
I043 | How do you compare two ensemble configs fairly? | same CV folds + seeds, nested selection, paired test
I044 | Why bagging helps high-variance low-bias models specifically | averaging is a variance reducer
I045 | Interaction capture: RF/XGB vs linear | trees find interactions automatically; linear needs engineered features
I046 | SVM objective | max margin hyperplane; soft margin trades slack with C
I047 | Support vectors definition | training points on/near the margin that define the boundary
I048 | What does C control? | penalty for margin violations; big C -> hard margin-ish, overfit
I049 | Kernel trick meaning | compute dot products in feature space implicitly via k(x,z)
I050 | RBF kernel intuition | similarity decays with distance; infinite-dim feature map
I051 | gamma in RBF | influence radius of each point; too high -> overfit (each point its own island)
I052 | SVM with 100k rows - concern | training ~ O(n^2..n^3); use LinearSVC or kernels on smaller sets
I053 | Why scale for SVM? | margin is geometry-based; unscaled axes dominate
I054 | When is linear SVM like logistic? | both linear discriminants; SVM hinge + margin vs logistic CE
I055 | SVM probability estimates | not native (hinge); calibrate with Platt if probabilities needed
I056 | k-means objective | minimize within-cluster sum of squared distances to centroids
I057 | Lloyd's algorithm steps | init k centers -> assign nearest -> recompute mean -> repeat
I058 | k-means sensitivity | initial centers; run k-means++ or multiple restarts
I059 | Elbow method caveat | subjective; pair with silhouette and domain sense
I060 | Silhouette near +1 vs near 0 | +1 compact well-separated; ~0 overlapping clusters
I061 | k-means failure modes | non-convex shapes, varying density/size, outliers
I062 | Hierarchical agglomerative linkage types | single/complete/average/Ward - single chains, Ward minimizes variance
I063 | DBSCAN parameters | eps (radius) + min_samples; density-reachable clusters + noise
I064 | DBSCAN advantage over k-means | arbitrary shapes + explicit noise, no k needed
I065 | When does DBSCAN fail? | varying density (one eps cannot fit all), high dims
I066 | PCA: what are principal components | directions of max variance, orthogonal, ordered
I067 | PCA steps | center -> covariance -> eigendecomposition -> top-k projection
I068 | Eigenvalue meaning in PCA | variance explained along that component
I069 | How many components to keep | cumulative explained variance >= 0.9-0.95 or elbow
I070 | PCA is unsupervised - consequence | may discard class-discriminative directions; use supervised alternatives when needed
I071 | PCA vs t-SNE vs UMAP purpose | PCA linear dim reduction; t-SNE/UMAP visualization (nonlinear, distances unreliable)
I072 | Why not trust t-SNE distances | it optimizes local structure; global distances/clusters sizes are distorted
I073 | Truncated vs standard SVD | sparse data: truncated SVD (no centering); PCA densifies
I074 | LDA vs PCA | LDA supervised: maximize class separation
I075 | When is dimensionality reduction harmful? | throwing away signal; check downstream metric, not variance alone
I076 | Feature engineering rule | features must be computable at prediction time without the target/future
I077 | Date/time features worth creating | hour, weekday, month, holiday, days-since, rolling stats
I078 | Interaction feature example | income x age (nonlinear) or product of two weak signals
I079 | Binning numeric features - tradeoff | robust to outliers/linearity but loses granularity; use sparingly
I080 | Why ratios can be better than raw counts | normalize by exposure/scale (e.g., spend per order)
I081 | Target encoding danger | leaks target; must use out-of-fold/ordered encoding
I082 | Frequency encoding | category -> its count; compact but collides rare and common meanings
I083 | Polynomial features cost | p features -> O(p^2) quadratics; explode + multicollinearity; regularize
I084 | Feature selection: filter vs wrapper vs embedded | filter: univariate score; wrapper: model-based search; embedded: L1/tree importance
I085 | Mutual information vs correlation | MI catches any dependence, not just linear
I086 | Why remove correlated redundant features? | stability + speed + interpretability (not always accuracy)
I087 | Recursive Feature Elimination | train, drop weakest by importance, repeat on shrinking set
I088 | Why feature selection must be inside CV | selection on full data leaks test info
I089 | Domain feature example: fraud | velocity (count in window), device consistency, amount vs history
I090 | Check a new feature's value | does CV improve? is it causal-ish? does it leak? cost at serving?
I091 | Imbalance: why accuracy lies | majority class dominates; 99% acc can be 0% recall on minority
I092 | First imbalance move (not resampling) | change the metric + baseline
I093 | SMOTE idea | interpolate between minority neighbors to synthesize samples
I094 | SMOTE pitfalls | interpolating noise/duplicates; creates synthetic points that may cross regions
I095 | Borderline-SMOTE vs SMOTE | focuses synthesis near the decision border
I096 | Class weights vs resampling | weights adjust loss without new samples; simpler, less overfit risk
I097 | Threshold tuning after probabilities | choose threshold maximizing PR/F-beta or minimizing expected cost
I098 | Why calibrate after resampling | resampling skews probability scale
I099 | Imbalance + CV split requirement | stratify on the minority class
I100 | Extreme imbalance (0.1%): realistic expectation | need PR curve + huge data or cost-aware design; don't expect magic
I101 | GridSearchCV vs RandomizedSearchCV | grid exhausts a lattice; random samples the space - better for many params
I102 | Random search with 3 params vs grid: why better? | each param gets more distinct values for the same budget
I103 | Optuna idea | TPE-based Bayesian optimization with pruning
I104 | What is nested CV for? | unbiased estimate of the tuned pipeline's performance
I105 | Inner vs outer CV in nested | inner selects; outer evaluates selection process
I106 | Hyperparameter overfitting symptom | best grid score >> fresh holdout score
I107 | Which params to tune first (trees)? | depth/leaves, then lr/min_child, then regularization
I108 | Early stopping vs tuned epochs | early stop on val is itself a selection - report final estimate carefully
I109 | Reproducible tuning checklist | fixed seeds, data version hash, param logs, same folds
I110 | Why log-scale search for lr/regularization? | these act multiplicatively; linear grid wastes budget
I111 | Model probability != true probability why? | miscalibrated scores (bias, regularization, class imbalance)
I112 | Reliability diagram | binned predicted vs observed frequency; diagonal = calibrated
I113 | Brier score | mean squared error between probability and outcome
I114 | Platt scaling | logistic fit on logits to recalibrate
I115 | Isotonic regression calibration | nonparametric monotone map; needs more data than Platt
I116 | When is calibration useless? | you only need ranks/thresholds, not trustworthy probabilities
I117 | Cost-sensitive threshold formula idea | predict positive when P(y=1) * benefit > P(y=0) * cost
I118 | Expected calibration error (ECE) | weighted avg |acc - conf| per bin
I119 | Does calibration change accuracy/ranking? | no (monotone), only the probability scale
I120 | Where to calibrate in the pipeline | on validation data only, after model selection
I121 | Feature importance from trees: caveat | favors high-cardinality/continuous features; not causal
I122 | Permutation importance idea | shuffle a feature, measure score drop; model-agnostic
I123 | Permutation importance pitfall | correlated features: shuffling one still leaves info via the other
I124 | Partial dependence plot shows | average prediction vs one feature, marginalizing others
I125 | ICE vs PDP | ICE per-instance curves; PDP is their average
I126 | SHAP value meaning | additive contribution of a feature to a prediction (Shapley)
I127 | Why SHAP is consistent | Shapley axioms: efficiency, symmetry, dummy, additivity
I128 | SHAP vs permutation importance | SHAP local + global from one model; permutation global only
I129 | LIME idea | locally fit an interpretable surrogate around one prediction
I130 | When explanations mislead | extrapolated regions, correlated features, unstable neighborhoods
I131 | Global vs local explanation | global: which features matter overall; local: why THIS prediction
I132 | Explainability for a rejected loan (regulator) | use SHAP on the decision + monotonic features where possible
I133 | Can a black box be auditable? | partial yes: fairness metrics, shadow tests, docs, explanations - not full reasoning
I134 | Feature attribution sum property | attributions add up to the prediction (minus baseline)
I135 | Why not use importance for feature selection alone | importance ~ predictive, not causal; correlated redundancy
I136 | Tuning vs regularization overlap | both control capacity; tune jointly, don't double-shrink blindly
I137 | What is the irreducible error | Bayes error from noise in y you cannot model away
I138 | Overfit detection in cross-val | val fold scores vary wildly / far below train
I139 | CV variance grows when | few rows, unstable models (trees), heterogeneous folds
I140 | Why GroupKFold | samples from the same group must stay in one side (user-level leakage)
I141 | Model comparison: what makes it valid | same data, same folds, same preprocessing, repeated seeds
I142 | Which wins more often on tabular: tuned RF vs tuned GBDT? | GBDT usually; RF wins on very noisy/wide data
I143 | Categorical with 1000 levels: options | frequency/ordinal, target (OOF), embeddings, CatBoost native
I144 | Outlier in train vs in test handling | train: robust scaler/winsorize; test: same transform, never refit
I145 | What does a validation curve show (param vs score) | sweet spot between underfit and overfit
I146 | When to prefer simple model over slightly better complex | interpretability, latency, monitoring, small data, stability
I147 | Ensemble of diverse models vs same-model bagging? | diversity captures different inductive biases; stack/blend with care
I148 | Why does feature scaling barely matter for trees | splits are threshold-based, invariant to monotone scaling
I149 | Missing-value strategy for trees | XGBoost/LGBM handle natively; RF needs imputation or surrogate splits
I150 | Describe how you would build a churn model end to end | define churn + horizon -> features (usage/velocity) -> time split -> baseline -> GBDT -> PR/threshold -> monitor drift

================================================================================
ADVANCED (A001-A150) - optimization, probabilistic/causal ML, shift,
security, fairness, experiments, anomaly, time series, ML systems,
MLOps, monitoring, cost
================================================================================
A001 | Why Adam over SGD? | per-parameter adaptive step from first/second moments; robust to scale
A002 | Adam bias correction | early moments are ~0; divide by 1-beta^t
A003 | AdamW difference | weight decay decoupled from the gradient step
A004 | When does SGD beat Adam? | generalization in some regimes (esp. vision); Adam can overfit fast
A005 | Learning rate too high symptom | loss diverges/oscillates
A006 | Warmup why? | early huge gradients destabilize; ramp lr first
A007 | Cosine decay idea | smooth lr decrease; helps final convergence
A008 | Gradient clipping purpose | cap gradient norm to prevent exploding steps
A009 | Batch size effect on gradient noise | bigger batch = lower noise, but less regularization per step
A010 | When use gradient accumulation | simulate large batch on small memory
A011 | MLE idea | choose params maximizing P(data|params)
A012 | MAP vs MLE | MAP adds a prior: argmax P(params|data) ~ prior x likelihood
A013 | Posterior predictive | integrate predictions over posterior, not just a point estimate
A014 | Conjugate prior example | Beta prior + Binomial likelihood -> Beta posterior
A015 | Bayesian vs frequentist CI | credible interval is a probability statement about the param; CI is about the procedure
A016 | Gaussian process in one line | distribution over functions defined by a mean + kernel covariance
A017 | Kernel choice in GP | RBF smooth; Matern rougher; reflects prior smoothness
A018 | Aleatoric vs epistemic uncertainty | aleatoric: irreducible noise; epistemic: lack of data (reducible)
A019 | When do you need uncertainty, not just a point prediction | medical, finance, active learning, safe RL
A020 | Calibrated probabilities are not uncertainty | they capture aleatoric confidence, not model ignorance
A021 | Correlation vs causation | correlation is a statistic; causation needs an intervention/identification
A022 | Confounder example | ice cream & drowning share summer/heat
A023 | Backdoor criterion idea | adjust for a set blocking all backdoor paths from treatment to outcome
A024 | ATE vs CATE | average effect vs conditional average per subgroup
A025 | RCT vs observational | RCT randomizes treatment (no confounding); observational needs adjustment
A026 | Propensity score use | balance treated/control on observed covariates
A027 | Why predictive ML differs from causal | predictive optimizes accuracy; causal requires valid identification, may drop predictive features
A028 | Double ML idea | orthogonalize treatment & outcome on covariates, then regress residuals
A029 | Uplift modeling goal | which customers respond MORE due to treatment (CATE for action)
A030 | Instrumental variable intuition | exogenous shifter of treatment that affects outcome only through treatment
A031 | Covariate shift | P(x) changes, P(y|x) same; reweight/retrain, watch feature drift
A032 | Label shift | P(y) changes, P(x|y) same; adjust priors
A033 | Concept drift | P(y|x) changes - model is wrong now; retrain/adapt
A034 | Train-prod mismatch sources | sampling bias, feature pipeline drift, label definition change
A035 | Detecting drift without labels | PSI/KS on features, prediction distribution
A036 | PSI intuition | measures how much a distribution shifted (sum of (actual-expected) * log ratio per bin)
A037 | PSI threshold rule of thumb | <0.1 stable; 0.1-0.25 moderate; >0.25 major shift
A038 | Adapting to drift options | retrain, online update, robust features, ensemble of windowed models
A039 | Why periodic retraining may still fail | drift within the window or silent label changes; monitor in between
A040 | Adversarial example definition | small perturbation that flips the prediction
A041 | Why do adversarial examples exist? | models learn non-robust high-frequency features; linearity in high dims
A042 | Data poisoning | attacker injects corrupted training samples to bias the model
A043 | Model extraction attack | query the API to train a copy; mitigate: rate limits, watermarking, differential privacy
A044 | Membership inference | determine if a row was in training from the model's confidence
A045 | Backdoor attack | trigger pattern that flips predictions at inference
A046 | Defenses overview | adversarial training, input sanitization, robust stats, monitoring queries
A047 | Why security matters for fraud models specifically | adversaries adapt to the model (feedback loop)
A048 | Differential privacy intuition | add calibrated noise so individual rows barely change outputs
A049 | Fairness: demographic parity | selection rates equal across groups
A050 | Equalized odds | equal TPR/FPR across groups
A051 | Equal opportunity | equal TPR across groups
A052 | Why can't you satisfy parity and calibration together? | base rates differ; constraints conflict (proven tradeoffs)
A053 | Bias source in training data | historical bias, sampling bias, measurement bias in labels
A054 | Fairness metric choice depends on | legal framing (impact vs opportunity), base rates, cost asymmetry
A055 | What is disparate impact testing | compare outcome rates across protected groups; use 80% rule or statistical tests
A056 | Why fairness audits need new data | model may be fair on training distribution but not deployment
A057 | Hypothesis in ML experiments | testable claim with a metric and a control
A058 | Ablation study | remove one component at a time and measure the drop
A059 | Why random seeds + repeated runs | single run conflates noise with signal
A060 | Statistical significance for model A vs B | paired test on per-fold/per-sample scores, report CI
A061 | Pre-registration value | prevents p-hacking/selecting the result after seeing it
A062 | Error analysis before new models | find WHERE it fails (classes/slices) - often cheaper fixes
A063 | Experiment report skeleton | question, data, method, results, ablations, limitations
A064 | Why baseline matters in any claim | without a bar you cannot attribute improvement to your idea
A065 | Multiple comparisons problem in tuning | many configs tried = some will look good by chance; correct or use nested CV
A066 | Isolation Forest idea | isolate points with few random cuts; anomalies isolate fast
A067 | Why IsolForest works without labels | anomaly = easy to isolate, rare, different
A068 | One-class SVM | finds a boundary around normal data; sensitive to kernel/gamma
A069 | LOF idea | compare local density vs neighbors; outliers have low relative density
A070 | Anomaly detection evaluation problem | no/rare labels; use hit-rate@k, precision at threshold, domain validation
A071 | Streaming anomaly vs batch | streaming needs incremental stats + alert thresholds
A072 | Time series: stationarity | mean/var constant over time; required by ARIMA-class models
A073 | Trend vs seasonality vs cycle | long-term direction; fixed-period pattern; non-fixed fluctuation
A074 | Differencing | subtract lag-1 (or seasonal) to remove trend/seasonality
A075 | Why naive/persistence is the baseline for series | smooth signals: last value is often unbeatable at horizon 1
A076 | Lag features for ML forecasting | use past values as columns; pick lags by ACF/PACF or domain
A077 | Rolling features and leakage | rolling stats must use only past data (shift before computing)
A078 | Time-series split vs KFold | KFold shuffles time -> leaks future; use expanding/rolling window
A079 | ARIMA vs ML forecasting when | ARIMA for small smooth series + interpretability; ML (trees/GBM with lags) for rich exogenous features
A080 | Forecast horizon effect | persistence decays with horizon; ML feature models win longer horizons (EXAMPLE.py S21)
A081 | Batch vs online inference | batch: precompute on schedule; online: per-request prediction
A082 | When online inference needed | decision latency < minutes (fraud, ads, chat)
A083 | Feature store purpose | single source of features for training and serving (consistency)
A084 | Point-in-time correctness | features must reflect what was known at prediction time (no future)
A085 | Training-serving skew | offline vs online features differ -> silent degradation
A086 | Model serving options | REST (FastAPI), gRPC, batch job, edge; match latency need
A087 | Latency budget for a fraud check | ~50-100 ms end to end: gateway + features + model
A088 | How to scale inference | stateless replicas + load balancer; batch requests; cache common queries
A089 | Queue vs synchronous inference | async queue for batch/backpressure; sync for interactive
A090 | Feature freshness SLA | depends on decision loop: real-time features for fraud, daily for churn scoring
A091 | Model registry purpose | version, stage, lineage, rollback
A092 | CI/CD for ML: what CI tests | data schema, pipeline code, unit tests, training reproducibility, eval vs baseline
A093 | CD for models | automated deploy after gates: canary/shadow, rollback on metric drop
A094 | Shadow deployment | run new model on live traffic, compare offline, don't serve its answers
A095 | Canary deployment | serve small % of traffic; promote if guardrails hold
A096 | Docker role for models | package code+deps+weights for reproducible deploys
A097 | MLflow tracking unit | a run = code version + params + metrics + artifacts
A098 | Data versioning vs code versioning | dataset hash/lineage in DVC so runs are reproducible
A099 | When not to use Kubernetes | small team, tiny traffic: managed serverless/containers simpler
A100 | Model monitoring stack (4 signals) | data drift, concept drift, prediction drift, performance (when labels arrive)
A101 | Alert threshold design | base on normal variability; alert on trend, not single point
A102 | Drift detected - then what? | triage: data bug vs real drift -> retrain/revert/adjust features
A103 | Why monitor input distribution | silent feature bugs (unit changes, null rates) break models
A104 | Golden-set evaluation | fixed labeled set re-scored on every deploy
A105 | Model health without labels | compare prediction distribution & feature stats to reference
A106 | Feedback loop risk | model decisions change future labels (fraud, recsys) - monitor label distribution
A107 | Inference cost drivers | tokens/latency/compute + infra; cache & batch to cut
A108 | Cloud cost control for training | spot instances, right-size GPUs, checkpoints, experiment hygiene
A109 | When to move from API model to self-hosted | volume/price/latency/privacy cross a threshold - measure it
A110 | Auto-scaling on what metric | queue depth / p95 latency, not just CPU
A111 | Failure modes of a model API (list 5) | 4xx validation, 5xx upstream, timeout, stale model, drift
A112 | Retry strategy for flaky inference | exponential backoff + idempotency + circuit breaker
A113 | How you find a silent accuracy drop | golden set + drift monitors + periodic labeled sampling
A114 | Postmortem structure | impact, timeline, root cause, prevention, action items
A115 | SLA vs SLO vs SLI | indicator (latency), objective (99% < 200 ms), agreement with consequences
A116 | Error analysis playbook (5 questions) | which slices fail, which classes, which features, leakage?, data vs model issue
A117 | Why your CV beat production | training-serving skew: different features, timing, data prep
A118 | Debugging a model regression after deploy | diff inputs, features, code, data window, then rollback
A119 | Your model is 2% better offline but not online | online confounders (cache, selection, delay); A/B test properly
A120 | Handling a pipeline bug discovered late | freeze deploys, recompute affected artifacts, annotate lineage
A121 | Explain an end-to-end ML system you built | ingest -> validate -> features -> train/eval -> registry -> serve -> monitor -> retrain
A122 | Design fraud detection for 1k req/s | gateway, feature fetch (Redis), model (batched GPU), rules pre-filter, async alerts
A123 | Design a recommendation refresh at 10-min cadence | batch compute candidates, cache, serve top-k, log interactions
A124 | Where do you put the model: edge vs cloud | privacy/latency vs updateability/compute
A125 | Multi-model routing | route by cost/quality/latency tier with fallbacks
A126 | Model versioning strategy | registry + immutable artifacts + explicit traffic split
A127 | Handling PII in features | tokenize/minimize, access control, retention policy, don't log raw values
A128 | Why log predictions | audit, debugging, drift analysis; store with feature snapshot
A129 | Cold start for new users/products | fallback rules, popularity priors, content-based features
A130 | Concept drift in a recsys | popularity cycles; retrain schedule must track content churn
A131 | Long-tail problem in production ML | rare slices lack data; stratify eval by slice, use hierarchy
A132 | When is rule-based better than the model | high-stakes simple logic, explainability mandates, cold start
A133 | Human-in-the-loop design | model flags uncertain/low-confidence cases to reviewers
A134 | Cascading failures | upstream feature service down -> stale features -> bad predictions; fail closed/fallback
A135 | Why experiment on real traffic (A/B) | offline metrics don't capture behavior changes/interactions
A136 | A/B test pitfalls | interference between arms, novelty effect, early stopping, multiple metrics
A137 | How long to run an A/B test | until power reached for the effect size you care about
A138 | Segment analysis after A/B | guardrail + segment metrics to catch heterogeneous effects
A139 | Metric that gaming can break | click-through optimized clicks, not value; design goal metrics
A140 | Shift in business metric vs model metric | monitor the business outcome, not just model score
A141 | Cost-benefit of retraining | quality gain vs compute/risk; schedule on drift not calendar alone
A142 | Explain MLOps vs DevOps | MLOps adds data/model versioning, eval gates, drift monitoring
A143 | Reproducibility blockers | unseeded runs, env drift, floating data, nondeterministic GPU ops
A144 | Data quality gate in prod | schema + range + null-rate + drift checks before retraining
A145 | Managing multiple models | registry, owners, per-model SLOs and dashboards
A146 | Feature ownership | one owner per feature; contracts for schema and semantics
A147 | Why documentation matters in prod ML | models decay and change; docs = the operating manual
A148 | Capacity planning for inference | peak QPS x latency x replicas with headroom + autoscale
A149 | When to rebuild vs retrain | feature/pipeline redesign or new data source -> rebuild; else retrain
A150 | Final system question - sketch a churn-warning platform | events -> features -> scheduled score -> CRM action -> outcome tracking -> drift + retraining

================================================================================
EXPERT (X001-X100) - deep theory, statistics, research method, framing
================================================================================
X001 | Derive the bias-variance decomposition | E[(y-f)^2] = noise + (E[f]-y)^2 + E[(f-E[f])^2]
X002 | Why can't you minimize bias and variance together? | capacity moves both oppositely; total has a minimum
X003 | Bayes error vs irreducible noise | smallest achievable error; you can never beat it
X004 | What does Rademacher complexity bound? | uniform deviation of the hypothesis class - capacity measure
X005 | VC dimension intuition | largest set of points the class can shatter
X006 | Why overparameterized models generalize | implicit regularization (SGD bias, margins, simplicity); active research
X007 | Double descent | test error rises then FALLS past interpolation threshold
X008 | Consistency of kNN | with k->inf and k/n->0, kNN converges to Bayes risk (under conditions)
X009 | Why L2 regularization = Gaussian prior | MAP with Normal(0, sigma^2) prior on weights
X010 | Why L1 = Laplace prior | MAP with Laplace prior; sharp peak at 0 -> sparsity
X011 | Central limit vs law of large numbers | LLN: mean -> expectation; CLT: distribution of mean -> Normal
X012 | What is a sufficient statistic? | captures all info about the parameter in the data (e.g., mean for Normal)
X013 | Likelihood vs probability | likelihood: P(data|param) as function of param (not a distribution over param)
X014 | Fisher information meaning | curvature of log-likelihood; lower bound on estimator variance (Cramer-Rao)
X015 | Confidence interval misinterpretation to correct | "95% chance the true value is in THIS interval" - it is about the procedure
X016 | Why do p-values mislead at scale | 1000 tests -> ~50 false positives at 0.05; multiplicity
X017 | Sequential testing problem | peeking inflates type I error; correct with alpha spending/bayes factors
X018 | Bootstrap vs asymptotic CI | bootstrap: resample data for SE - fewer assumptions; asymptotic: CLT-based formula
X019 | Permutation test idea | shuffle labels to build the null distribution of a statistic
X020 | When are t-tests invalid for model comparison | paired/decorrelated comparisons needed; t assumes independence
X021 | Entropy interpretation | average number of bits needed to encode outcomes of X
X022 | Cross entropy = entropy + KL | H(P,Q) = H(P) + KL(P||Q)
X023 | Why minimize CE = maximize likelihood | CE = -E[log q]; minimizing = MLE under q
X024 | Mutual information and MI=0 | I(X;Y)=0 iff X,Y independent
X025 | Why use log-loss not 0-1 loss for learning | 0-1 is non-smooth, no gradient; CE is a smooth surrogate
X026 | Information gain vs Gini equivalence | similar impurity orderings; Gini cheaper (no log)
X027 | Conditional entropy H(Y|X) meaning | remaining uncertainty in Y given X
X028 | KL is not a distance | asymmetric, violates triangle inequality
X029 | JS divergence fix | symmetrized + bounded version of KL
X030 | Why softmax + CE numerically | log-softmax avoids overflow/underflow (subtract max)
X031 | Gradient of CE wrt logits | (softmax(z) - y) - clean linear error signal
X032 | Convexity and global optima | convex loss -> every local min is global; linear models convex, NNs not
X033 | SGD convergence rate | ~1/sqrt(t) for general convex; faster with strong convexity
X034 | Why momentum helps | averages gradient noise, damps oscillation in narrow valleys
X035 | What is a saddle point problem | many zero-gradient directions in high-dim loss; noise/adaptive methods escape
X036 | Second-order methods why rare in DL | Hessian O(p^2) infeasible; approximate (K-FAC, Adam as diagonal proxy)
X037 | Condition number effect on GD | ratio of max/min curvature; ill-conditioned zig-zags
X038 | Weight decay = L2 on what | shrinks weights toward 0 every step, decoupled in AdamW
X039 | BatchNorm why helps | normalizes layer inputs -> smoother loss landscape + higher lr
X040 | LayerNorm vs BatchNorm for sequences | LayerNorm per-token, no batch dependence - works for variable length
X041 | What is representation learning | learning useful internal features from data rather than hand-crafting
X042 | Self-supervised learning idea | build labels from the data itself (contrastive, masked prediction)
X043 | Contrastive loss idea | pull positives together, push negatives apart in embedding space
X044 | Why do embeddings transfer? | lower layers capture generic structure reusable across tasks
X045 | Autoencoder bottleneck effect | forces compressed representation; may discard rare but important info
X046 | SimCLR vs BYOL | SimCLR needs negatives; BYOL uses a momentum teacher - no negatives
X047 | Why pretraining helps small data | warm start regularizes + transfers features
X048 | Catastrophic forgetting | fine-tuning erases old knowledge; fix with replay, regularizers (EWC), LoRA
X049 | Few-shot learning strategies | meta-learning, strong priors/pretraining, in-context learning
X050 | When do foundation models help vs hurt | help when distribution matches pretraining; hurt on specialized/rare domains
X051 | Reproducing a paper: first 3 steps | fix data/splits, reimplement baseline, match eval code before touching the new method
X052 | Why ablate your own components | prove each piece contributes; prevent incremental complexity claims
X053 | Benchmark contamination check | overlap test between train corpus and eval set; n-gram probes
X054 | Leaderboard overfitting | many teams tune on public test -> test becomes training; be skeptical
X055 | Paper result vs your result gap (sources) | data, seeds, hardware, eval code, hyperparams - isolate each
X056 | How to read a paper fast | abstract -> figures/results -> method -> related work last
X057 | What makes a good research question | falsifiable, measurable, non-obvious, feasible
X058 | Ablation ordering | remove strongest-claim component first; test interactions later
X059 | Statistical claim checklist for papers | repeated seeds, CI/error bars, paired tests, effect size
X060 | When is a baseline comparison unfair | different compute/budget, tuning effort, or data
X061 | Designing a novel experiment | hypothesis -> controls -> metrics -> ablations -> robustness checks
X062 | Negative result value | saves others' time; publish with clean setup and diagnostics
X063 | Reproducibility kit contents | code, seeds, env lock, data hashes, run configs, eval scripts
X064 | HPO as part of a paper claim | tuned fairly for ALL methods or state tuning budgets
X065 | Out-of-distribution generalization eval | test on shifted data; report not just iid accuracy
X066 | The limits of your model class | linear can't XOR; trees can't extrapolate; deep nets need scale/data
X067 | When is more data not the answer | label noise high, wrong objective, irreducible error, sampling bias
X068 | Framing problem: regression metric on ranking task | you optimized RMSE; you need ranking quality (NDCG) - reframe
X069 | Impossibility: what can't ML do for you | guarantee fairness, infer causation from correlation alone, beat Bayes error
X070 | Explaining your model to a domain expert | SHAP top factors + counterfactual "what would change the decision"
X071 | Ethical ceiling vs technical ceiling | fairness/accountability limits usually bind before accuracy does
X072 | When you should NOT trust your CV | leakage, non-iid data (time/groups), tiny sample, distribution shift
X073 | Diagnosing a NaNs/exploding training run | lr too high, unscaled inputs, bad init, no clipping - bisect
X074 | Model gives confident wrong answers | calibration + domain coverage check; maybe data mismatch
X075 | Silent data bug hunt (no labels) | schema diff, null rate jump, units change, join cardinality
X076 | Overfit to validation during research | hold out a fresh set; report nested estimates
X077 | Your feature importance disagrees with domain knowledge | check correlated features, scaling, leakage, then trust evidence carefully
X078 | Why you might still ship a slightly worse model | robustness, latency, cost, explainability, simpler ops
X079 | Debugging a gradient issue | check grad norms per layer, NaN source, loss at init (should match theory)
X080 | Failure forensics: production model died silently | golden set -> drift monitors -> logs -> feature replay -> rollback
X081 | How to think about uncertainty in deployment | never ship point estimates for high-stakes; CI + fallback thresholds
X082 | Scientific integrity in ML | report all trials, not the best seed; version data; publish ablations
X083 | Theory vs practice tension | theory guides (capacity, bias-variance) but practice wins on engineering + data
X084 | Evaluate a claim "my model is 99.9% accurate" | ask: on what data, which classes, what baseline, over what time
X085 | Design an eval for a rare-event detector | PR@k, calibration on events, temporal holdout, false-alarm cost
X086 | Why offline gains vanish online | distribution/serving differences; measure online with A/B
X087 | Multi-task vs single-task when | related tasks + data sharing help; conflicting labels hurt
X088 | Label noise handling at scale | confident learning / cleanlab, noise-robust loss, evaluate on clean subset
X089 | Synthetic data risks | distribution mismatch, benchmark contamination, hidden artifacts
X090 | Active learning idea | query labels where model uncertainty/data value is highest
X091 | Curriculum learning | order training from easy to hard; can speed convergence
X092 | Why a model can be calibrated yet useless | calibrated margins but no separation - need discrimination too
X093 | Dimensionality vs sample size (p >> n) | regularization/priors essential; naive MLE explodes
X094 | Interpretability of deep models honestly | feature attributions approximate; test by perturbation
X095 | When is transfer harmful | source too different, negative transfer, catastrophic forgetting
X096 | Metric design trap | optimizing proxy (clicks) while goal (value) drifts - align metrics to goals
X097 | Handling long-horizon feedback | credit assignment hard (churn after months) - surrogate early signals + validation
X098 | What experiment proves causality in your system | randomized rollout with guardrail metrics
X099 | Communicating uncertainty to leadership | ranges + scenarios + cost of being wrong, not just point numbers
X100 | The question behind every ML interview | can you go from messy data to a reliable, monitored decision and defend it

================================================================================
END OF INTERVIEW BANK - PART 4 (EXPERT 100)
================================================================================
COUNTS: B100 + I150 + A150 + X100 = 500 Q&A (meets master-prompt targets)
