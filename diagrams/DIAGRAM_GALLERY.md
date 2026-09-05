# AI/ML Engineering — Diagram Gallery Index

64 diagrams, grouped by course / topic. Regenerate with `python diagrams/generate_gallery.py`; integrity review: `python diagrams/verify_diagrams.py`.

## LLM Course — internals, training, inference (diagrams/llm/)

| Diagram | Description |
|---------|-------------|
| ![kv_cache](llm/kv_cache.png)<br>`kv_cache.png` 939x639 | O(T^2) full recompute vs O(T) cached decoding, annotated with measured speedups (2.4x @ T=128 ... 11x @ T=768) |
| ![kv_cache_memory](llm/kv_cache_memory.png)<br>`kv_cache_memory.png` 917x591 | KV cache memory growth: ~0.5 MiB/token => 2 GiB @ 4k, 64 GiB @ 128k context |
| ![llm_architecture](llm/llm_architecture.png)<br>`llm_architecture.png` 1004x1057 | Decoder-only GPT stack: token embeddings -> RoPE -> N blocks (RMSNorm, MHA, SwiGLU FFN) -> final norm -> LM head -> sampling |
| ![lora](llm/lora.png)<br>`lora.png` 1167x648 | W' = W + (alpha/r) B A: adapter low-rank decomposition with the verified 48x trainable-parameter reduction |
| ![prefill_decode](llm/prefill_decode.png)<br>`prefill_decode.png` 1595x610 | Prefill phase (parallel, compute-bound) vs decode phase (one token at a time, memory-bandwidth-bound) |
| ![pretraining_pipeline](llm/pretraining_pipeline.png)<br>`pretraining_pipeline.png` 1764x554 | Raw data -> filter -> clean -> dedup -> tokenize -> pack -> shuffle -> batch -> train -> validate -> checkpoint |
| ![quantization](llm/quantization.png)<br>`quantization.png` 917x618 | 7B model memory: FP32 28 GB -> FP16/BF16 14 GB -> INT8 7 GB -> INT4 3.5 GB |
| ![sampling](llm/sampling.png)<br>`sampling.png` 1595x610 | Temperature reshaping of logits [2, 1, 0.1] (T=0.1 sharp, T=1.0, T=2.0 flat), top-k and top-p truncation |
| ![scaling_laws](llm/scaling_laws.png)<br>`scaling_laws.png` 1171x639 | REAL lab sweep: val loss 2.723 (S) -> 2.617 (M) -> 2.421 (L) vs bigram 2.763; ppl 15.2/13.7/11.3 |
| ![speculative_decoding](llm/speculative_decoding.png)<br>`speculative_decoding.png` 1275x583 | Draft (gamma tokens) -> verify in parallel -> accept/reject -> resample: ~2.6x fewer forwards |

## GenAI + Agents Course — RAG agents, multi-agent, bench (diagrams/agents/)

| Diagram | Description |
|---------|-------------|
| ![agent_loop](agents/agent_loop.png)<br>`agent_loop.png` 1167x756 | Observe -> reason -> act (tool call) -> observe result -> repeat until final answer; tool registry + safety guards |
| ![embedding_bench](agents/embedding_bench.png)<br>`embedding_bench.png` 1594x609 | REAL lab numbers: dense 9/9 vs lexical 6/9 hit@1; paraphrases 6/6 vs 4/6; HF MiniLM matches local SVD |
| ![multi_agent](agents/multi_agent.png)<br>`multi_agent.png` 1221x756 | Supervisor pattern: supervisor routes to writer / reviewer / researcher specialists, results return to supervisor |
| ![prod_rag_server](agents/prod_rag_server.png)<br>`prod_rag_server.png` 1329x775 | Production RAG server: auth/RBAC -> per-key quota -> Redis-style cache -> RAG brain -> SQL request log |
| ![rag_agent](agents/rag_agent.png)<br>`rag_agent.png` 1406x734 | RAG agent: query -> retrieve -> ground -> generate with citations; tool calls for calculator/retrieval; injection guard |

## ML Course — lifecycle + model diagnostics (diagrams/ml/)

| Diagram | Description |
|---------|-------------|
| ![confusion_matrix](ml/confusion_matrix.jpg)<br>`confusion_matrix.jpg` 922x885 | TP/TN/FP/FN layout with derived metrics (accuracy, precision, recall, F1) |
| ![cross_validation](ml/cross_validation.jpg)<br>`cross_validation.jpg` 1785x734 | K-fold CV: fold rotation, train/validation blocks per fold |
| ![decision_boundary](ml/decision_boundary.jpg)<br>`decision_boundary.jpg` 1184x884 | Classifier decision boundary vs data points (linear and non-linear) |
| ![feature_importance](ml/feature_importance.png)<br>`feature_importance.png` 1046x637 | Permutation / tree-based feature importance ranking with drop-off |
| ![imbalance](ml/imbalance.png)<br>`imbalance.png` 1469x637 | Class imbalance: skewed distribution + SMOTE oversampling of the minority class |
| ![kmeans](ml/kmeans.png)<br>`kmeans.png` 1418x637 | K-means iterations: centroids moving, assignments updating, convergence on the objective |
| ![learning_curves](ml/learning_curves.png)<br>`learning_curves.png` 923x639 | Train vs validation error as training set grows: high-variance (gap) vs high-bias (plateau) diagnosis |
| ![ml_lifecycle](ml/ml_lifecycle.png)<br>`ml_lifecycle.png` 1764x576 | Full ML lifecycle: data -> EDA -> features -> train/val/test -> baseline -> model -> tune -> deploy -> monitor -> drift -> retrain (closed loop) |
| ![ml_workflow](ml/ml_workflow.jpg)<br>`ml_workflow.jpg` 2385x885 | End-to-end ML workflow diagram (data -> modeling -> evaluation -> deployment) |
| ![model_comparison](ml/model_comparison.jpg)<br>`model_comparison.jpg` 1785x1035 | Model comparison: accuracy/latency/size trade-offs |
| ![overfitting](ml/overfitting.jpg)<br>`overfitting.jpg` 1485x884 | Overfitting vs underfitting vs good fit curves |
| ![pca](ml/pca.png)<br>`pca.png` 1524x638 | PCA projection: data cloud, first principal component direction, explained-variance scree |
| ![pr_curve](ml/pr_curve.png)<br>`pr_curve.png` 923x637 | Precision-recall curve with baseline and trade-off regions (for imbalanced classes) |
| ![roc_curve](ml/roc_curve.jpg)<br>`roc_curve.jpg` 1184x884 | ROC curve with AUC annotation |
| ![test](ml/test.jpg)<br>`test.jpg` 1200x900 | Train / validation / test split with leakage warning |
| ![time_series](ml/time_series.png)<br>`time_series.png` 1029x637 | Trend + seasonality decomposition with forecast over the historical series |

## Mathematics (diagrams/math/)

| Diagram | Description |
|---------|-------------|
| ![bias_variance](math/bias_variance.jpg)<br>`bias_variance.jpg` 1484x884 | Bias-variance decomposition: low/high bias x low/high variance quadrants |
| ![entropy_gini](math/entropy_gini.jpg)<br>`entropy_gini.jpg` 2085x734 | Entropy and Gini impurity as functions of class probability |
| ![gradient_descent](math/gradient_descent.jpg)<br>`gradient_descent.jpg` 1484x1185 | Gradient descent: cost surface, steps to minimum, learning-rate effect |
| ![sigmoid](math/sigmoid.jpg)<br>`sigmoid.jpg` 1484x884 | Sigmoid curve with its derivative |

## Deep Learning (diagrams/dl/)

| Diagram | Description |
|---------|-------------|
| ![activation_functions](dl/activation_functions.jpg)<br>`activation_functions.jpg` 1784x1522 | Sigmoid / tanh / ReLU / GELU / SiLU curves with derivative behavior |
| ![backpropagation](dl/backpropagation.jpg)<br>`backpropagation.jpg` 2085x734 | Backpropagation flow: forward pass, loss, gradients flowing backward through the graph |
| ![cnn_architecture](dl/cnn_architecture.jpg)<br>`cnn_architecture.jpg` 2085x734 | CNN stack: conv -> pool -> conv -> pool -> flatten -> dense |
| ![gradient_descent](dl/gradient_descent.jpg)<br>`gradient_descent.jpg` 2233x772 | Neural-net gradient descent: loss landscape and optimization path |
| ![lstm_gates](dl/lstm_gates.jpg)<br>`lstm_gates.jpg` 2085x884 | LSTM cell: forget / input / output gates and cell-state highway |
| ![neural_network](dl/neural_network.jpg)<br>`neural_network.jpg` 1485x1034 | MLP: input -> hidden layers -> output with weights and activations |

## Backend / APIs (diagrams/backend/)

| Diagram | Description |
|---------|-------------|
| ![api_flow](backend/api_flow.jpg)<br>`api_flow.jpg` 2085x734 | API request flow: client -> route -> validation -> service -> response |
| ![backend_architecture](backend/backend_architecture.jpg)<br>`backend_architecture.jpg` 2085x1184 | Backend architecture: FastAPI, DB, cache, workers |

## NLP (diagrams/nlp/)

| Diagram | Description |
|---------|-------------|
| ![nlp_pipeline](nlp/nlp_pipeline.jpg)<br>`nlp_pipeline.jpg` 2385x734 | NLP pipeline: raw text -> clean -> tokenize -> features -> model |
| ![sentiment_analysis](nlp/sentiment_analysis.jpg)<br>`sentiment_analysis.jpg` 1482x884 | Sentiment classification flow with example scores |
| ![word_embeddings](nlp/word_embeddings.jpg)<br>`word_embeddings.jpg` 1485x1185 | Word embeddings: high-dim one-hot to dense vector space with similar words near each other |

## RAG (diagrams/rag/)

| Diagram | Description |
|---------|-------------|
| ![rag_pipeline](rag/rag_pipeline.jpg)<br>`rag_pipeline.jpg` 2385x884 | RAG pipeline: documents -> chunk -> embed -> vector DB -> retrieve -> LLM -> grounded answer |
| ![rag_vs_finetuning](rag/rag_vs_finetuning.jpg)<br>`rag_vs_finetuning.jpg` 2085x885 | RAG vs fine-tuning comparison: knowledge updates, cost, hallucination |

## Transformers (diagrams/transformers/)

| Diagram | Description |
|---------|-------------|
| ![self_attention](transformers/self_attention.jpg)<br>`self_attention.jpg` 1485x1034 | Scaled dot-product attention: Q, K, V, scores, softmax, weighted sum |
| ![transformer_architecture](transformers/transformer_architecture.jpg)<br>`transformer_architecture.jpg` 1785x1484 | Transformer block: attention + add&norm + FFN + add&norm |

## Training-run graphs (diagrams/graphs/)

| Diagram | Description |
|---------|-------------|
| ![gradient_descent_from_scratch](graphs/gradient_descent_from_scratch.png)<br>`gradient_descent_from_scratch.png` 1320x495 | From-scratch gradient descent run: loss over iterations |
| ![lstm_sine_forecast](graphs/lstm_sine_forecast.png)<br>`lstm_sine_forecast.png` 1000x350 | LSTM sine-wave forecast vs ground truth |
| ![nn_training_loss](graphs/nn_training_loss.png)<br>`nn_training_loss.png` 1200x750 | Neural-network training loss curves |
| ![pca_demo](graphs/pca_demo.png)<br>`pca_demo.png` 1200x900 | PCA demo projection |
| ![vis_01_loss_curve](graphs/vis_01_loss_curve.png)<br>`vis_01_loss_curve.png` 550x385 | Training loss curve (matplotlib basics) |
| ![vis_02_regression](graphs/vis_02_regression.png)<br>`vis_02_regression.png` 550x385 | Regression fit example |
| ![vis_03_confusion](graphs/vis_03_confusion.png)<br>`vis_03_confusion.png` 440x385 | Confusion matrix example |
| ![vis_04_roc](graphs/vis_04_roc.png)<br>`vis_04_roc.png` 550x385 | ROC curve example |

## Graph ML (diagrams/graph/)

| Diagram | Description |
|---------|-------------|
| ![link_prediction_auc](graph/link_prediction_auc.png)<br>`link_prediction_auc.png` 793x575 | GCN link prediction: real AUC benchmark numbers |
| ![message_passing](graph/message_passing.png)<br>`message_passing.png` 1089x554 | Graph message passing: node -> aggregate neighbors -> update (GCN step) |

## Recommenders (diagrams/recommenders/)

| Diagram | Description |
|---------|-------------|
| ![als_vs_sgd_rmse](recommenders/als_vs_sgd_rmse.png)<br>`als_vs_sgd_rmse.png` 793x547 | ALS vs SGD matrix factorization: real RMSE benchmark numbers |
| ![two_stage_recsys](recommenders/two_stage_recsys.png)<br>`two_stage_recsys.png` 1069x626 | Two-stage recommender: candidate generation -> ranking |

## Reinforcement Learning (diagrams/rl/)

| Diagram | Description |
|---------|-------------|
| ![agent_env_loop](rl/agent_env_loop.png)<br>`agent_env_loop.png` 1069x531 | RL loop: agent <-> environment, action/state/reward cycle |
| ![qlearning_vs_optimal](rl/qlearning_vs_optimal.png)<br>`qlearning_vs_optimal.png` 775x575 | Q-learning returns vs optimal: real benchmark numbers |
