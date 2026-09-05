# Math Cards - per-algorithm quick-reference library

One card per algorithm family: the OBJECTIVE, the LOSS, the UPDATE or
closed form, the COMPLEXITY, the KEY EQUATIONS with every symbol defined,
the FAILURE MODES, and the one-line connection to other algorithms.
Use these as revision sheets before interviews and as anchors while
reading the full lessons.

| Card file | Algorithms |
|-----------|------------|
| cards_01_linear_models.txt | Linear regression (OLS/normal eq), Ridge, Lasso/ElasticNet, Logistic regression, softmax + cross-entropy |
| cards_02_trees_ensembles.txt | Decision trees, Random Forest, AdaBoost, Gradient Boosting/XGBoost |
| cards_03_distance_probability.txt | KNN, Naive Bayes, SVM, perceptron |
| cards_04_unsupervised.txt | K-Means, Gaussian Mixture Model, PCA, DBSCAN, anomaly scores |
| cards_05_nn_optimizers.txt | MLP + backprop, gradient descent family (SGD/momentum/AdaGrad/RMSProp/Adam) |
| cards_06_sequences_attention.txt | RNN/LSTM gates, scaled dot-product attention |
| cards_07_anomaly_time_series.txt | Isolation Forest score, ARIMA/SARIMA, Holt-Winters |
| cards_08_graph_rl.txt | GCN/GAT/GraphSAGE message passing, MDP/Bellman, Q-learning/DQN, policy gradients/PPO |

Format of every card:
```
CARD: <name>
PROBLEM       what it solves
MODEL         the function/assumptions
OBJECTIVE     the quantity optimized (write it out)
CLOSED FORM   or UPDATE RULE (with every symbol defined)
KEY PROPERTY  what makes it work / the guarantee
COMPLEXITY    train / predict
FAILURES      when it breaks
CONNECTS TO   other cards
```
