# ADVANCED SYSTEM DESIGN EXERCISES
# Complete these exercises to practice system design for AI/ML

"""
EXERCISE 1: Design a Real-Time Recommendation System
=====================================================

Design a system that provides real-time recommendations to users.

Requirements:
- Handle 10 million users
- Handle 100 million items
- Sub-100ms latency
- Update recommendations in real-time
- Handle cold start

Your Task:
1. Draw the architecture diagram (ASCII)
2. Choose appropriate technologies
3. Design the data pipeline
4. Design the serving layer
5. Handle cold start problem
6. Design the feedback loop
"""

# Your solution here:

"""
ARCHITECTURE:
┌─────────────────────────────────────────────┐
│               Real-Time Rec System          │
│                                             │
│  ┌──────────┐    ┌──────────┐               │
│  │ User     │    │ Feature  │               │
│  │ Request  │───→│ Store    │               │
│  └────┬─────┘    └────┬─────┘               │
│       ↓               ↓                     │
│  ┌──────────┐    ┌──────────┐               │
│  │ API      │←──│ Redis    │               │
│  │ Gateway  │    │ Cache    │               │
│  └────┬─────┘    └──────────┘               │
│       ↓                                     │
│  ┌──────────┐    ┌──────────┐               │
│  │ Rec      │    │ Candidate│               │
│  │ Service  │───→│ Gen      │               │
│  └────┬─────┘    └──────────┘               │
│       ↓                                     │
│  ┌──────────┐    ┌──────────┐               │
│  │ Ranker   │    │ Post-    │               │
│  │ Model    │───→│ Process  │               │
│  └──────────┘    └──────────┘               │
│                                             │
│  Data Pipeline:                             │
│  Kafka → Spark → Feature Store → Model     │
└─────────────────────────────────────────────┘
"""


"""
EXERCISE 2: Design a Fraud Detection System
============================================

Design a system that detects fraudulent transactions in real-time.

Requirements:
- Process 10,000 transactions per second
- Sub-50ms detection latency
- Handle class imbalance (0.1% fraud)
- Provide explanations for decisions
- Adapt to new fraud patterns

Your Task:
1. Design the real-time pipeline
2. Choose appropriate models
3. Design the feature store
4. Design the explanation system
5. Design the feedback loop
6. Design monitoring
"""

# Your solution here:


"""
EXERCISE 3: Design a RAG-Based Knowledge Assistant
===================================================

Design a system that answers questions from enterprise documents.

Requirements:
- Support 100,000+ documents
- Sub-2-second response time
- Provide citations
- Handle document updates
- Multi-tenant support

Your Task:
1. Design the ingestion pipeline
2. Design the retrieval system
3. Design the generation system
4. Design the citation system
5. Design the update mechanism
6. Design multi-tenancy
"""

# Your solution here:


"""
EXERCISE 4: Design an A/B Testing Platform
==========================================

Design a platform for running A/B tests on ML models.

Requirements:
- Support multiple concurrent experiments
- Statistical significance testing
- Gradual rollout
- Rollback capability
- Metrics tracking

Your Task:
1. Design the experiment configuration
2. Design the traffic splitting
3. Design the metrics collection
4. Design the significance testing
5. Design the rollout mechanism
6. Design the reporting dashboard
"""

# Your solution here:


"""
EXERCISE 5: Design a Model Monitoring System
============================================

Design a system that monitors ML model performance in production.

Requirements:
- Detect data drift
- Detect concept drift
- Track model performance
- Alert on anomalies
- Support multiple models

Your Task:
1. Design the metrics collection
2. Design the drift detection
3. Design the alerting system
4. Design the dashboard
5. Design the retraining trigger
6. Design the model versioning
"""

# Your solution here:


"""
EXERCISE 6: Design a Scalable ML Training Pipeline
==================================================

Design a system for training ML models at scale.

Requirements:
- Handle terabyte-scale datasets
- Support distributed training
- Experiment tracking
- Model versioning
- Automated hyperparameter tuning

Your Task:
1. Design the data pipeline
2. Design the training infrastructure
3. Design the experiment tracking
4. Design the model registry
5. Design the hyperparameter tuning
6. Design the deployment pipeline
"""

# Your solution here:


"""
GRADING RUBRIC:
- Architecture clarity: 20%
- Technology choices: 20%
- Scalability considerations: 20%
- Error handling: 15%
- Monitoring and observability: 15%
- Documentation: 10%
"""
