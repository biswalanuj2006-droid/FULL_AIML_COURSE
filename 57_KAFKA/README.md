# Module 57: Kafka for Real-Time AI

_(Renumbered from Module 32 on 2026-09-03 to resolve the numbering
collision with 32_FINE_TUNING.)_

Streaming events through Kafka so models predict on data as it arrives —
the backbone of real-time fraud, recommendation, and monitoring systems.

## What You Will Learn

- Why streaming: batch latency is too slow for some decisions
- Kafka architecture: brokers, topics, partitions, offsets
- Producers and consumers; consumer groups and rebalancing
- Delivery semantics: at-most-once, at-least-once, exactly-once concepts
- Ordering, retention, and replication basics
- Python clients: producing and consuming events
- Serialization and schema considerations
- Real-time ML: consume events → score with a model → store result
- Failure modes: lag, poison messages, consumer crashes

## Module Files

| File | Topic |
|------|-------|
| kafka_complete.txt | Full streaming course |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

## Prerequisites

- Python concurrency basics (54_ADVANCED_PYTHON)
- 40_DATABASES helpful for result storage

## Exit Criteria

- [ ] You can explain topic/partition/offset in one minute
- [ ] You can run a producer-consumer pipeline locally
- [ ] You built a streaming prediction pipeline with a real model
