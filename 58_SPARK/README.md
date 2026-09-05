# Module 58: Spark & Distributed Data (PySpark)

_(Renumbered from Module 33 on 2026-09-03 to resolve the numbering
collision with 33_MLOPS.)_

Processing datasets too big for one machine: lazy DataFrames, distributed
compute, and the Spark ML ecosystem.

## What You Will Learn

- When you outgrow pandas: memory, single-node limits
- Spark architecture: driver, executors, partitions, DAG
- PySpark DataFrames vs pandas (schema, lazy evaluation)
- Transformations vs actions; caching and persistence
- Common operations: filters, aggregations, joins, window functions
- Reading/writing Parquet and partitioning
- Spark SQL basics
- MLlib overview; distributed preprocessing
- When Spark is right vs pandas vs Polars vs Dask

## Module Files

| File | Topic |
|------|-------|
| spark_complete.txt | Full PySpark course |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

## Prerequisites

- 03_PANDAS (you must know what Spark replaces)
- 40_DATABASES/SQL basics helpful

## Exit Criteria

- [ ] You can explain lazy evaluation and the DAG
- [ ] You can rewrite a pandas pipeline in PySpark
- [ ] You know the memory/compute trade-offs of distributed data
