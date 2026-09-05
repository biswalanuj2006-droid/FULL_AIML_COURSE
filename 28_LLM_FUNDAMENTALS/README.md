# Module 28: LLM Fundamentals

What a large language model actually is under the API: training stages,
tokens, weights, and the engineering around using them well.

## What You Will Learn

- Scaling: parameters, data, compute; why "large"
- Training pipeline: pretraining → instruction tuning → alignment
- Tokenizers (BPE intuition) and vocabulary design
- Context windows and their limits; long-context techniques overview
- Working with models via API and via Hugging Face locally
- Structured outputs, JSON mode, function calling
- Cost/rate-limit engineering and caching
- Quantization concepts (int8/FP16) for running open models
- Evaluation of LLM outputs; benchmarks vs your own evals
- Open vs closed models: when each makes sense

## Module Files

| File | Topic |
|------|-------|
| llm_fundamentals.txt | Complete foundations notes |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

Related later modules: 29_RAG, 32_FINE_TUNING, 34_MODEL_DEPLOYMENT.

## Prerequisites

- 26_TRANSFORMERS, 27_GENAI_FOUNDATIONS

## Exit Criteria

- [ ] You can explain pretraining vs fine-tuning in one minute
- [ ] You can call an LLM API with structured output
- [ ] You can run a small open model locally and quantify its speed
