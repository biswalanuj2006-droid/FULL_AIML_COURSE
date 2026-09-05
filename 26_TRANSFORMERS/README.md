# Module 26: Transformers

The architecture behind BERT, GPT, and every modern LLM — encoder, decoder,
multi-head attention, and the Hugging Face ecosystem.

## What You Will Learn

- From attention to Transformer: why parallelism beats recurrence
- Multi-head attention, positional encoding, feed-forward blocks
- Residual connections and layer norm
- Encoder stack (BERT-style), decoder stack (GPT-style), masking
- Architecture comparison: encoder-only, decoder-only, encoder-decoder
- Hugging Face: AutoTokenizer, AutoModel, pipelines, model hub
- Using pretrained models for classification, generation, embeddings
- Training/inference basics; context windows; KV-cache concept
- When to use transformers vs classical NLP (Modules 23-24)

## Module Files

| File | Topic |
|------|-------|
| transformers_complete.txt | Architecture progression |
| huggingface_deep_dive.txt | Practical HF library course |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

Runnable code: `code/transformers/01_hf_pipeline.py`.

## Prerequisites

- 25_ATTENTION is mandatory
- 19_DEEP_LEARNING for training basics

## Exit Criteria

- [ ] You can label every block of a Transformer diagram
- [ ] You can run inference with 3 different HF pipelines
- [ ] You can fine-tune a small pretrained model on a classification task
