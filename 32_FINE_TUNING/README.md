# Module 32: Fine-Tuning

Adapting pretrained models to your data: full fine-tuning vs parameter-
efficient methods (LoRA/QLoRA) and when to bother at all.

## What You Will Learn

- Fine-tuning vs prompting vs RAG: decision framework
- Transfer learning: why pretrained weights are a head start
- Full fine-tuning: data prep, hyperparameters, overfitting risks
- LoRA: low-rank adapters and why they work
- QLoRA: quantization + LoRA for consumer GPUs
- PEFT library workflows; Hugging Face Trainer
- Instruction tuning datasets and formatting
- Evaluating a fine-tuned model vs the base model (honestly)
- Fine-tuning for classification, chat, and tool use

## Module Files

| File | Topic |
|------|-------|
| fine_tuning_complete.txt | Full theory → hands-on code |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

## Prerequisites

- 26_TRANSFORMERS (HF ecosystem)
- 19_DEEP_LEARNING training basics

## Exit Criteria

- [ ] You can explain LoRA in one minute
- [ ] You fine-tuned a small model and measured the improvement
- [ ] You can argue when NOT to fine-tune
