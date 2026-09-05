# Module 27: Generative AI Foundations

How generative models work — autoregression, sampling, and the concepts that
explain both LLMs and image/video generators.

## What You Will Learn

- Generative vs discriminative modeling
- Autoregressive generation: predict next token, sample, repeat
- Tokenization, context windows, and prompt framing
- Sampling: temperature, top-k, top-p, and their trade-offs
- Decoding strategies: greedy vs beam vs sampling
- Hallucination: why it happens, grounding as mitigation
- Model families: diffusion (images) and autoregressive (text) overview
- Embeddings and how models "know" relations between concepts
- Inference costs: tokens, batching, and why context length is expensive

## Module Files

| File | Topic |
|------|-------|
| genai_foundations.txt | Full conceptual + hands-on notes |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

## Prerequisites

- 26_TRANSFORMERS strongly recommended
- 29_RAG follows naturally after this

## Exit Criteria

- [ ] You can explain temperature's effect on sampling
- [ ] You can describe the full generation loop of an LLM
- [ ] You can list three causes of hallucination and one mitigation each
