# Module 25: Attention

The mechanism behind every modern LLM — taught from zero with a tiny
numerical example.

## What You Will Learn

- Why sequence models need a way to focus: the RNN bottleneck
- Query, Key, Value: where the names come from (retrieval analogy)
- Attention scores, scaling by sqrt(d_k), softmax, weighted sum
- Self-attention vs cross-attention; masks for decoding
- The full attention formula decoded symbol by symbol
- A hand-worked tiny example (the one in the notes)
- From attention to multi-head (bridge to transformers)

## Module Files

| File | Topic |
|------|-------|
| attention_complete.txt | Full theory → math → worked example |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

## Prerequisites

- 18_ANN; 05_MATHEMATICS (dot products, softmax)
- 23/24 NLP background helpful but not required

## Exit Criteria

- [ ] You can explain Q/K/V in one minute with an analogy
- [ ] You can compute a 2-token attention output by hand
- [ ] You can implement attention in NumPy
