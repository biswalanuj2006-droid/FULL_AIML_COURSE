# Module 21: Recurrent Neural Networks

Networks for sequences: how recurrence works, how training happens through
time, and why plain RNNs struggle on long sequences.

## What You Will Learn

- Sequence data: text, audio, sensor streams, time series
- Recurrence and the hidden state; unrolling in time
- RNN forward pass and shared weights
- Backpropagation Through Time (BPTT)
- Vanishing/exploding gradients — the core RNN failure
- Activation and initialization choices for RNNs
- Sequence tasks: many-to-one, one-to-many, many-to-many
- Implementing and training RNNs (Keras/PyTorch)
- When RNNs still make sense vs transformers (see Module 25/26)

## Module Files

| File | Topic |
|------|-------|
| rnn_complete.txt | Full theory → math → code |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

Related code: `code/rnn/01_keras_lstm_sine.py` (windowing + LSTM, verified).

## Prerequisites

- 18_ANN, 19_DEEP_LEARNING
- 56_TIME_SERIES helpful for data intuition

## Exit Criteria

- [ ] You can draw an unrolled RNN and label all weight matrices
- [ ] You can explain BPTT and vanishing gradients
- [ ] You built one sequence model end to end
