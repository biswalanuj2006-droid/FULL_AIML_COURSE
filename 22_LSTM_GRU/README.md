# Module 22: LSTM & GRU

The gated fixes for the vanishing-gradient problem — and how they became the
workhorse of sequence modeling before transformers.

## What You Will Learn

- LSTM cell state vs hidden state
- Forget, input, and output gates: equations and intuition
- Why gates control gradient flow
- GRU: update + reset gates, fewer parameters than LSTM
- LSTM vs GRU vs vanilla RNN: capacity, speed, data needs
- Stacked/bidirectional recurrent layers
- Building LSTM forecasters and text models
- Regularization: dropout on recurrent connections, embedding dropout

## Module Files

| File | Topic |
|------|-------|
| lstm_gru_complete.txt | Full theory → math → code |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

Related code: `code/rnn/01_keras_lstm_sine.py`.

## Prerequisites

- 21_RNN (recurrence and BPTT)
- 18_ANN

## Exit Criteria

- [ ] You can write the LSTM gate equations and explain each gate
- [ ] You can compare LSTM vs GRU on a real sequence task
- [ ] Your LSTM beats the naive baseline in Module 16 time-series style
