# Module 19: Deep Learning

Real deep learning with PyTorch and TensorFlow/Keras: training at scale,
regularization, and the engineering habits that make nets work.

## What You Will Learn

- PyTorch: tensors, autograd, Dataset/DataLoader, nn.Module, training loops
- Keras: Sequential/Functional API, compile/fit, callbacks
- Optimizers: SGD, Momentum, RMSProp, Adam — and when each is used
- Regularization: dropout, weight decay, batch norm, early stopping
- Learning-rate schedules, warmup, and why LR is the key dial
- Initialization and gradient problems (vanishing/exploding)
- Overfitting diagnosis: train vs val curves, data augmentation
- Checkpointing, saving/loading, seeding, reproducible experiments
- GPU usage basics: .to(device), batches, memory hygiene

## Module Files

| File | Topic |
|------|-------|
| pytorch_deep_dive.txt | PyTorch-focused progression |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

Runnable code: `code/dl/01_keras_mlp_mnist.py`,
`code/cnn/01_pytorch_cnn.py` (both verified working).

## Prerequisites

- 18_ANN (concepts), 02_NUMPY, 05_MATHEMATICS

## Exit Criteria

- [ ] You can train a model in both PyTorch and Keras unaided
- [ ] You can diagnose overfitting from loss curves
- [ ] You can checkpoint, reload, and run inference from a saved model
