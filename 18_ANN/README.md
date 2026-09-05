# Module 18: Artificial Neural Networks

The neuron, the layer, the forward pass — built from zero with math, then
code. Everything in deep learning builds on this module.

## What You Will Learn

- Perceptron → neuron: weighted sum + bias + activation
- Activation functions: sigmoid, tanh, ReLU family, softmax
- Layers, hidden units, and what depth buys you
- Forward propagation as matrix operations
- Loss functions and why they are chosen per task
- Backpropagation: chain rule, gradients layer by layer
- Parameter updates and the training loop
- Initialization, learning rate, and first debugging instincts
- NumPy MLP that actually learns (bridge to PyTorch in Module 19)

## Module Files

| File | Topic |
|------|-------|
| ann_complete.txt | Full theory → math → code progression |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

Related code: `code/06_neural_network_from_scratch.py`,
`code/ml/softmax_cross_entropy_from_scratch.py`,
`code/ml/gradient_descent_from_scratch.py`.

## Prerequisites

- 05_MATHEMATICS calculus (chain rule, gradients)
- 02_NUMPY matrix fluency

## Exit Criteria

- [ ] You can write the forward pass of an MLP by hand
- [ ] You can derive the gradient of one weight in a 2-layer net
- [ ] Your NumPy net beats chance on a real dataset
