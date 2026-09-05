# Module 20: Convolutional Neural Networks

The architecture that made computers see — convolutions, pooling, and the
classic CNN families.

## What You Will Learn

- Why not MLPs for images: parameter explosion, locality, translation
- Convolution: kernel, stride, padding, channels, feature maps
- Output-size and parameter-count math
- Pooling, flattening, and the classifier head
- Training a CNN on real image data (PyTorch/Keras)
- Classic architectures: LeNet, AlexNet, VGG, ResNet, EfficientNet/MobileNet
- Transfer learning: pretrained backbones, fine-tuning strategies
- Data augmentation and regularization for vision
- Feature-map visualization intuition

## Module Files

| File | Topic |
|------|-------|
| cnn_complete.txt | Full theory → math → code progression |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

Runnable code: `code/cnn/01_pytorch_cnn.py` (MNIST CNN, verified).

## Prerequisites

- 18_ANN, 19_DEEP_LEARNING basics
- 17_CV_FOUNDATIONS helpful for data handling

## Exit Criteria

- [ ] You can compute an output size and parameter count by hand
- [ ] You can train a CNN and reach ~99% on MNIST
- [ ] You can fine-tune a pretrained backbone on a new dataset
