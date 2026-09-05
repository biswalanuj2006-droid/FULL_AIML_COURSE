# Module 17: Computer Vision Foundations

How images become numbers and classical techniques that still power
pipelines today — before you touch deep learning.

## What You Will Learn

- Images as arrays: pixels, channels, dtypes, resolutions
- Color spaces (RGB, grayscale, HSV) and why they matter
- Loading/resizing/interpolation with Pillow, OpenCV, torchvision
- Filtering: blur, sharpen, edge detection (Sobel/Canny intuition)
- Thresholding, morphology, contours
- Geometric transforms: rotation, crop, perspective
- Classical feature concepts (corners, descriptors) — bridge to CNN
- Augmentation for deep learning (what and why)

## Module Files

| File | Topic |
|------|-------|
| cv_foundations_complete.txt | Full theory + OpenCV/Pillow walkthrough |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

## Prerequisites

- 02_NUMPY (arrays are images)
- Basic 03_PANDAS/visualization not required

## Exit Criteria

- [ ] You can read an image and manipulate its pixels with NumPy
- [ ] You can apply edge detection and thresholding with OpenCV
- [ ] You understand why augmentations must not leak into validation
