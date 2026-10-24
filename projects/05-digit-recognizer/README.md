---
title: "Handwritten Digit Recognizer"
level: Advanced
summary: "MNIST from scratch: PCA, classical baselines (naive Bayes, softmax, random forest, kNN), then a multi-layer perceptron and a convolutional network trained with hand-written backpropagation and Adam."
problem: |
  Recognise handwritten digits (0-9) from 28×28 grayscale images: the problem that made neural networks famous (LeNet,
  1998). It is **10-class image classification** with 70,000 images. The goal is not just accuracy but understanding *why*
  each model family lands where it does, and what convolution buys over a plain network.
dataset:
  name: "MNIST"
  source: "LeCun, Cortes & Burges · original IDX files (mirror used by torchvision)"
  url: "https://ossci-datasets.s3.amazonaws.com/mnist/"
  kaggle: "competitions/digit-recognizer"
  license: "CC BY-SA 3.0"
  size: "60,000 train + 10,000 test images · 11 MB compressed"
algorithms: [PCA, Gaussian Naive Bayes, Softmax Regression, Random Forest, kNN, MLP (backprop + Adam + dropout), CNN (im2col convolution + max-pooling)]
skills:
  - Working with image tensors (n, channels, height, width)
  - PCA for compression and visualisation
  - "A fair protocol: train / validation / official test set"
  - Mini-batch training, Adam, dropout, learning curves
  - Writing a training loop by hand (forward → loss → backward → step)
  - Keeping the best validation weights (early stopping)
  - Confusion analysis and looking at confident mistakes
lessons:
  - 00-math-foundations/01-linear-algebra
  - 00-math-foundations/02-calculus-gradients
  - 05-feature-selection-extraction/03-pca
  - 08-classification/02-softmax-regression
  - 08-classification/03-k-nearest-neighbors
  - 11-neural-networks/02-mlp-backpropagation
  - 11-neural-networks/03-training-deep-networks
  - 11-neural-networks/04-convolutional-networks
steps:
  - title: "1 · Download"
    script: download_data.py
    description: |
      Parses the original **IDX binary format** (a 4-byte magic number, big-endian int32 dimensions, then raw bytes).
      Falls back to Keras's `mnist.npz` or the Kaggle competition. Everything is saved as one `data/mnist.npz`.
  - title: "2 · Explore and compress"
    script: explore.py
    description: |
      Classes are balanced, and 9% of the pixels (the border) are always 0. PCA needs only **43 of 784 dimensions for
      80% of the variance** and 150 for 95%. Reconstructions with k = 5…150 show what each amount of compression keeps.
  - title: "3 · Classical baselines"
    script: baselines.py
    description: |
      Gaussian NB, softmax regression, a random forest and kNN (on PCA-50 features). Every network has to beat these.
      kNN with k = 3 already reaches 97.5%. Takes about 1½ minutes.
  - title: "4 · Multi-layer perceptron"
    script: train_mlp.py
    description: |
      784-256-128-10 with ReLU, softmax cross-entropy and Adam, trained for 15 epochs, with and without dropout. The
      script plots the learning curves, the confusion matrix, the most confident mistakes and what the first layer
      learned. Takes about 1½ minutes.
  - title: "5 · Convolutional network"
    script: train_cnn.py
    description: |
      Two conv + max-pool blocks and a small dense head, trained with a **hand-written loop** for 4 epochs (≈ 2 minutes).
      For a quick try: `python train_cnn.py --epochs 1 --n-train 5000`.
workflow: |
  flowchart LR
    A[IDX files] --> B[Scale to 0-1 · 55k/5k/10k split]
    B --> C[PCA: 43 dims = 80%]
    C --> D[Baselines: NB · softmax · RF · kNN]
    B --> E[MLP 784-256-128-10<br/>Adam · dropout]
    B --> F[CNN conv-pool-conv-pool-dense]
    D & E & F --> G[Official 10k test set]
    G --> H[Confusions · confident mistakes · filters]
---

# Handwritten Digit Recognizer: from pixels to convolutions

## Results on the official 10,000-image test set

| Model | Test accuracy | Parameters / memory | Train time |
|---|---|---|---|
| Gaussian naive Bayes (PCA-50) | 87.9% | 1k numbers | < 1 s |
| Softmax regression (raw pixels, 15k images) | 90.9% | 7,850 weights | 5 s |
| Random forest, 30 trees (PCA-50, 15k images) | 92.1% | ~30 trees | 37 s |
| kNN k = 3 (PCA-50) | 97.5% | **all 55,000 images** | 0 s train / 19 s predict |
| MLP 784-256-128-10, no regularisation | 97.7% | 235,146 | 39 s |
| MLP 784-256-128-10, dropout 0.2 | 98.2% | 235,146 | 40 s |
| **CNN (2 conv blocks + dense 64)** | **98.4%** | **52,138** | ~2 min |

## What to notice
- **Linear models plateau around 91%.** A digit isn't a linear function of its pixels.
- **kNN is strong but expensive.** It must store and scan the whole training set for every prediction.
- **Dropout** adds about 0.4 points on test here; the train-validation gap shrinks only slightly, so with 55k images overfitting is mild.
- **The CNN beats the MLP with 4.5× fewer parameters.** Its conv layers hold just 1,248 weights, because each 3×3
  filter is reused at every position. That built-in assumption (local patterns that can appear anywhere) is the
  *inductive bias* that makes convolutions the right tool for images.
- **The remaining mistakes are hard.** The 3↔5, 4↔9 and 7↔9 confusions and the "confident mistakes" grid are often
  ambiguous even for people.

## Extend it
1. Data augmentation: shift each training image by ±2 pixels. CNNs typically pass 99% with this.
2. Add `BatchNorm` after each conv block and compare the learning curves.
3. Export the CNN weights and write the forward pass in C++ (see `cpp/include/mlcpp/neural_network.hpp` for the pattern).
