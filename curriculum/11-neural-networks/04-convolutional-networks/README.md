---
title: Convolutional Neural Networks
order: 4
summary: Convolution as a sliding dot product, padding and stride, parameter sharing, pooling, the im2col trick, and the convolution backward pass - implemented in NumPy.
level: advanced
minutes: 60
why: Images (and audio, and time series) have local structure that is the same everywhere. Convolutions exploit this with tiny shared filters, giving models that are far more accurate and use far fewer parameters than dense networks on images.
background: Fukushima's Neocognitron (1980) and LeCun's LeNet (1989-1998, cheque reading) introduced CNNs. AlexNet (Krizhevsky, Sutskever & Hinton, 2012) won ImageNet by a huge margin and started the deep-learning era.
use_cases:
  - Image classification, object detection, segmentation
  - Medical imaging, satellite imagery
  - Audio (spectrograms) and 1-D signals
where_used:
  - 05-digit-recognizer
prerequisites:
  - 11-neural-networks/02-mlp-backpropagation
code:
  - mlscratch/neural_network.py:im2col
  - mlscratch/neural_network.py:Conv2D
  - mlscratch/neural_network.py:MaxPool2D
demo: demo.py
pros:
  - Parameter sharing → few parameters, translation equivariance
  - Learns a hierarchy of features (edges → shapes → objects)
cons:
  - Compute-heavy; pure NumPy is slow (use PyTorch for real work)
workflow: |
  flowchart LR
    A[image C×H×W] --> B[Conv: F filters k×k → F×H×W]
    B --> C[ReLU]
    C --> D[MaxPool 2×2 → F×H/2×W/2]
    D --> E[more conv blocks]
    E --> F[Flatten → Dense → softmax]
---

## Convolution
A filter (kernel) $K$ of size $k\times k$ slides over the image; at each position it computes a dot product:
$$
\text{out}[f, i, j] = \sum_{c}\sum_{u=0}^{k-1}\sum_{v=0}^{k-1} x[c,\ i\cdot s + u,\ j\cdot s + v]\;W[f, c, u, v] + b[f]
$$
(Deep-learning "convolution" is technically cross-correlation - the kernel isn't flipped.)
Output size: $\left\lfloor\frac{H + 2p - k}{s}\right\rfloor + 1$ for padding $p$ and stride $s$. "Same" padding $p = (k-1)/2$ keeps the size.

**Why it works:**
- **Local connectivity** - each output looks at a small patch (pixels are correlated locally).
- **Parameter sharing** - the same filter everywhere: an edge detector useful at the top-left is useful at the bottom-right.
  A 3×3 conv from 1 to 32 channels has $32\cdot 9 + 32 = 320$ parameters; a dense layer from a 28×28 image to 28×28×32 would have ~20 million.
- **Translation equivariance** - shift the input, the feature map shifts.

## Pooling
Max pooling takes the maximum of each 2×2 block: halves the resolution, adds a little translation invariance, no
parameters. Backward: the gradient flows only to the position that was the max.

## im2col: convolution as one matrix product
Unroll every $C\times k\times k$ patch into a row → a matrix of shape $(N\cdot H_{out}\cdot W_{out},\ Ck^2)$. Reshape the
filters to $(F, Ck^2)$. Then the whole convolution is `cols @ W.T` - a single, fast matrix multiplication. This is
exactly `Conv2D.forward`.

## Backward pass
With $g$ = upstream gradient reshaped to rows (one per output position):
- $\partial L/\partial W = g^\top\,\text{cols}$ (reshape to the filter shape),
- $\partial L/\partial b = \sum g$,
- $\partial L/\partial x = \text{col2im}(g\,W)$ - scatter-add each row back to the patch it came from.
The tests check this numerically.

## Exercises
1. Run the demo: apply hand-made edge filters to a digit and look at the feature maps.
2. Count the parameters of the demo CNN vs an MLP with similar accuracy.
3. Verify the output-size formula for k=5, s=2, p=1 on a 28×28 input.
