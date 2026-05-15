# 10. CNN Fundamentals

Convolutional Neural Networks (CNNs) are specialized neural networks for processing data with a grid-like topology, such as images. Unlike standard MLPs, CNNs use **Parameter Sharing** and **Local Connectivity** to handle high-dimensional data efficiently.

---

## 1. Core Components of a CNN

### A. The Convolution Operation
The heart of a CNN. A small matrix (Filter/Kernel) slides over the input image to perform element-wise multiplication and summation. This extracts features like edges, textures, and patterns.

**Mathematical Formula:**
For an input $I$ and a kernel $K$, the output feature map $S$ at position $(i, j)$ is:
$$S(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(i+m, j+n)K(m, n)$$

### B. Padding
Padding adds extra pixels (usually zeros) around the border of the input.
* **Valid Padding:** No padding. The output is smaller than the input.
* **Same Padding:** Adds padding so the output size matches the input size.

### C. Stride
Stride is the number of pixels the filter shifts over the input matrix. 
* Stride = 1: Move one pixel at a time.
* Stride = 2: Move two pixels (effectively downsampling the image).

---

## 2. Calculating Output Dimensions
To determine the size of the feature map after a convolution layer, use the formula:

$$O = \frac{W - K + 2P}{S} + 1$$

Where:
* $W$: Input size (Width/Height)
* $K$: Filter (Kernel) size
* $P$: Padding
* $S$: Stride

---

## 3. Why CNNs over MLPs?
1.  **Translation Invariance:** A CNN can recognize an object regardless of where it appears in the frame.
2.  **Reduced Parameters:** In an MLP, every pixel is a weight. In a CNN, the same small filter is reused across the entire image, drastically reducing the number of learnable parameters.

---

## 4. Implementation in Python (NumPy)

This script demonstrates a basic 2D Convolution operation (Forward Pass) to show how filters detect features.

```python
import numpy as np

def convolve2d(image, kernel, stride=1, padding=0):
    # Add padding to the input image
    if padding > 0:
        image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)
    
    img_h, img_w = image.shape
    ker_h, ker_w = kernel.shape
    
    # Calculate output dimensions
    out_h = (img_h - ker_h) // stride + 1
    out_w = (img_w - ker_w) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    # Perform convolution
    for i in range(0, out_h):
        for j in range(0, out_w):
            # Extract the current region of interest
            region = image[i*stride : i*stride+ker_h, j*stride : j*stride+ker_w]
            # Element-wise multiplication and sum
            output[i, j] = np.sum(region * kernel)
            
    return output

# Example: Simple 5x5 Image and a 3x3 Vertical Edge Filter
image = np.array([
    [10, 10, 10, 0, 0],
    [10, 10, 10, 0, 0],
    [10, 10, 10, 0, 0],
    [10, 10, 10, 0, 0],
    [10, 10, 10, 0, 0]
])

v_kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

result = convolve2d(image, v_kernel)
print("Detected Vertical Edges:\n", result)
```
