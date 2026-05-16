# 11. Pooling Layers and Feature Maps

After a convolution operation extracts features, **Pooling layers** are used to reduce the spatial dimensions (Width x Height) of the input volume for the next convolutional layer.

---

## 1. The Purpose of Pooling
Pooling serves three primary functions in a CNN:
1.  **Dimensionality Reduction:** It reduces the number of parameters and computation in the network, helping to control overfitting.
2.  **Invariance:** It makes the network invariant to small translations, distortions, and variations in the input image.
3.  **Feature Condensing:** It summarizes the presence of features in patches of the feature map.

---

## 2. Types of Pooling

### A. Max Pooling
The most common type. It slides a window over the input and takes the **maximum value** from the region.
* **Logic:** "Which part of this region has the strongest presence of the feature we are looking for?"
* **Effect:** Preserves the most prominent features (like edges).

### B. Average Pooling
It calculates the **average value** for each patch on the feature map.
* **Logic:** "What is the general presence of this feature in this region?"
* **Effect:** Provides a smoother downsampling; often used in specific architectures like SqueezeNet or at the very end of a network.

---

## 3. Calculating Output Dimensions
The formula for pooling is the same as convolution, but usually, pooling has **no padding** ($P=0$).

$$O = \frac{W - K}{S} + 1$$

Where:
* $W$: Input size
* $K$: Filter (Pool) size
* $S$: Stride (Usually $S = K$ to ensure non-overlapping regions)

---

## 4. Feature Maps
A **Feature Map** is the output of one filter applied to the previous layer. 
* Early layers produce feature maps that look like **edges and blobs**.
* Deeper layers produce feature maps that represent **complex shapes** (eyes, wheels, faces).

---

## 5. Implementation in Python (NumPy)

This implementation demonstrates both Max and Average pooling logic.

```python
import numpy as np

def pooling2d(image, pool_size=2, stride=2, mode='max'):
    img_h, img_w = image.shape
    
    # Calculate output dimensions
    out_h = (img_h - pool_size) // stride + 1
    out_w = (img_w - pool_size) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    for i in range(0, out_h):
        for j in range(0, out_w):
            # Define the current window
            curr_y = i * stride
            curr_x = j * stride
            region = image[curr_y:curr_y+pool_size, curr_x:curr_x+pool_size]
            
            # Apply pooling logic
            if mode == 'max':
                output[i, j] = np.max(region)
            elif mode == 'average':
                output[i, j] = np.mean(region)
                
    return output

# Example: 4x4 Input Matrix
input_map = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

max_pooled = pooling2d(input_map, pool_size=2, stride=2, mode='max')
avg_pooled = pooling2d(input_map, pool_size=2, stride=2, mode='average')

print("Max Pooled (2x2):\n", max_pooled)
print("Average Pooled (2x2):\n", avg_pooled)
```
