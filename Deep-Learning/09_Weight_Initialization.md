# 09. Weight Initialization

Weight initialization is the process of setting the initial values for the parameters (weights) of a neural network. Proper initialization is critical for ensuring that gradients flow correctly through the layers during the first few passes of training.

---

## 1. Why is Initialization Important?
If weights are initialized poorly, two major problems occur:
1.  **Vanishing Gradients:** If weights are too small, the signal shrinks as it passes through layers, eventually becoming zero.
2.  **Exploding Gradients:** If weights are too large, the signal grows exponentially, leading to numerical overflow (NaN values).

> **Rule of Thumb:** We want the variance of the activations to be the same across all layers.

---

## 2. Common Initialization Techniques

### A. Zero/Constant Initialization
* **The Problem:** If all weights are zero or the same constant, every neuron in a hidden layer will perform the exact same calculation. This is called **Symmetry**, and the network will never learn different features.
* **Use Case:** Only used for **Biases**.

### B. Xavier (Glorot) Initialization
Designed for layers using **Sigmoid** or **Tanh** activation functions. It keeps the variance of the input and output gradients the same.
* **Formula (Uniform):** $W \sim U\left(-\frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}\right)$
* **Formula (Normal):** $W \sim N\left(0, \frac{2}{n_{in} + n_{out}}\right)$

### C. He (Kaiming) Initialization
The current standard for **ReLU** and its variants. Since ReLU kills half of the input (negative values), we need a larger variance to keep the signal alive.
* **Formula (Normal):** $W \sim N\left(0, \frac{2}{n_{in}}\right)$

---

## 3. Summary Table

| Method | Activation Function | Variance ($\sigma^2$) |
| :--- | :--- | :--- |
| **Xavier (Glorot)** | Sigmoid / Tanh | $\frac{1}{n_{in}}$ or $\frac{2}{n_{in} + n_{out}}$ |
| **He (Kaiming)** | ReLU / Leaky ReLU | $\frac{2}{n_{in}}$ |
| **Orthogonal** | Any (Deep Networks) | Preserves Eigenvalues |

---

## 4. Implementation in Python (NumPy)

This script demonstrates how to implement Xavier and He initialization from scratch.

```python
import numpy as np

def xavier_init(n_in, n_out):
    """Xavier initialization for Sigmoid/Tanh."""
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, size=(n_in, n_out))

def he_init(n_in, n_out):
    """He initialization for ReLU."""
    std = np.sqrt(2 / n_in)
    return np.random.normal(0, std, size=(n_in, n_out))

# Example Usage
input_nodes = 784  # e.g., MNIST image pixels
hidden_nodes = 256

# Weights for a Sigmoid layer
W_sigmoid = xavier_init(input_nodes, hidden_nodes)

# Weights for a ReLU layer
W_relu = he_init(input_nodes, hidden_nodes)

print(f"Xavier Sample Variance: {np.var(W_sigmoid):.6f}")
print(f"He Sample Variance: {np.var(W_relu):.6f}")
```
