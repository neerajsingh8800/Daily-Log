# 03. Activation Functions

Activation functions are the mathematical gates of a neural network. They determine whether a neuron should be "fired" (activated) or not, based on whether the input is relevant for the model's prediction.

---

## 1. Why do we need Activation Functions?
Without activation functions, a neural network is just a giant linear regression model ($y = wx + b$). No matter how many layers you add, a combination of linear functions is always a linear function.

**Non-linearity** allows the network to learn complex patterns (images, video, audio, and non-linear data distributions).

---

## 2. Common Activation Functions

### A. Sigmoid Function
The Sigmoid function curves into an "S" shape. It squashes any input value into a range between **0 and 1**.
* **Formula:** $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
* **Use Case:** Usually used in the **Output Layer** for binary classification.
* **Pros:** Smooth gradient, clear predictions.
* **Cons:** **Vanishing Gradient Problem** (gradients become very small for high/low inputs, stopping the network from learning).

### B. Hyperbolic Tangent (Tanh)
Tanh is similar to Sigmoid but squashes inputs to a range between **-1 and 1**.
* **Formula:** $$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
* **Use Case:** Often used in hidden layers.
* **Pros:** Zero-centered (makes the optimization process easier than Sigmoid).
* **Cons:** Also suffers from the vanishing gradient problem.

### C. ReLU (Rectified Linear Unit)
ReLU is the most popular activation function in Deep Learning today.
* **Formula:** $$f(z) = \max(0, z)$$
* **Use Case:** Default choice for **Hidden Layers**.
* **Pros:** Computationally efficient; does not saturate in the positive region (solves vanishing gradient for $z > 0$).
* **Cons:** **Dying ReLU problem** (neurons can "die" and stay at 0 if they receive large negative gradients).

### D. Leaky ReLU
A variation of ReLU that prevents the "Dying ReLU" problem by allowing a small, non-zero gradient when the input is negative.
* **Formula:** $$f(z) = \max(\alpha z, z)$$ (where $\alpha$ is a small constant like 0.01).

### E. Softmax
* **Formula:** $$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$
* **Use Case:** **Output Layer** for multi-class classification (probabilities sum up to 1).

---

## 3. Comparison Table

| Activation | Range | Best For | Main Issue |
| :--- | :--- | :--- | :--- |
| **Sigmoid** | (0, 1) | Binary Output | Vanishing Gradient |
| **Tanh** | (-1, 1) | Hidden Layers | Vanishing Gradient |
| **ReLU** | [0, $\infty$) | Hidden Layers | Dying ReLU |
| **Leaky ReLU** | (-$\infty$, $\infty$) | Hidden Layers | Determining $\alpha$ |
| **Softmax** | (0, 1) | Multi-class Output | Computationally expensive |

---

## 4. Implementation in Python (NumPy)

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, z * alpha)

def softmax(z):
    exp_z = np.exp(z - np.max(z)) # Subtracting max(z) for numerical stability
    return exp_z / exp_z.sum(axis=0)

# Testing the functions
test_input = np.array([-2, -1, 0, 1, 2])

print(f"Input: {test_input}")
print(f"ReLU: {relu(test_input)}")
print(f"Sigmoid: {sigmoid(test_input)}")
print(f"Leaky ReLU: {leaky_relu(test_input)}")
```
