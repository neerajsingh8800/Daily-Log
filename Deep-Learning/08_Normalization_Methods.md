# 08. Normalization Methods

Normalization techniques are essential for training deep neural networks efficiently. They help stabilize the learning process, allow for higher learning rates, and act as a form of regularization.

---

## 1. Why Normalization?
In deep networks, the distribution of each layer's inputs changes during training as the parameters of the previous layers change. This is known as **Internal Covariate Shift**. Normalization ensures that:
* Gradients don't vanish or explode.
* Training is less sensitive to weight initialization.
* The optimization landscape becomes smoother.

---

## 2. Batch Normalization (BN)
Batch Norm normalizes the activations of a layer across the **mini-batch** for each feature independently.

### The Algorithm
For a mini-batch $\mathcal{B} = \{x_1, \dots, x_m\}$, we compute:

1.  **Batch Mean:** $\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i$
2.  **Batch Variance:** $\sigma^2_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2$
3.  **Normalize:** $\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}}$
4.  **Scale and Shift:** $y_i = \gamma \hat{x}_i + \beta$

Where $\gamma$ (scale) and $\beta$ (shift) are **learnable parameters** that allow the network to undo the normalization if that's what's best for the model.

---

## 3. Layer Normalization (LN)
Unlike Batch Norm, Layer Norm normalizes across the **features** for each training example independently.

* **Batch Norm:** Dependent on batch size (problematic for small batches).
* **Layer Norm:** Independent of batch size. Widely used in **Recurrent Neural Networks (RNNs)** and **Transformers**.

---

## 4. Summary Comparison

| Feature | Batch Normalization | Layer Normalization |
| :--- | :--- | :--- |
| **Normalization Axis** | Across the batch (N) | Across the features (C, H, W) |
| **Batch Size Dependency** | High | None |
| **Best For** | CNNs, Computer Vision | RNNs, Transformers, NLP |
| **Test Time** | Uses running mean/variance | Same as training time |

---

## 5. Implementation in Python (NumPy)

This demonstrates the manual calculation of a Batch Normalization forward pass.

```python
import numpy as np

def batch_norm_forward(X, gamma, beta, eps=1e-5):
    # X shape: (N, D) where N is batch size
    
    # 1. Mean of the batch
    mu = np.mean(X, axis=0)
    
    # 2. Variance of the batch
    var = np.var(X, axis=0)
    
    # 3. Normalize
    X_hat = (X - mu) / np.sqrt(var + eps)
    
    # 4. Scale and shift
    out = gamma * X_hat + beta
    
    return out, X_hat, mu, var

# Example Usage
N, D = 3, 2  # Batch of 3 samples, 2 features each
X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# Learnable parameters (initialized to 1s and 0s)
gamma = np.ones(D)
beta = np.zeros(D)

output, _, _, _ = batch_norm_forward(X, gamma, beta)
print("Normalized Output:\n", output)
```
