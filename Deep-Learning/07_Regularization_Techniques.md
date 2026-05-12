# 07. Regularization Techniques

Regularization is a set of techniques used to prevent **Overfitting**. Overfitting occurs when a model learns the noise and details in the training data to the extent that it negatively impacts the performance of the model on new data.

---

## 1. The Bias-Variance Tradeoff
* **High Bias:** The model is too simple (Underfitting).
* **High Variance:** The model is too complex and sensitive to training data noise (Overfitting).
* **Goal:** Find the "Sweet Spot" where the total error is minimized.

---

## 2. L1 and L2 Regularization (Weight Decay)
These techniques add a penalty term to the Loss Function based on the magnitude of the weights.

### A. L2 Regularization (Ridge)
Adds the sum of the squared weights to the loss. It forces weights to be small but not necessarily zero.
$$L_{total} = L_{original} + \lambda \sum w^2$$
* **Effect:** Prevents any single feature from having a disproportionately large influence.

### B. L1 Regularization (Lasso)
Adds the sum of the absolute values of the weights to the loss.
$$L_{total} = L_{original} + \lambda \sum |w|$$
* **Effect:** Leads to **Sparsity** (some weights become exactly zero), effectively acting as a built-in feature selector.

---

## 3. Dropout
Dropout is a computationally cheap but powerful regularization technique. 
* **How it works:** During training, a specified percentage (e.g., 50%) of neurons are randomly "dropped" (set to zero) in each forward pass.
* **Why it works:** It prevents neurons from co-adapting (relying too much on each other) and forces the network to learn more robust, redundant representations.

> **Note:** Dropout is only active during **Training**. During **Inference (Testing)**, all neurons are used, but their outputs are scaled.

---

## 4. Early Stopping
A form of regularization where you monitor the model's performance on a validation set and stop the training process once the validation loss starts to increase, even if the training loss is still decreasing.

---

## 5. Implementation in Python (NumPy)

This implementation shows how to add an L2 penalty to the loss and how to apply a Dropout mask.

```python
import numpy as np

def l2_regularization_loss(weights, lambda_reg):
    return (lambda_reg / 2) * np.sum(np.square(weights))

def apply_dropout(layer_output, dropout_rate):
    """
    Inverted Dropout implementation.
    """
    # Create a mask of zeros and ones
    mask = (np.random.rand(*layer_output.shape) > dropout_rate)
    
    # Scale the remaining neurons to maintain the expected sum of activations
    # This avoids having to change anything during test time.
    scale = 1.0 / (1.0 - dropout_rate)
    
    return layer_output * mask * scale

# Example Usage
h1 = np.array([[0.5, 0.8, -0.2, 0.1]]) # Hidden layer output
dropout_rate = 0.5

h1_dropped = apply_dropout(h1, dropout_rate)
print(f"Original Layer Output: {h1}")
print(f"Output after Dropout: {h1_dropped}")
```
