# 04. Loss Functions (Cost Functions)

A Loss Function measures how far the network's prediction is from the actual target value. In Deep Learning, the goal of optimization is to minimize this value using Gradient Descent.

---

## 1. Regression vs. Classification
Loss functions are generally divided into two categories based on the type of problem you are solving:
* **Regression:** Predicting continuous values (e.g., house prices).
* **Classification:** Predicting discrete labels (e.g., Cat vs. Dog).

---

## 2. Regression Loss Functions

### A. Mean Squared Error (MSE / L2 Loss)
The most common loss for regression. It squares the difference between predicted and actual values, which penalizes large errors heavily.
* **Formula:** $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
* **Pros:** Easy to compute gradients (differentiable).
* **Cons:** Very sensitive to outliers.

### B. Mean Absolute Error (MAE / L1 Loss)
It takes the absolute difference. It is more "robust" to outliers than MSE.
* **Formula:** $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
* **Pros:** Not overly influenced by outliers.
* **Cons:** The gradient is constant, which can make it harder for the model to converge precisely.

---

## 3. Classification Loss Functions

### A. Binary Cross-Entropy (Log Loss)
The standard loss for binary classification (0 or 1). It measures the performance of a model where the output is a probability between 0 and 1.
* **Formula:** $$L = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$
* **Use Case:** Predicting a single class (Yes/No).

### B. Categorical Cross-Entropy
Used for multi-class classification problems.
* **Formula:** $$L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$
* **Requirement:** Usually requires a **Softmax** activation in the final layer.

---

## 4. Summary Table

| Loss Function | Problem Type | Output Activation | Robust to Outliers? |
| :--- | :--- | :--- | :--- |
| **MSE** | Regression | Linear / None | No |
| **MAE** | Regression | Linear / None | Yes |
| **Binary Cross-Entropy** | Binary Classification | Sigmoid | N/A |
| **Categorical Cross-Entropy** | Multi-class Classification | Softmax | N/A |

---

## 5. Implementation in Python (NumPy)

```python
import numpy as np

# 1. Mean Squared Error
def mean_squared_error(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

# 2. Mean Absolute Error
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# 3. Binary Cross-Entropy
def binary_cross_entropy(y_true, y_pred):
    # Clip values to prevent log(0) which is undefined
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example Usage
y_actual = np.array([1, 0, 1, 1])
y_predicted = np.array([0.9, 0.1, 0.8, 0.4])

print(f"MSE: {mean_squared_error(y_actual, y_predicted):.4f}")
print(f"BCE Loss: {binary_cross_entropy(y_actual, y_predicted):.4f}")
```
