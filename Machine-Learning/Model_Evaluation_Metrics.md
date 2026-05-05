# Model Evaluation Metrics

In Machine Learning, evaluation metrics are the "yardsticks" used to measure the performance of a model. Choosing the right metric is critical because it determines how you optimize your model and how you interpret its success.

---

## 1. Classification Metrics

Classification metrics are used when the output is a category (e.g., Spam or Not Spam).

### A. Confusion Matrix
A table used to describe the performance of a classification model.
* **TP (True Positive):** Predicted Yes, Actual Yes.
* **TN (True Negative):** Predicted No, Actual No.
* **FP (False Positive):** Predicted Yes, Actual No (Type I Error).
* **FN (False Negative):** Predicted No, Actual Yes (Type II Error).

### B. Accuracy
The ratio of correctly predicted observations to the total observations.
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

### C. Precision
The ratio of correctly predicted positive observations to the total predicted positives. Use this when the cost of a **False Positive** is high (e.g., Spam detection).
$$Precision = \frac{TP}{TP + FP}$$

### D. Recall (Sensitivity)
The ratio of correctly predicted positive observations to all observations in the actual class. Use this when the cost of a **False Negative** is high (e.g., Cancer detection).
$$Recall = \frac{TP}{TP + FN}$$

### E. F1-Score
The harmonic mean of Precision and Recall. It is useful when you have an imbalanced dataset.
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

---

## 2. Regression Metrics

Regression metrics are used when the output is a continuous value (e.g., House Prices).

### A. Mean Absolute Error (MAE)
The average of the absolute differences between predictions and actual values. It is robust to outliers.
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### B. Mean Squared Error (MSE)
The average of the squared differences. It penalizes larger errors more heavily.
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### C. R-Squared ($R^2$)
Represents the proportion of variance for a dependent variable that's explained by an independent variable.
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

---

## 3. Implementation Code

```python
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Classification Example ---
y_true_cls = [0, 1, 1, 0, 1, 1]
y_pred_cls = [0, 1, 0, 0, 1, 1]

print("Classification Metrics:")
print(f"Accuracy:  {accuracy_score(y_true_cls, y_pred_cls):.2f}")
print(f"Precision: {precision_score(y_true_cls, y_pred_cls):.2f}")
print(f"Recall:    {recall_score(y_true_cls, y_pred_cls):.2f}")
print(f"F1-Score:  {f1_score(y_true_cls, y_pred_cls):.2f}")

print("\n" + "-"*30 + "\n")

# --- Regression Example ---
y_true_reg = [3.0, -0.5, 2.0, 7.0]
y_pred_reg = [2.5, 0.0, 2.1, 7.8]

print("Regression Metrics:")
print(f"MAE: {mean_absolute_error(y_true_reg, y_pred_reg):.2f}")
print(f"MSE: {mean_squared_error(y_true_reg, y_pred_reg):.2f}")
print(f"R2 : {r2_score(y_true_reg, y_pred_reg):.2f}")
```
