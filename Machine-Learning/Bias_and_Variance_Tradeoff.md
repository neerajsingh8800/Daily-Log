# Bias-Variance Tradeoff

The **Bias-Variance Tradeoff** is a central problem in supervised learning. Ideally, we want a model that accurately captures the regularities in its training data, but also generalizes well to unseen data. Unfortunately, it is typically impossible to do both simultaneously.

---

## 1. Core Definitions

### **Bias**
Bias is the difference between the average prediction of our model and the correct value which we are trying to predict. 
* **High Bias:** The model pays very little attention to the training data and oversimplifies the model (**Underfitting**).
* **Example:** Using Linear Regression for highly non-linear data.

### **Variance**
Variance is the variability of model prediction for a given data point or a value which tells us spread of our data.
* **High Variance:** The model pays a lot of attention to training data and does not generalize on the data which it hasn’t seen before (**Overfitting**).
* **Example:** A Decision Tree that is allowed to grow deep without pruning.

---

## 2. Mathematical Formulation

If we let $Y$ be the variable we are predicting and $X$ be our predictor, we assume there is a relationship $Y = f(X) + \epsilon$, where the error term $\epsilon$ is normally distributed with a mean of zero.

The **Total Expected Error** can be decomposed as:

$$Error = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

Where:
* $\text{Bias} = E[\hat{f}(x)] - f(x)$
* $\text{Variance} = E[\hat{f}(x)^2] - E[\hat{f}(x)]^2$
* **Irreducible Error** is the noise that cannot be reduced by any model.

---

## 3. The Tradeoff

* **Low Bias, High Variance:** Model is complex and overfits. (e.g., KNN with $k=1$)
* **High Bias, Low Variance:** Model is simple and underfits. (e.g., Linear Regression on non-linear data)
* **The Goal:** Find the "Sweet Spot" where the total error is minimized.

---

## 4. Implementation Example (Python)

This script demonstrates how increasing model complexity (Polynomial degree) affects the Bias and Variance.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# 1. Generate Synthetic Data
def true_fun(X):
    return np.cos(1.5 * np.pi * X)

np.random.seed(0)
n_samples = 30
X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.1

# 2. Compare Degrees (Complexity)
degrees = [1, 4, 15] # 1=High Bias, 15=High Variance

plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    
    polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("pf", polynomial_features), ("lr", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

    # Evaluate using Cross Validation
    scores = cross_val_score(pipeline, X[:, np.newaxis], y, scoring="neg_mean_squared_error", cv=10)

    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    plt.plot(X_test, true_fun(X_test), label="True function")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    plt.title(f"Degree {degrees[i]}\nMSE = {-scores.mean():.2e}")
    plt.legend(loc="best")

plt.show()
```
