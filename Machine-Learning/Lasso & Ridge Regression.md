# Lasso and Ridge Regression with Cross-Validation

Regularization is a technique used to prevent **overfitting** by adding a penalty term to the cost function of a model. In linear models, this is primarily done through **Lasso** and **Ridge** regression.

---
## 2. Lasso Regression (L1 Regularization)
Lasso (Least Absolute Shrinkage and Selection Operator) adds a penalty equal to the **absolute value of the magnitude** of coefficients.

### Mathematical Formula
The cost function for Lasso is:
$$Cost = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{m} |\beta_j|$$

* **Key Characteristic:** It can shrink coefficients to **exactly zero**, performing automated **feature selection**.
* **Best For:** High-dimensional datasets where you suspect only a subset of features are actually important.

### Implementation with LassoCV
`LassoCV` automatically finds the best alpha by exploring a regularization path and using cross-validation.

```python
from sklearn.linear_model import LassoCV
import numpy as np

# 1. Initialize LassoCV
# eps: Length of the path (ratio of alpha_min / alpha_max)
# n_alphas: Number of alphas along the path to explore
# cv: Number of cross-validation folds
lasso_cv = LassoCV(eps=0.001, n_alphas=100, cv=5, random_state=42)

# 2. Fit the model
# (Assumes X_train_scaled and y_train are already defined from Ridge section)
lasso_cv.fit(X_train_scaled, y_train)

# 3. Best alpha found during Cross-Validation
print(f"Optimal Alpha: {lasso_cv.alpha_}")

# 4. Feature Selection Results
# Count how many features were "zeroed out" by Lasso
coeffs = lasso_cv.coef_
useful_features = np.sum(coeffs != 0)
eliminated_features = np.sum(coeffs == 0)

print(f"Features kept: {useful_features}")
print(f"Features eliminated: {eliminated_features}")

# 5. Evaluate
print(f"Training Score: {lasso_cv.score(X_train_scaled, y_train)}")
print(f"Test Score: {lasso_cv.score(X_test_scaled, y_test)}")
```

## 1. Ridge Regression (L2 Regularization)
Ridge regression adds a penalty equal to the **square of the magnitude** of coefficients.

### Mathematical Formula
The cost function for Ridge is:
$$Cost = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{m} \beta_j^2$$

* **Key Characteristic:** It shrinks coefficients toward zero but **never** makes them exactly zero.
* **Best For:** Handling multicollinearity (when features are highly correlated).

### Implementation with RidgeCV
`RidgeCV` implements ridge regression with built-in cross-validation of the alpha parameter.

```python
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# Load Data
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Scaling is MANDATORY for regularization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define alphas to test: 10^-3 to 10^2
alphas = [0.001, 0.01, 0.1, 1, 10, 100]

# Initialize and fit RidgeCV
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train_scaled, y_train)

print(f"Optimal Alpha: {ridge_cv.alpha_}")
print(f"Training Score: {ridge_cv.score(X_train_scaled, y_train)}")
print(f"Test Score: {ridge_cv.score(X_test_scaled, y_test)}")
```
