# Linear Regression

Linear Regression is a fundamental supervised learning algorithm used to predict a continuous numerical value (Target) based on one or more input features (Predictors). It assumes a linear relationship between the input variables ($X$) and the single output variable ($y$).

---

### 1. The Mathematical Model
The goal is to find the "Line of Best Fit" represented by the equation:

$$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon$$

**Where:**
* $y$: Predicted value (Dependent variable)
* $\beta_0$: Intercept (Bias)
* $\beta_1, \beta_n$: Coefficients (Weights)
* $x_n$: Features (Independent variables)
* $\epsilon$: Error term (Residuals)

---

### 2. Key Assumptions
To get the most accurate results, Linear Regression typically assumes:
1. **Linearity:** The relationship between $X$ and $y$ is linear.
2. **Independence:** Observations are independent of each other.
3. **Homoscedasticity:** The variance of residual errors is constant across all levels of the independent variables.
4. **Normality:** The residuals should be normally distributed.

---

### 3. Python Implementation
This implementation uses `scikit-learn` to create a simple model, train it on synthetic data, and evaluate its performance.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate Synthetic Data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 2. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make Predictions
y_pred = model.predict(X_test)

# 5. Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"Coefficient: {model.coef_[0][0]:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# 6. Visualize Results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Feature X')
plt.ylabel('Target y')
plt.title('Linear Regression Model Fit')
plt.legend()
plt.show()
```
