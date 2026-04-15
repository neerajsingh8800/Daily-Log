# XGBoost (Extreme Gradient Boosting)

### What is it?
XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It implements machine learning algorithms under the Gradient Boosting framework.

### Why use it?
- **Speed:** It is much faster than standard Gradient Boosting.
- **Performance:** It has a built-in regularization which helps to prevent overfitting.
- **Handling Missing Values:** It has an internal strategy for missing values.

### Simple Implementation:
```python
import xgboost as xgb
# Load data and train
model = xgb.XGBClassifier()
# model.fit(X_train, y_train)
```
