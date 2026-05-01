# CatBoost (Categorical Boosting)

**CatBoost** is an open-source gradient boosting library developed by Yandex. It is designed to handle categorical variables automatically and efficiently while providing state-of-the-art results for various machine learning tasks.

---

## 1. Why CatBoost?
Traditional Gradient Boosting Decision Trees (GBDTs) like XGBoost and LightGBM require manual preprocessing of categorical data (e.g., One-Hot Encoding or Label Encoding). CatBoost introduces two major innovations:
* **Symmetric Trees:** It uses oblivious trees, which are balanced and less prone to overfitting.
* **Categorical Feature Support:** It uses **Ordered Target Statistics** to convert categorical values into numerical features without leaking information from the target variable.

---

## 2. Key Theoretical Concepts

### A. Ordered Target Statistics
To convert a categorical feature into a number, CatBoost calculates the mean of the target values for that category. To avoid **target leakage**, it uses only the data points that came *before* the current one in a random permutation.

The formula for the transformed value $\hat{x}_{i}^k$ for the $i$-th object's $k$-th categorical feature is:

$$\hat{x}_{i}^k = \frac{\sum_{j=1}^{n} [x_{j}^k = x_{i}^k] \cdot y_j + a \cdot P}{\sum_{j=1}^{n} [x_{j}^k = x_{i}^k] + a}$$

Where:
* **$y_j$**: The target value of the $j$-th object.
* **$P$**: A prior value (usually the mean target value in the entire dataset).
* **$a$**: The weight of the prior (a hyperparameter).
* **$[x_{j}^k = x_{i}^k]$**: An indicator function that is 1 if the features match, 0 otherwise.

### B. Ordered Boosting
Standard GBDT suffers from **prediction shift**. CatBoost solves this by using a technique where for each sample, a separate model is trained using only the samples that preceded it in a random permutation, ensuring the gradient used for a sample is never calculated using that sample’s own target value.

---

## 3. Advantages & Disadvantages

| Advantages | Disadvantages |
| :--- | :--- |
| Handles categorical features automatically. | Slower training time compared to LightGBM. |
| Reduces the need for extensive hyperparameter tuning. | Large model size due to symmetric tree structures. |
| Robust against overfitting (Ordered Boosting). | Memory intensive during the preprocessing phase. |

---

## 4. Implementation Code

Below is a Python implementation using the `catboost` library.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool, metrics

# 1. Load your dataset
# For demonstration, assume 'df' has categorical columns like 'City', 'Gender'
# df = pd.read_csv('your_data.csv')

# 2. Identify categorical features indices
cat_features_indices = [0, 2, 5] # Example indices of categorical columns

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize CatBoostClassifier
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    verbose=100  # Prints logs every 100 iterations
)

# 5. Fit model
model.fit(
    X_train, y_train,
    cat_features=cat_features_indices,
    eval_set=(X_test, y_test),
    plot=True # Generates a live learning curve in Jupyter
)
```

# 6. Make Predictions
preds = model.predict(X_test)
print(f"Model Accuracy: {model.score(X_test, y_test)}")
