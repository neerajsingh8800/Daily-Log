# Hyperparameter Tuning in Machine Learning

Hyperparameter tuning (or optimization) is the process of finding the configuration of hyperparameters that results in the best performance on a validation dataset. Unlike model parameters, hyperparameters are not learned from the data; they are set before the training process begins.

---

## 1. Parameters vs. Hyperparameters

| Feature | Model Parameters | Hyperparameters |
| :--- | :--- | :--- |
| **Definition** | Internal to the model, learned from data. | External to the model, set by the user. |
| **Examples** | Weights ($w$) and Bias ($b$) in Linear Regression. | Learning rate, number of hidden layers, $k$ in KNN. |
| **Goal** | Minimize the loss function. | Optimize model architecture and generalization. |

---

## 2. Core Search Strategies

### A. Grid Search
Grid Search performs an exhaustive search over a specified subset of the hyperparameter space. It builds a model for every single combination of the provided parameters.

* **Pros:** Guaranteed to find the best combination within the grid.
* **Cons:** Computationally expensive; suffers from the "curse of dimensionality."

### B. Random Search
Random Search samples combinations from a statistical distribution. Instead of trying every value, it tries a fixed number of iterations.

* **Pros:** Often finds a "good enough" solution much faster than Grid Search.
* **Cons:** Not guaranteed to find the absolute global optimum.

### C. Bayesian Optimization
A more advanced technique that uses the results of previous iterations to pick the next set of hyperparameters. It builds a probabilistic model (usually a Gaussian Process) to predict which hyperparameters might yield better results.

---

## 3. The Mathematics of Evaluation

To avoid overfitting during tuning, we use **$k$-Fold Cross-Validation**. The data is split into $k$ subsets; the model trains on $k-1$ and validates on the remaining one.

The average performance is calculated as:

$$Accuracy_{avg} = \frac{1}{k} \sum_{i=1}^{k} Accuracy_i$$

Where:
* $k$ is the number of folds.
* $Accuracy_i$ is the score of the model on the $i$-th fold.

---

## 4. Implementation (Scikit-Learn)

Below is a practical implementation using a **Random Forest Classifier** and **GridSearchCV**.

```python
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# 1. Load data
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define the Model
rf = RandomForestClassifier()

# 3. Define the Hyperparameter Grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

# 4. Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=rf, 
    param_grid=param_grid, 
    cv=5,            # 5-fold cross-validation
    n_jobs=-1,       # Use all available CPU cores
    verbose=2
)

# 5. Fit the model
grid_search.fit(X_train, y_train)

# 6. Best Parameters and Score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# 7. Evaluate on Test Set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print(f"Test Set Accuracy: {test_accuracy:.4f}")
```
