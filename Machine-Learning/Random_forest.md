# Random Forest Classifier & Regressor

## Introduction
**Random Forest** is an ensemble learning method that operates by constructing a multitude of decision trees at training time. For classification tasks, the output is the class selected by the majority of trees. For regression tasks, it is the mean or average prediction of the individual trees.

It is one of the most popular and powerful algorithms in Machine Learning because it handles the limitations of Decision Trees—specifically, their tendency to **overfit** the training data.

---

## Core Concepts

### 1. Ensemble Learning (Bagging)
Random Forest is based on **Bootstrap Aggregating**, or **Bagging**. 
* **Bootstrapping:** It creates multiple subsets of the original dataset by sampling with replacement. 
* **Aggregating:** It combines the predictions from all trees to produce a final result.

### 2. Feature Randomness
Unlike a standard decision tree that searches for the best feature among *all* available features when splitting a node, Random Forest picks a **random subset of features**. This ensures that the trees are decorrelated; one highly predictive feature won't dominate every single tree.

### 3. Out-of-Bag (OOB) Error
Since Random Forest uses bootstrapping, about $1/3$ of the data is left out for each tree. This "Out-of-Bag" data can be used to evaluate the model's performance without needing a separate validation set.

---

## The Mathematics Behind the Forest

### Gini Impurity (Classification)
Used to determine how "pure" a node is. A node is pure ($G = 0$) if all samples belong to one class.
$$G = 1 - \sum_{i=1}^{n} (P_i)^2$$
Where $P_i$ is the probability of an object being classified into a particular class.

### Entropy & Information Gain
Entropy measures the level of disorder. Information Gain is the reduction in entropy after a dataset is split.
$$E = -\sum_{i=1}^{n} P_i \log_2(P_i)$$

### Prediction Aggregation
* **Classification (Majority Vote):** $$\hat{y} = \text{mode}\{T_1(x), T_2(x), ..., T_n(x)\}$$
* **Regression (Average):** $$\hat{y} = \frac{1}{n} \sum_{i=1}^{n} T_i(x)$$

---

## Key Hyperparameters
| Hyperparameter | Description |
| :--- | :--- |
| `n_estimators` | The number of trees in the forest (Default: 100). |
| `max_depth` | The maximum depth of each tree. Controls overfitting. |
| `min_samples_split` | Minimum samples required to split an internal node. |
| `max_features` | The number of features to consider when looking for the best split. |
| `bootstrap` | Whether bootstrap samples are used when building trees. |

---

## Implementation Code

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris

# 1. Load Dataset
data = load_iris()
X = data.data
y = data.target

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Train Model
# We use 100 trees and set random_state for reproducibility
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# 4. Make Predictions
y_pred = rf_model.predict(X_test)

# 5. Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Feature Importance
importances = pd.DataFrame({
    'feature': data.feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop Features:\n", importances)
```
