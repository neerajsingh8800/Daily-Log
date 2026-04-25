# K-Nearest Neighbors (KNN)

K-Nearest Neighbors (KNN) is a **non-parametric**, **lazy learning** algorithm used for both classification and regression. It is one of the simplest yet most effective algorithms in machine learning.

---

## 1. Core Concepts
* **Non-parametric:** It doesn't make any assumptions about the underlying data distribution.
* **Lazy Learner:** It does not "learn" a discriminative function from the training data. Instead, it stores the entire training dataset and performs computations only during the prediction phase.
* **Instance-based:** It relies on the memory of previously seen instances to classify new ones.

---

## 2. How the Algorithm Works
1.  **Choose the number $K$** of neighbors (usually an odd number to avoid ties).
2.  **Calculate the distance** between the new data point and all training data points.
3.  **Sort the distances** in ascending order.
4.  **Pick the top $K$** nearest neighbors.
5.  **Vote (for Classification):** Assign the class that appears most frequently among the $K$ neighbors.
6.  **Average (for Regression):** Assign the mean of the $K$ labels.

---

## 3. Mathematical Foundations: Distance Metrics
The performance of KNN depends heavily on how we define "closeness."

### Euclidean Distance
The most common metric, representing the straight-line distance between two points $P$ and $Q$ in $n$-dimensional space:
$$d(P, Q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}$$

### Manhattan Distance
Used when the path is restricted to a grid (like city blocks):
$$d(P, Q) = \sum_{i=1}^{n} |q_i - p_i|$$

### Minkowski Distance
A generalized form of both Euclidean and Manhattan:
$$d(P, Q) = \left( \sum_{i=1}^{n} |q_i - p_i|^p \right)^{1/p}$$
* If $p=1$, it is Manhattan distance.
* If $p=2$, it is Euclidean distance.

---

## 4. Choosing the Optimal $K$
* **Small $K$:** Sensitive to noise and outliers (can lead to **Overfitting**).
* **Large $K$:** Smoother decision boundaries but might include points from other classes (can lead to **Underfitting**).
* **The Elbow Method:** We often plot the error rate against different values of $K$ and pick the point where the error starts to stabilize.

---

## 5. Implementation (Python)

Below is a clean implementation using `scikit-learn` on the Iris dataset.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import load_iris

# 1. Load Dataset
data = load_iris()
X = data.data
y = data.target

# 2. Preprocessing
# KNN is distance-based, so Feature Scaling is MANDATORY
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Initialize and Train KNN
# We'll use K=5 and Euclidean distance (minkowski with p=2)
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train, y_train)

# 5. Predictions
y_pred = knn.predict(X_test)

# 6. Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```
