# Support Vector Machines (SVM)

Support Vector Machine (SVM) is a powerful and versatile **Supervised Machine Learning** algorithm used for both classification and regression. However, it is most commonly used for classification problems.

---

## 1. Core Intuition
The goal of SVM is to find the **Optimal Hyperplane** in an N-dimensional space that distinctly classifies the data points. 

* **Hyperplane:** A decision boundary that separates different classes.
* **Support Vectors:** These are the data points that are closest to the hyperplane. They are the critical elements of the training set; if they were removed, the position of the dividing hyperplane would change.
* **Margin:** The distance between the hyperplane and the nearest data point from either set. SVM aims to **maximize** this margin (Large Margin Classifier).

---

## 2. Mathematical Foundation

### The Hyperplane Equation
In a 2D space, a line is defined as $y = ax + b$. In SVM, we use the vector form:
$$w \cdot x + b = 0$$
Where:
* $w$ is the weight vector (normal to the hyperplane).
* $x$ is the input feature vector.
* $b$ is the bias.

### Linear Classification
For a binary classification problem where labels are $y \in \{-1, 1\}$:
* $w \cdot x_i + b \ge +1$ if $y_i = 1$
* $w \cdot x_i + b \le -1$ if $y_i = -1$

This can be combined into a single constraint:
$$y_i(w \cdot x_i + b) \ge 1$$

### Maximizing the Margin
The distance between the two support vector planes is $\frac{2}{\|w\|}$. To maximize this distance, we need to **minimize** $\|w\|$. This is typically solved as a constrained optimization problem using Lagrange Multipliers.

---

## 3. Kernel Trick
When data is not linearly separable in the current dimension, SVM uses **Kernels** to project the data into a higher-dimensional space where a linear boundary can be found.

Common Kernel functions:
* **Linear:** $K(x, x') = x \cdot x'$
* **Polynomial:** $K(x, x') = (\gamma x \cdot x' + r)^d$
* **RBF (Radial Basis Function):** $K(x, x') = \exp(-\gamma \|x - x'\|^2)$

---

## 4. Implementation in Python

Using the `scikit-learn` library to implement a simple SVM classifier.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load Dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # We only take the first two features for visualization
y = iris.target

# Binary classification: only take class 0 and 1
X = X[y != 2]
y = y[y != 2]

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Create and Train the Model
# Using RBF kernel; C is the regularization parameter
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# 4. Make Predictions
y_pred = model.predict(X_test)

# 5. Evaluate the Model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```
