# Logistic Regression: Classfication Algorithm

Logistic Regression is a fundamental statistical method used in Machine Learning for **binary classification** problems (where the outcome has two possible categories, such as 0 or 1, Yes or No, Spam or Not Spam).

---

## 1. Theoretical Background

Unlike Linear Regression, which predicts continuous values, Logistic Regression predicts the **probability** that a given input point belongs to a specific class. 

### The Sigmoid Function
To map any real-valued number to a probability value between 0 and 1, we use the **Sigmoid (or Logistic) function**:

$$S(z) = \frac{1}{1 + e^{-z}}$$

Where $z$ is the result of the linear equation:
$$z = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n$$

### Decision Boundary
By default, if the probability $P(y=1) \geq 0.5$, we classify the output as **1**. Otherwise, we classify it as **0**.

---

## 2. Cost Function: Log Loss

We cannot use Mean Squared Error (MSE) for Logistic Regression because the Sigmoid function makes the cost function non-convex, leading to multiple local minima. Instead, we use **Binary Cross-Entropy (Log Loss)**:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))]$$

* If $y=1$, the cost decreases as the prediction approaches 1.
* If $y=0$, the cost decreases as the prediction approaches 0.

---

## 3. Implementation (Python)

Below is a standard implementation using `scikit-learn` for a practical workflow and `numpy` for the conceptual logic.

### Using Scikit-Learn
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Generate Sample Data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                           n_clusters_per_class=1, random_state=42)

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Predictions
y_pred = model.predict(X_test)

# 5. Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```
