# Naive Bayes Classifier

Naive Bayes is a probabilistic machine learning algorithm based on **Bayes' Theorem**. It is primarily used for classification tasks like spam filtering and sentiment analysis.

## 1. Why "Naive"?
It is called "Naive" because it assumes that all features are **independent** of each other. In reality, features often correlate, but this simplification makes the algorithm incredibly fast and effective for high-dimensional data.

---

## 2. Theoretical Foundation

### Bayes' Theorem
The theorem calculates the probability of a class ($y$) given the input features ($X$):

$$P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)}$$

### The Naive Equation
Since we assume independence, the probability for multiple features $x_1, x_2, \dots, x_n$ is calculated as:

$$P(y|x_1, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i|y)$$

---

## 3. Types of Naive Bayes
* **GaussianNB:** For continuous data (assumes a Normal distribution).
* **MultinomialNB:** For discrete counts (text classification/word counts).
* **BernoulliNB:** For binary/boolean features (yes/no data).

---

## 4. Implementation (Python)

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
data = load_iris()
X, y = data.data, data.target

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Initialize the Gaussian Naive Bayes model
model = GaussianNB()

# 4. Train the model
model.fit(X_train, y_train)

# 5. Predict and Evaluate
y_pred = model.predict(X_test)

print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nDetailed Report:\n", classification_report(y_test, y_pred))
