# Decision Trees

Decision Trees are a non-parametric supervised learning method used for both **classification** and **regression**. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

---

## 1. Core Concepts

A Decision Tree consists of:
* **Root Node:** The top-most node representing the entire dataset, which gets split into two or more homogeneous sets.
* **Splitting:** The process of dividing a node into two or more sub-nodes.
* **Decision Node:** When a sub-node splits into further sub-nodes.
* **Leaf / Terminal Node:** Nodes that do not split; they represent the final prediction/class.
* **Pruning:** The process of removing sub-nodes of a decision node to prevent overfitting.

---

## 2. Mathematical Foundation (Splitting Criteria)

The model decides where to split based on the "purity" of the resulting nodes.

### A. Entropy
Entropy measures the impurity or randomness in the dataset.
$$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$
*Where $p_i$ is the probability of an element belonging to class $i$.*

### B. Information Gain
Information Gain is the reduction in entropy after a dataset is split on an attribute.
$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

### C. Gini Impurity
Used by the CART (Classification and Regression Tree) algorithm. It is computationally faster than Entropy because it doesn't involve logarithms.
$$Gini = 1 - \sum_{i=1}^{n} (p_i)^2$$

---

## 3. Important Hyperparameters

To prevent **Overfitting**, we tune these parameters in Scikit-Learn:
* `criterion`: "gini" or "entropy".
* `max_depth`: The maximum depth of the tree.
* `min_samples_split`: The minimum number of samples required to split an internal node.
* `min_samples_leaf`: The minimum number of samples required to be at a leaf node.

---

## 4. Implementation Example

Using `scikit-learn` to classify the Iris dataset:

```python
from sklearn.datasets import load_iris
from sklearn.model_code_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Load data
data = load_iris()
X, y = data.data, data.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Fit
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 4. Evaluate
accuracy = clf.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 5. Visualize the Tree
plt.figure(figsize=(12,8))
plot_tree(clf, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.show()
```
