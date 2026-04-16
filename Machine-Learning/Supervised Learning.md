## Supervised Learning

Supervised Learning is the most common sub-field of Machine Learning. It involves training an algorithm on a **labeled dataset**, where the input data ($x$) is paired with the correct output ($y$). The goal is to learn a mapping function so accurately that the model can predict the output for new, unseen data.

The core relationship can be expressed as:
$y = f(x) + \epsilon$
*Where y is the prediction, f(x) is the learned function, and epsilon represents irreducible error.*

---

## 1. Classification (Categorical Output)
Used when the target variable is a category or discrete label.

* **Logistic Regression:** Predicts the probability of a binary outcome (e.g., Yes/No, 0/1).
* **K-Nearest Neighbors (KNN):** Classifies data points based on the majority class of their nearest neighbors.
* **Support Vector Machines (SVM):** Creates a boundary (hyperplane) to separate different classes with maximum margin.
* **Naive Bayes:** A probabilistic classifier based on Bayes' Theorem, assuming independence between features.
* **Decision Trees:** A flowchart-like structure that splits data based on feature values to reach a classification.

## 2. Regression (Continuous Output)
Used when the target variable is a continuous numerical value.

* **Linear Regression:** Fits a straight line to establish a relationship between independent and dependent variables.
* **Polynomial Regression:** Used when the data relationship is non-linear; it fits a curve instead of a line.
* **Ridge & Lasso Regression:** Specialized versions of linear regression that use "Regularization" to prevent overfitting.

## 3. Ensemble Methods
These combine multiple models to create a more powerful and stable predictor.

* **Random Forest:** An ensemble of many Decision Trees, usually trained with the "bagging" method to improve accuracy.
* **Gradient Boosting (XGBoost/LightGBM):** Sequentially builds models where each new model corrects the errors made by previous ones.

---

## The Supervised Learning Workflow
1.  **Data Collection:** Gather features and their corresponding labels.
2.  **Preprocessing:** Handle missing values, outliers, and scale the data.
3.  **Splitting:** Divide data into **Training** (80%) and **Testing** (20%) sets.
4.  **Training:** Fit the chosen algorithm to the training data.
5.  **Evaluation:** Check performance using metrics like **Accuracy**, **F1-Score** (Classification), or **RMSE** (Regression).
