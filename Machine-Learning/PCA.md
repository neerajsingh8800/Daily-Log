# Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is an **unsupervised learning** algorithm used for **dimensionality reduction**. It transforms a large set of variables into a smaller one that still contains most of the information in the large set.

---

## 1. The Core Intuition
PCA works by identifying the "principal components"—the directions (vectors) along which the data varies the most. 
* **Goal:** Reduce the number of features ($d$) to $k$ features while maximizing the variance.
* **Benefit:** Simplifies data, reduces noise, and helps in visualizing high-dimensional datasets.

---

## 2. Mathematical Steps
PCA follows a specific linear algebra pipeline:

### Step 1: Standardization
Since PCA is sensitive to variances, we must scale the data to have a mean of 0 and a standard deviation of 1.
$$z = \frac{x - \mu}{\sigma}$$

### Step 2: Covariance Matrix Computation
We calculate the covariance matrix to understand how variables in the input data are varying from the mean with respect to each other.
$$Cov(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$$

### Step 3: Eigenvalues & Eigenvectors
We solve the characteristic equation to find the **Eigenvalues** ($\lambda$) and **Eigenvectors** ($v$):
$$Av = \lambda v$$
* **Eigenvectors:** Determine the direction of the new feature space.
* **Eigenvalues:** Determine the magnitude (how much variance is explained by that direction).

### Step 4: Feature Vector
We sort eigenvalues in descending order and choose the top $k$ eigenvectors to form a projection matrix.

---

## 3. Key Concepts to Remember
* **Explained Variance Ratio:** Tells you how much information (variance) can be attributed to each of the principal components.
* **Orthogonality:** All principal components are perpendicular to each other, meaning they are uncorrelated.
* **Information Loss:** By reducing dimensions, you inevitably lose some detail; the goal is to keep this loss minimal (e.g., keeping 95% of the variance).

---

## 4. Python Implementation (Scikit-Learn)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# 1. Load Dataset
data = load_iris()
X = data.data
y = data.target

# 2. Standardize the data (Crucial for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA
# We reduce the 4-feature Iris dataset to 2 dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Check Explained Variance
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Total Variance Retained: {np.sum(pca.explained_variance_ratio_):.2%}")

# 5. Visualization
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=40)
plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Target Class')
plt.show()
```
