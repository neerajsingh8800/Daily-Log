# K-Means Clustering: A Comprehensive Guide

K-Means is one of the most popular and simple **unsupervised machine learning** algorithms. Its goal is to group similar data points together and discover underlying patterns.

---

## 1. Core Concepts & Theory

K-Means clustering identifies $K$ number of centroids, and then allocates every data point to the nearest cluster, while keeping the centroids as small as possible. 

### Key Characteristics:
* **Unsupervised:** It works on unlabeled data.
* **Centroid-based:** It organizes data into non-hierarchical groups.
* **Iterative:** It refines the position of centroids until convergence.

### How it Works (The Algorithm):
1.  **Initialization:** Choose the number of clusters $K$ and randomly select $K$ points as initial centroids.
2.  **Assignment:** Assign each data point to the nearest centroid based on the squared Euclidean distance.
3.  **Update:** Calculate the new mean (center) of all points in each cluster. These become the new centroids.
4.  **Repeat:** Repeat steps 2 and 3 until centroids no longer move significantly or the maximum number of iterations is reached.

---

## 2. Mathematical Foundation

The objective of K-Means is to minimize the **Within-Cluster Sum of Squares (WCSS)**, also known as **Inertia**.

### Euclidean Distance
To determine the "closeness" of a point $P_1(x_1, y_1)$ to a centroid $C_1(x_2, y_2)$, we use:
$$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

### Objective Function
The algorithm seeks to minimize the squared error for all clusters:
$$J = \sum_{j=1}^{K} \sum_{i=1}^{n} \|x_i^{(j)} - c_j\|^2$$
Where:
* $K$ is the number of clusters.
* $n$ is the number of data points.
* $x_i^{(j)}$ is the $i^{th}$ point in cluster $j$.
* $c_j$ is the centroid of cluster $j$.

---

## 3. Choosing the Right 'K' (The Elbow Method)
Since we have to provide $K$ manually, we often use the **Elbow Method**. We plot the WCSS values against different values of $K$. The point where the graph forms an "elbow" (the rate of decrease sharply slows down) is usually the optimal $K$.

---

## 4. Python Implementation (Scikit-Learn)

Here is a practical implementation using `numpy` and `scikit-learn`.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Generate Synthetic Data
# Creating 300 samples with 4 distinct centers
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 2. Visualizing the Elbow Method to find optimal K
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# 3. Applying K-Means to the dataset
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# 4. Visualizing the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=50, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=50, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=50, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=50, c='cyan', label='Cluster 4')

# Plotting the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='yellow', label='Centroids')
plt.title('Clusters of Data')
plt.legend()
plt.show()
```
