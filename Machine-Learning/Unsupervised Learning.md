# Unsupervised Learning

Unsupervised Learning is a type of machine learning that looks for previously unknown patterns in a dataset without pre-existing labels. It is primarily used for exploratory data analysis to find the underlying structure of data.

---

## 🚀 Core Objectives
* **Clustering:** Automatically grouping data points with similar characteristics.
* **Dimensionality Reduction:** Compressing data by reducing the number of variables while keeping essential information.
* **Association:** Discovering rules that describe large portions of your data (e.g., Market Basket Analysis).

---

## 🛠 Key Algorithms & Definitions

### 1. Clustering Algorithms
* **K-Means Clustering:** * **Definition:** Groups data into $K$ number of clusters by minimizing the distance between points and their cluster centroid.
    * **Use Case:** Customer segmentation.
* **Hierarchical Clustering:** * **Definition:** Creates a tree of clusters (Dendrogram) by either merging small clusters (Agglomerative) or splitting large ones (Divisive).
* **DBSCAN (Density-Based Spatial Clustering):** * **Definition:** Groups points that are closely packed together and identifies points in low-density regions as outliers.
    * **Use Case:** Identifying anomalies or non-linear patterns.

### 2. Dimensionality Reduction
* **PCA (Principal Component Analysis):** * **Definition:** A linear technique that identifies the axes (Principal Components) where the data has the most variance.
* **t-SNE:** * **Definition:** A non-linear technique used for high-dimensional data visualization by preserving local relationships between points.

### 3. Association Rules
* **Apriori Algorithm:** * **Definition:** Identifies frequent itemsets in a database to establish "if-then" relationships.
    * **Use Case:** "Users who bought this also bought..." recommendations.

---

## 📊 Evaluation Metrics
Since there are no "correct" labels, we use internal consistency metrics:

| Metric | Description | Best Value |
| :--- | :--- | :--- |
| **Silhouette Score** | Measures how similar a point is to its own cluster vs. others. | Close to +1 |
| **Elbow Method** | A visual tool to find the optimal number of clusters ($K$). | The "elbow" point |
| **Inertia** | Sum of squared distances of samples to their closest cluster center. | Lower is better |

---

## 💡 Quick Comparison Table

| Feature | Supervised Learning | Unsupervised Learning |
| :--- | :--- | :--- |
| **Input Data** | Labeled (X + y) | Unlabeled (X only) |
| **Goal** | Prediction / Classification | Pattern / Structure Discovery |
| **Feedback** | Direct (Loss functions) | No direct feedback |

---
