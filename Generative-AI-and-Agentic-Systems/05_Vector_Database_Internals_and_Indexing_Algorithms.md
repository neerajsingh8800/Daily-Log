# Vector Database Internals and Indexing Algorithms

Standard relational databases optimize for exact matching over structured scalar values using B-Trees or Hash Indexes. However, vector searches required for Retrieval-Augmented Generation (RAG) operate over dense, high-dimensional floating-point embeddings ($\mathbb{R}^d$). Performing an exact K-Nearest Neighbor (k-NN) search requires an exhaustive linear scan ($O(N)$) across the entire dataset, which quickly stalls under enterprise loads.

This document explores the architectural mechanics of **Approximate Nearest Neighbor (ANN)** algorithms, focusing on the trade-offs between search speed, memory footprint, and recall accuracy.

---

## 1. Vector Search Trade-offs & The ANN Paradigm

To handle millions of vectors at sub-second latencies, vector databases trade perfect accuracy for immense speed gains using ANN indexing. The performance of these indexes is evaluated across three competing dimensions:

1. **Inverted File Index (IVF):** Limits the search scope by partitioning the vector space into localized clusters. (Optimizes **Latency**).
2. **Product Quantization (PQ):** Compresses high-dimensional vectors into tiny byte-codes. (Optimizes **Memory**).
3. **Hierarchical Navigable Small Worlds (HNSW):** Constructs a multi-layered geometric graph optimized for fast skip-scans. (Optimizes **Latency & Recall** at the cost of high memory usage).

---

## 2. Mathematical Foundations of Compression and Partitioning

### I. Inverted File Index (IVF) Cell Partitioning
IVF uses **K-Means Clustering** to partition a multi-dimensional vector space into Voronoi cells. 

Given a set of database vectors $\{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N\}$, the space is clustered into $K$ centroids $\{\mathbf{c}_1, \mathbf{c}_2, \dots, \mathbf{c}_K\}$. During indexing, every vector is assigned to its nearest centroid's posting list:

$$\mathbf{x}_i \in \text{List}(\mathbf{c}_j) \iff j = \arg\min_{k} \|\mathbf{x}_i - \mathbf{c}_k\|^2$$

* **Query Execution (`nprobe`):** When a user query $\mathbf{q}$ arrives, the engine calculates the distance to all centroids, picks the top $M$ closest centroids (where $M = \text{nprobe}$), and restricts its vector search *exclusively* to those specific posting lists. This slashes the search space from $N$ to $\approx \frac{\text{nprobe}}{K} \times N$.

### II. Product Quantization (PQ) Compression
Product Quantization reduces the physical storage footprint of a vector by breaking it down into smaller sub-vectors and quantizing them independently.

1. A $d$-dimensional vector is split into $m$ distinct, lower-dimensional sub-vectors of size $d' = d/m$.
2. For each sub-space, a mini-codebook containing $2^* = 256$ centroids is trained.
3. Every sub-vector is replaced by the 1-byte index ($0\text{--}255$) of its nearest sub-space centroid. This compresses a $d \times 32\text{-bit}$ float vector down to a compact $m \times 8\text{-bit}$ byte configuration.

### III. Asymmetric Distance Computation (ADC)
To maximize throughput, the database avoids decompressing vectors during query execution. Instead, it uses **Asymmetric Distance Computation (ADC)**. 

Given an unquantized user query vector $\mathbf{q}$ and a compressed database vector $\mathbf{x}$, the query is split into matching sub-vectors $\mathbf{q} = [\mathbf{q}_1, \dots, \mathbf{q}_m]$. The engine pre-computes the exact distances from each query piece $\mathbf{q}_i$ to all 256 centroids within that sub-space's codebook, storing them in a quick-lookup table. The approximate distance is then computed by adding up the pre-calculated values directly from the code indices:

$$\text{Dist}_{\text{ADC}}(\mathbf{q}, \mathbf{x}) \approx \sum_{i=1}^{m} \|\mathbf{q}_i - \mathbf{c}_i[\text{code}_i(\mathbf{x})]\|^2$$

---

## 3. Production-Grade Implementation: IVF-PQ Vector Index From Scratch

Below is a complete, self-contained Python architecture implementing a high-dimensional vector database index using an Inverted File Index (IVF) paired with Product Quantization (PQ) compression and Asymmetric Distance Lookups (ADC).

```python
import math
import random
from typing import List, Dict, Tuple, Any

class IVFPQIndex:
    """Pure Python implementation of an IVF-PQ Approximate Nearest Neighbor Index."""
    def __init__(self, d: int, nlist: int, m: int, nprobe: int):
        assert d % m == 0, "Vector dimension 'd' must be perfectly divisible by sub-vector slice count 'm'."
        self.d = d                  # Original dimension footprint
        self.nlist = nlist          # Number of IVF cluster centroids
        self.m = m                  # Number of PQ sub-vector segments
        self.nprobe = nprobe        # Number of IVF cells to probe during query scans
        self.d_prime = d // m       # Dimension of each sub-vector segment
        
        # Index Storage Structures
        self.ivf_centroids: List[List[float]] = []
        self.ivf_posting_lists: Dict[int, List[Tuple[int, List[int]]]] = {} # centroid_id -> [(doc_id, pq_codes)]
        self.pq_codebooks: List[List[List[float]]] = [] # m sub-spaces -> 256 centroids -> d_prime floats

    @staticmethod
    def _euclidean_distance(v1: List[float], v2: List[float]) -> float:
        """Computes standard Euclidean distance."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

    def _mini_kmeans(self, data: List[List[float]], k: int, iterations: int = 5) -> List[List[float]]:
        """A lightweight, zero-dependency K-Means implementation to train centroids."""
        if not data:
            return [[0.0] * len(data[0])] for _ in range(k)]
        
        # Initialize centroids randomly from available data points
        centroids = [list(x) for x in random.sample(data, min(k, len(data)))]
        while len(centroids) < k:
            centroids.append([0.0] * len(data[0]))
            
        for _ in range(iterations):
            clusters = [[] for _ in range(k)]
            for item in data:
                best_idx = min(range(k), key=lambda idx: self._euclidean_distance(item, centroids[idx]))
                clusters[best_idx].append(item)
                
            for idx in range(k):
                if clusters[idx]:
                    # Recalculate mean center points
                    centroids[idx] = [sum(col) / len(clusters[idx]) for col in zip(*clusters[idx])]
        return centroids

    def fit(self, training_vectors: List[List[float]]):
        """Trains both the global IVF layout and the sub-space PQ codebooks."""
        print(" -> Training IVF Cluster Centroids...")
        self.ivf_centroids = self._mini_kmeans(training_vectors, self.nlist, iterations=5)
        for i in range(self.nlist):
            self.ivf_posting_lists[i] = []
            
        print(" -> Splitting sub-vectors and training PQ Codebooks...")
        for sub_idx in range(self.m):
            sub_data = []
            start_dim = sub_idx * self.d_prime
            end_dim = start_dim + self.d_prime
            
            for vec in training_vectors:
                sub_data.append(vec[start_dim:end_dim])
                
            # Train exactly 256 quantization prototypes per sub-space block
            sub_codebook = self._mini_kmeans(sub_data, k=256, iterations=4)
            self.pq_codebooks.append(sub_codebook)
        print(" Index training routines finalized successfully.")

    def add(self, doc_id: int, vector: List[float]):
        """Quantizes an incoming vector and routes it to its matching IVF cell."""
        # Step 1: Route to the nearest global IVF centroid
        best_centroid_id = min(range(self.nlist), key=lambda idx: self._euclidean_distance(vector, self.ivf_centroids[idx]))
        
        # Step 2: Perform Product Quantization across sub-vectors
        pq_codes = []
        for sub_idx in range(self.m):
            start_dim = sub_idx * self.d_prime
            end_dim = start_dim + self.d_prime
            sub_vec = vector[start_dim:end_dim]
            
            # Find closest sub-space centroid code matching this segment
            best_code = min(range(256), key=lambda idx: self._euclidean_distance(sub_vec, self.pq_codebooks[sub_idx][idx]))
            pq_codes.append(best_code)
            
        # Append documentation data entry to the targeted inverted posting list bucket
        self.ivf_posting_lists[best_centroid_id].append((doc_id, pq_codes))

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[int, float]]:
        """Executes an asymmetric, chunked distance lookup over probed posting lists."""
        # Step 1: Identify the closest IVF cells to query based on nprobe settings
        centroid_distances = [(idx, self._euclidean_distance(query_vector, self.ivf_centroids[idx])) for idx in range(self.nlist)]
        centroid_distances.sort(key=lambda x: x[1])
        probed_centroid_ids = [centroid_distances[i][0] for i in range(min(self.nprobe, self.nlist))]
        
        # Step 2: Precompute the Asymmetric Distance Lookup Table (ADC Table)
        # adc_table[sub_space_index][code_byte_id] -> float distance
        adc_table: List[List[float]] = []
        for sub_idx in range(self.m):
            start_dim = sub_idx * self.d_prime
            end_dim = start_dim + self.d_prime
            query_sub_vec = query_vector[start_dim:end_dim]
            
            sub_distances = []
            for code_idx in range(256):
                centroid_sub_vec = self.pq_codebooks[sub_idx][code_idx]
                sub_distances.append(self._euclidean_distance(query_sub_vec, centroid_sub_vec))
            adc_table.append(sub_distances)
            
        # Step 3: Scan the selected cells and look up distances via the ADC table
        results = []
        for centroid_id in probed_centroid_ids:
            for doc_id, codes in self.ivf_posting_lists[centroid_id]:
                # Asymmetric distance aggregation loop
                approx_dist = sum(adc_table[sub_idx][codes[sub_idx]] for sub_idx in range(self.m))
                results.append((doc_id, approx_dist))
                
        # Return sorted nearest neighbors
        results.sort(key=lambda x: x[1])
        return results[:top_k]


# --- Operational Execution Verification Sandbox ---
if __name__ == "__main__":
    random.seed(42) # Lock seed mechanics for deterministic logging checks
    
    # Configuration parameters: 8 dimensions, 3 coarse IVF lists, 2 PQ sub-spaces
    dimensions = 8
    ivf_lists = 3
    pq_subspaces = 2
    probe_cells = 2
    
    # Generate 20 random mock training embeddings
    mock_dataset = [[random.uniform(-1.0, 1.0) for _ in range(dimensions)] for _ in range(20)]
    
    # Initialize index object
    index = IVFPQIndex(d=dimensions, nlist=ivf_lists, m=pq_subspaces, nprobe=probe_cells)
    index.fit(mock_dataset)
    
    # Populate index entries
    for identity, embedding in enumerate(mock_dataset, start=501):
        index.add(doc_id=identity, vector=embedding)
        
    # Execute an approximate search pipeline step
    runtime_query = [0.1, -0.5, 0.9, 0.2, -0.2, 0.4, 0.7, -0.1]
    search_matches = index.search(runtime_query, top_k=3)
    
    print("\n=== IVF-PQ Approximate Nearest Neighbor Execution ===")
    print(f"Query Vector: {runtime_query}")
    print(f"Top 3 Closest Match Results (ID, Approx Distance): {search_matches}")
```
