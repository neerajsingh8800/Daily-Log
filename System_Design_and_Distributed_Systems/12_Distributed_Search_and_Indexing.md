# Module 12: Distributed Search and Indexing

Search engines process textual and high-dimensional vector data across distributed clusters. Distributed search systems rely on **Inverted Indexes** for lexical keyword retrieval and **Hierarchical Navigable Small World (HNSW)** graphs for dense vector semantic search.

This module covers lexical indexing algorithms (Inverted Indexes, Term Frequency-Inverse Document Frequency / TF-IDF, Okapi BM25), high-dimensional Approximate Nearest Neighbor (ANN) search via HNSW vector graphs, mathematical formulations for scoring and graph routing, and a complete distributed search engine implementation in Python.

---

## 1. Theoretical Foundations

### 1.1 Lexical Search vs. Vector Semantic Search

* **Inverted Index & Okapi BM25 (Lexical Search)**:
  * Maps distinct tokens/words to a posting list containing document IDs, positions, and term frequencies.
  * BM25 scores relevance by computing term saturation and normalizing against document length, preventing long documents from dominating results.
  * Used in traditional search engines like Elasticsearch and Apache Lucene.

* **Vector Search & HNSW (Semantic Search)**:
  * Embeds unstructured text, images, or audio into dense numerical vectors ($d$-dimensional space, e.g., $d=1536$).
  * **HNSW (Hierarchical Navigable Small World)** builds a multi-layer proximity graph where upper layers contain long-range skip links (fast routing) and lower layers contain short-range nearest-neighbor links (precision refinement).
  * Achieves logarithmic search time ($O(\log N)$) for Approximate Nearest Neighbor (ANN) queries. Used in vector databases like Pinecone, Milvus, and Qdrant.
 
  * ---

### 1.2 Mathematical Foundations

#### 1. Okapi BM25 Scoring Formula
For a search query $Q = \{q_1, q_2, \dots, q_n\}$ and a document $D$, the BM25 relevance score $S_{\text{BM25}}(D, Q)$ is:

$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{\vert{}D\vert{}}{\text{avgdl}}\right)}$$

Where:
* $f(q_i, D)$: Term frequency of query token $q_i$ in document $D$.
* $\vert{}D\vert{}$ and $\text{avgdl}$: Length of document $D$ and average document length across the corpus.
* $k_1$: Term frequency saturation parameter (typically $k_1 \in [1.2, 2.0]$).
* $b$: Document length normalization parameter (typically $b = 0.75$).

The Inverse Document Frequency $\text{IDF}(q_i)$ for total $N$ documents and $n(q_i)$ documents containing $q_i$ is:

$$\text{IDF}(q_i) = \ln \left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 \right)$$

#### 2. Vector Similarity Metrics (Cosine & Euclidean Distance)
For query vector $\mathbf{u}$ and document vector $\mathbf{v}$ in $d$-dimensional space:

* **Cosine Similarity** ($\cos \theta \in [-1, 1]$):

$$\text{Sim}_{\text{Cosine}}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\Vert{}\mathbf{u}\Vert{} \Vert{}\mathbf{v}\Vert{}} = \frac{\sum_{i=1}^{d} u_i v_i}{\sqrt{\sum_{i=1}^{d} u_i^2} \sqrt{\sum_{i=1}^{d} v_i^2}}$$

* **Euclidean Distance** ($L_2$ Distance):

$$d_{L2}(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^{d} (u_i - v_i)^2}$$

---

## 2. Search Paradigm Comparison

| Feature / Metric | Inverted Index (BM25) | Vector Graph (HNSW) | Hybrid Search |
| :--- | :--- | :--- | :--- |
| **Search Mechanism** | Keyword token matching | Dense vector spatial proximity | Reciprocal Rank Fusion (RRF) |
| **Query Type** | Exact match, SKU, names | Conceptual, natural language | Combined keyword + context |
| **Index Complexity** | $O(N)$ postings list space | $O(N \cdot M)$ graph edges | Dual index overhead |
| **Latency Penalty** | Extremely fast ($<5\text{ms}$) | Medium ($5-20\text{ms}$) | Requires fusion step |
| **Out-of-Vocabulary (OOV)**| Poor (Requires exact token) | Excellent (Handled by embedding) | High resilience |

---

## 3. Production Search & Indexing Engine Implementation

This Python module implements an **Inverted Index with BM25 Scoring** combined with a **Cosine Similarity Vector Search Engine**.

### Prerequisites

```bash
pip install numpy
```

### Python Implementation (search_engine.py)
```python
import math
import re
from typing import List, Dict, Tuple, Set
import numpy as np


# -------------------------------------------------------------------
# 1. LEXICAL INVERTED INDEX WITH BM25 SCORING
# -------------------------------------------------------------------
class InvertedIndexBM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.index: Dict[str, Dict[int, int]] = {}  # term -> {doc_id: term_freq}
        self.doc_lengths: Dict[int, int] = {}       # doc_id -> doc length
        self.documents: Dict[int, str] = {}         # doc_id -> raw text
        self.avg_doc_length: float = 0.0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def add_document(self, doc_id: int, text: str):
        self.documents[doc_id] = text
        tokens = self._tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)

        for token in tokens:
            if token not in self.index:
                self.index[token] = {}
            self.index[token][doc_id] = self.index[token].get(doc_id, 0) + 1

        total_length = sum(self.doc_lengths.values())
        self.avg_doc_length = total_length / len(self.doc_lengths)

    def _idf(self, term: str) -> float:
        n_q = len(self.index.get(term, {}))
        if n_q == 0:
            return 0.0
        N = len(self.documents)
        return math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        query_tokens = self._tokenize(query)
        scores: Dict[int, float] = {}

        for token in query_tokens:
            if token not in self.index:
                continue

            idf = self._idf(token)
            postings = self.index[token]

            for doc_id, freq in postings.items():
                doc_len = self.doc_lengths[doc_id]
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
                bm25_score = idf * (numerator / denominator)

                scores[doc_id] = scores.get(doc_id, 0.0) + bm25_score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(doc_id, score, self.documents[doc_id]) for doc_id, score in ranked]


# -------------------------------------------------------------------
# 2. VECTOR SEARCH ENGINE (DENSE COSINE SIMILARITY)
# -------------------------------------------------------------------
class VectorSearchEngine:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.doc_ids: List[int] = []
        self.embeddings: List[np.ndarray] = []

    def add_vector(self, doc_id: int, vector: List[float]):
        arr = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm  # Normalize to unit length for fast dot-product cosine similarity
        self.doc_ids.append(doc_id)
        self.embeddings.append(arr)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[int, float]]:
        if not self.embeddings:
            return []

        q_arr = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_arr)
        if q_norm > 0:
            q_arr = q_arr / q_norm

        matrix = np.vstack(self.embeddings)
        similarities = np.dot(matrix, q_arr)

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.doc_ids[idx], float(similarities[idx])) for idx in top_indices]


# -------------------------------------------------------------------
# 3. HYBRID SEARCH ENGINE WITH RECIPROCAL RANK FUSION (RRF)
# -------------------------------------------------------------------
class HybridSearchEngine:
    def __init__(self, bm25_engine: InvertedIndexBM25, vector_engine: VectorSearchEngine):
        self.bm25 = bm25_engine
        self.vector_search = vector_engine

    def search(self, query_text: str, query_vector: List[float], top_k: int = 5, rrf_k: int = 60) -> List[Tuple[int, float]]:
        bm25_results = self.bm25.search(query_text, top_k=top_k * 2)
        vector_results = self.vector_search.search(query_vector, top_k=top_k * 2)

        rrf_scores: Dict[int, float] = {}

        # Compute RRF for BM25
        for rank, (doc_id, _, _) in enumerate(bm25_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Compute RRF for Vector Search
        for rank, (doc_id, _) in enumerate(vector_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_docs


# -------------------------------------------------------------------
# VERIFICATION RUNNER
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Testing Inverted Index BM25 Search ---")
    bm25 = InvertedIndexBM25()
    bm25.add_document(1, "Distributed systems require consensus protocols like Raft or Paxos.")
    bm25.add_document(2, "LSM trees optimize write performance in storage engines.")
    bm25.add_document(3, "Raft consensus protocol maintains linearizable state replication.")

    bm25_res = bm25.search("consensus protocol")
    for doc_id, score, text in bm25_res:
        print(f"Doc {doc_id} | Score: {score:.4f} | Text: '{text}'")

    print("\n--- Testing Vector Search Engine ---")
    vec_engine = VectorSearchEngine(dimension=4)
    vec_engine.add_vector(1, [0.1, 0.8, 0.5, 0.2])
    vec_engine.add_vector(2, [0.9, 0.1, 0.0, 0.3])
    vec_engine.add_vector(3, [0.2, 0.7, 0.6, 0.1])

    vec_res = vec_engine.search([0.15, 0.75, 0.55, 0.18], top_k=2)
    for doc_id, sim in vec_res:
        print(f"Doc {doc_id} | Cosine Similarity: {sim:.4f}")

    print("\n--- Testing Hybrid Search with Reciprocal Rank Fusion (RRF) ---")
    hybrid = HybridSearchEngine(bm25, vec_engine)
    hybrid_res = hybrid.search("consensus protocol", [0.15, 0.75, 0.55, 0.18], top_k=2)
    for doc_id, rrf_score in hybrid_res:
        print(f"Doc {doc_id} | Combined RRF Score: {rrf_score:.5f}")
```

## 4. Operational Best Practices

* Index Sharding Strategies: Split large search indexes horizontally using Document-based Sharding (local inverted index per shard, lower scatter-gather latency) rather than Term-based Sharding.
* Reciprocal Rank Fusion (RRF): Combine keyword search (BM25) and dense vector search (HNSW) using RRF to achieve optimal retrieval accuracy without needing to normalize scale differences across score distributions.
* Vector Index Quantization: Apply Product Quantization (PQ) or Scalar Quantization (SQ8) to dense vectors to compress memory consumption by up to 75% with minimal loss in recall.
