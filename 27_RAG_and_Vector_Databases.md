# 27. Retrieval-Augmented Generation (RAG) and Vector Databases

Even the largest Large Language Models suffer from fixed knowledge cutoff dates and a tendency to hallucinate when asked about niche, private, or rapidly changing data. **Retrieval-Augmented Generation (RAG)** solves this bottleneck by anchoring the LLM to an external, dynamic knowledge source. Instead of relying purely on parametric memory (weights), the model leverages non-parametric memory (external text databases) during inference.

---

## 1. The Core RAG Workflow

The standard RAG architecture operates across a dual-stage pipeline: an offline data ingestion stage and an online inference generation loop.
### A. Ingestion (Offline Pipeline)
1. **Document Chunking:** Massive documents are broken down into discrete segments ($D = \{c_1, c_2, \dots, c_n\}$) using strategy rules like fixed-size windows with token overlaps to preserve semantic continuity across borders.
2. **Vector Space Generation:** Each text chunk $c_i$ passes through a dense embedding encoder model $\phi$ to produce a fixed-dimensional vector representation: $\mathbf{e}_i = \phi(c_i) \in \mathbb{R}^d$.
3. **Storage Indexing:** These continuous coordinate arrays are stored inside a specialized high-dimensional **Vector Database** (e.g., Milvus, Pinecone, FAISS) for rapid scanning.

### B. Inference (Online Execution)
1. **Query Embedding:** A user inputs a question $q$. The system converts it using the exact same embedding encoder: $\mathbf{e}_q = \phi(q)$.
2. **Vector Index Retrieval:** The vector database searches the index to isolate the top-$k$ document chunks whose embeddings match closest to the query coordinates $\mathbf{e}_q$.
3. **Prompt Augmentation & Decoding:** The raw text strings of these top-$k$ chunks are inserted into the LLM's system prompt context window along with the user's initial question. The LLM conditions its autoregressive generation on this grounding data.

---

## 2. High-Dimensional Similarity Vector Metrics

Vector databases evaluate the proximity between query $\mathbf{a}$ and document chunk $\mathbf{b}$ using three primary geometric functions:

### A. Cosine Similarity
Evaluates the directional alignment between two vectors regardless of their magnitude or length scale. It outputs values bounded between $[-1, 1]$.

$$\text{Cosine Similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} = \frac{\sum_{i=1}^d a_i b_i}{\sqrt{\sum_{i=1}^d a_i^2} \sqrt{\sum_{i=1}^d b_i^2}}$$

### B. Dot Product (Inner Product)
If your embedding models normalize output vectors to unit length ($\|\mathbf{a}\| = \|\mathbf{b}\| = 1$), the cosine calculation simplifies directly to a raw Dot Product, which accelerates processing speeds:

$$\text{Dot Product}(\mathbf{a}, \mathbf{b}) = \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^d a_i b_i$$

### C. Euclidean Distance ($L_2$ Distance)
Measures the absolute geometric spatial distance between coordinate tips. Smaller distances correspond to higher semantic similarity.

$$D_{L2}(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2 = \sqrt{\sum_{i=1}^d (a_i - b_i)^2}$$

---

## 3. Vector Database Indexing Mechanics

Linearly comparing a query vector against millions of document items via a brute-force loop ($O(N)$ complexity) creates an unacceptable performance bottleneck in production systems. Vector databases trade absolute precision for immense speed by constructing **Approximate Nearest Neighbor (ANN)** indexes:

* **Inverted File Index (IVF):** Uses $K$-means clustering to partition the vector space into voronoi regions. During a search, the database identifies the closest cluster centroids and only evaluates vectors inside those specific buckets, dropping search complexity down to $O(\sqrt{N})$.
* **Hierarchical Navigable Small World (HNSW):** Constructs a multi-layered, graph-based routing index. The top layers contain sparse links for fast long-distance spatial jumps, while lower layers contain dense local configurations for granular precision tuning (achieving logarithmic $O(\log N)$ search time profiles).

---

## 4. Implementation in Python (PyTorch)

This standalone script demonstrates how to build a complete **RAG pipeline from scratch** using PyTorch. It implements document chunking, a basic cosine similarity matrix lookup index, prompt construction routing, and simulated generation conditions.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalistVectorDatabase:
    """
    An in-memory vector database executing exact cosine similarity searches.
    """
    def __init__(self, embedding_dim: int):
        self.dim = embedding_dim
        self.vectors = torch.empty((0, embedding_dim))
        self.metadata = []

    def insert(self, vector: torch.Tensor, text_chunk: str):
        """Adds an embedding vector and its raw text metadata source."""
        # Ensure the vector is flat with shape [1, embedding_dim]
        vector = vector.view(1, -1)
        self.vectors = torch.cat([self.vectors, vector.detach()], dim=0)
        self.metadata.append(text_chunk)

    def query(self, query_vector: torch.Tensor, top_k: int = 2):
        """
        Computes cosine similarities to retrieve the top-k text contexts.
        """
        if self.vectors.size(0) == 0:
            return []
            
        query_vector = query_vector.view(1, -1)
        
        # 1. Normalize vectors to compute cosine similarity via dot product
        norm_vectors = F.normalize(self.vectors, p=2, dim=1)
        norm_query = F.normalize(query_vector, p=2, dim=1)
        
        # 2. Compute similarity matrix using matrix multiplication: [1, N]
        similarities = torch.mm(norm_query, norm_vectors.t()).squeeze(0)
        
        # 3. Extract indices of the top-k highest scoring items
        top_scores, top_indices = torch.topk(similarities, min(top_k, len(self.metadata)))
        
        results = []
        for score, idx in zip(top_scores, top_indices):
            results.append({
                "context": self.metadata[idx.item()],
                "similarity": score.item()
            })
        return results


# Mock Ingestion Pipeline Initialization Execution Pass
if __name__ == "__main__":
    torch.manual_seed(42)
    E_dim = 128  # Simulating embedding feature dimensional depth
    
    # Instantiate the custom database engine
    db = MinimalistVectorDatabase(embedding_dim=E_dim)
    
    # 1. Simulate document source chunks
    document_chunks = [
        "The Adam optimizer uses running averages of both the first and second moments of gradients.",
        "Transformer networks rely entirely on attention mechanisms to draw global dependencies.",
        "Quantization compresses neural network parameters from standard FP32 formats down to 8-bit integers.",
        "Residual connections help mitigate vanishing gradient issues by passing activations straight through."
    ]
    
    # 2. Generate mock embeddings to populate the database
    print("--- Running Vector Database Ingestion ---")
    for chunk in document_chunks:
        mock_embedding = torch.randn(E_dim) # Mimics a sentence transformer encoder output
        db.insert(mock_embedding, chunk)
    print(f"Successfully indexed {len(db.metadata)} text document chunks.\n")
    
    # 3. Execute a simulated user retrieval request
    user_query = "How do transformers handle long sequences without recurrence?"
    print(f"User Query: '{user_query}'")
    
    mock_query_embedding = torch.randn(E_dim) # Query vector from the same embedding model
    retrieved_contexts = db.query(mock_query_embedding, top_k=2)
    
    print("\n--- Retrieved Database Context Items ---")
    for i, item in enumerate(retrieved_contexts, 1):
        print(f"Rank {i} Match [Score: {item['similarity']:.4f}]:")
        print(f"  Context: {item['context']}")
        
    # 4. Construct the finalized augmented system prompt for the LLM input
    augmented_prompt = "Instructions: Answer the query using the factual context snippets provided below.\n\n"
    for item in retrieved_contexts:
        augmented_prompt += f"Context: {item['context']}\n"
    augmented_prompt += f"\nQuery: {user_query}\nAnswer:"
    
    print("\n--- Finalized Augmented Prompt Sent to LLM Context Window ---")
    print(augmented_prompt)
```
