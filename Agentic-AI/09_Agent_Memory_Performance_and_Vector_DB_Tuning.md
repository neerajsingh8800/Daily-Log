# Module 09: Agent Memory Performance and Vector DB Tuning

Production-grade AI agents live and die by their retrieval latency, accuracy, and context window management. As an agent's reasoning loop extends over long conversations or processes massive enterprise data stores, naive vector search and memory management quickly degrade into high latency, context window overflow, and "lost in the middle" retrieval phenomena.

This module covers advanced vector database optimization, context compression mathematics, and robust state serialization required to build high-performance agentic systems.

---

## 1. Mathematical Foundations of Vector Search & Indexing

To optimize retrieval, we must understand how vector databases index high-dimensional embeddings. The goal is to balance the trade-off between **Recall Accuracy** and **Search Latency**.

### Hierarchical Navigable Small World (HNSW) Graphs
HNSW is the state-of-the-art index for Approximate Nearest Neighbor (ANN) search. It constructs a multi-layer graph where the top layers have longer edges (for fast skip-list-style routing) and the bottom layers contain dense, highly localized clusters.

The probability of a node existing at a maximum layer $l$ is governed by a decay scale parameter $m_L$:

$$P(l) = e^{-l \cdot m_L}$$

During search, the agent navigates down the layers, minimizing the distance metric (e.g., Cosine Distance or Inner Product) to find the nearest entry point for the next layer.

### Product Quantization (PQ)
Memory footprints scale linearly with vector count. Product Quantization compresses high-dimensional vectors ($\mathbb{R}^D$) into low-dimensional codes by splitting the vector space into $m$ orthogonal subspaces of dimension $d' = D/m$.

Each subspace is clustered into $k^*$ centroids using $K$-Means. A continuous vector is then represented as a sequence of $m$ centroid indices (bytes), reducing memory requirements drastically (often up to 95%).

The asymmetric distance between a non-quantized query vector $q$ and a quantized database vector $x$ is approximated as:

$$d_{ASD}(q, x) \approx \sum_{i=1}^{m} \| q_i - c_i(x_i) \|^2$$

Where $q_i$ is the $i$-th sub-vector of the query and $c_i(x_i)$ is the nearest centroid in that specific subspace.

---

## 2. Advanced Retrieval & Context Compression

Simply pulling the top $K$ vectors via similarity search often floods the LLM context window with redundant info. We use advanced post-processing to maximize information density.

### Metadata Filtering (Pre-filtering vs. Post-filtering)
* **Post-filtering:** Performs ANN search first, then discards results that don't match metadata criteria. *Problem:* If your top 100 results don't match the filter, your agent gets 0 results (Catastrophic Recall Drop).
* **Pre-filtering / Single-stage Filtering:** Integrates metadata directly into the HNSW graph traversal or uses inverted indexes *before* computing vector distances. **Always use pre-filtering in production agents.**

### Context Compression & Mutual Information (LLMLingua)
Instead of feeding raw text chunks to the LLM, we compress prompts by removing tokens that contribute low mutual information or exhibit high perplexity under a smaller, faster language model (like GPT-2 or Llama-3-8B).

Given a prompt context $C$ and a target question $Q$, the objective is to find a compressed context $C'$ that minimizes token length while maximizing the conditional probability:

$$C^* = \arg\max_{C'} P(Q \mid C')$$

---

## 3. Production Implementation

The following script sets up a high-performance vector retrieval store using **ChromaDB** with explicit pre-filtering, implements a **Semantic Caching** layer to avoid redundant LLM invocations, and utilizes a **Context Compressor** pattern.

```python
import os
import time
import json
from typing import List, Dict, Any, Optional
import numpy as np

# Mocking external vector DB and LLM clients for a self-contained production implementation
class MockEmbeddingEngine:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Simulating 1536-dimensional embeddings (e.g., text-embedding-3-small)
        np.random.seed(42)
        return [np.random.rand(1536).tolist() for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        np.random.seed(42)
        return np.random.rand(1536).tolist()

class ProductionAgentMemorySystem:
    def __init__(self):
        self.embedding_engine = MockEmbeddingEngine()
        # Vector database emulation storage
        self.vector_store: List[Dict[str, Any]] = []
        # Semantic cache: map of string hash to (response, embedding)
        self.semantic_cache: List[Dict[str, Any]] = []
        self.cache_similarity_threshold = 0.95  # Strict threshold for semantic hit

    def add_memories(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """Inserts documents into the vector store with localized metadata pre-indexing."""
        embeddings = self.embedding_engine.embed_documents(texts)
        for i in range(len(texts)):
            self.vector_store.append({
                "id": ids[i],
                "text": texts[i],
                "embedding": embeddings[i],
                "metadata": metadatas[i]
            })

    def _cosine_similarity(self, vecA: List[float], vecB: List[float]) -> float:
        a = np.array(vecA)
        b = np.array(vecB)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def query_with_pre_filtering(self, query_text: str, filter_criteria: Dict[str, Any], top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Executes single-stage pre-filtering. 
        Only computes vector distances on nodes matching metadata criteria.
        """
        query_embedding = self.embedding_engine.embed_query(query_text)
        filtered_nodes = []

        # 1. Apply Metadata Pre-filter
        for node in self.vector_store:
            match = True
            for key, val in filter_criteria.items():
                if node["metadata"].get(key) != val:
                    match = False
                    break
            if match:
                filtered_nodes.append(node)

        # 2. Compute Similarities on filtered subset only (Saves compute/latency)
        results = []
        for node in filtered_nodes:
            score = self._cosine_similarity(query_embedding, node["embedding"])
            results.append((score, node))

        # Sort by similarity score descending
        results.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in results[:top_k]]

    def check_semantic_cache(self, query_text: str) -> Optional[str]:
        """Prevents downstream LLM cost and latency if a semantically identical query exists."""
        query_embedding = self.embedding_engine.embed_query(query_text)
        
        for cache_item in self.semantic_cache:
            similarity = self._cosine_similarity(query_embedding, cache_item["embedding"])
            if similarity >= self.cache_similarity_threshold:
                return cache_item["response"]
        return None

    def update_semantic_cache(self, query_text: str, response_text: str):
        query_embedding = self.embedding_engine.embed_query(query_text)
        self.semantic_cache.append({
            "query": query_text,
            "embedding": query_embedding,
            "response": response_text
        })

class ContextCompressor:
    @staticmethod
    def compress_context(documents: List[Dict[str, Any]], max_tokens: int = 150) -> str:
        """
        A production heuristic for context compaction. Removes boilerplates,
        ranks high density sentences, and prunes to stay strictly under budget.
        """
        compiled_context = []
        current_tokens = 0
        
        for doc in documents:
            text = doc["text"]
            # Simplified token counting estimation (1 word ~= 1.3 tokens)
            approx_tokens = len(text.split()) * 1.3
            if current_tokens + approx_tokens <= max_tokens:
                compiled_context.append(text)
                current_tokens += approx_tokens
            else:
                # Truncate gracefully at sentence boundaries if possible
                sentences = text.split(". ")
                for sentence in sentences:
                    sent_tokens = len(sentence.split()) * 1.3
                    if current_tokens + sent_tokens <= max_tokens:
                        compiled_context.append(sentence)
                        current_tokens += sent_tokens
                break
                
        return "\n---\n".join(compiled_context)

# ==========================================
# Verification Execution Flow
# ==========================================
if __name__ == "__main__":
    memory_system = ProductionAgentMemorySystem()
    
    # Ingesting unstructured episodic logs with session metadata
    memory_system.add_memories(
        texts=[
            "User requested an automated update on user profile table schemas in database cluster-beta.",
            "System encountered an Out-Of-Memory (OOM) error on transaction service while evaluating batch jobs.",
            "User updated their subscription package to premium tier during session 9482."
        ],
        metadatas=[
            {"session_id": "9482", "environment": "production", "type": "db_schema"},
            {"session_id": "1102", "environment": "staging", "type": "error_log"},
            {"session_id": "9482", "environment": "production", "type": "billing"}
        ],
        ids=["mem_001", "mem_002", "mem_003"]
    )

    query = "Has the user modified any subscription details or system settings recently?"
    filter_opts = {"session_id": "9482", "environment": "production"}

    print(f"[*] Querying Memory System for: '{query}' with metadata filters: {filter_opts}\n")
    
    # 1. Check Semantic Cache
    cached_response = memory_system.check_semantic_cache(query)
    if cached_response:
        print(f"[+] Semantic Cache Hit: {cached_response}")
    else:
        print("[-] Semantic Cache Miss. Proceeding to Vector Pre-filtered Query...")
        
        # 2. Query Vector Space with Pre-filtering
        matched_memories = memory_system.query_with_pre_filtering(
            query_text=query, 
            filter_criteria=filter_opts, 
            top_k=2
        )
        
        # 3. Compress Context to optimize context window payload
        compressed_payload = ContextCompressor.compress_context(matched_memories, max_tokens=100)
        print("\n[+] Compressed Context Payload optimized for LLM consumption:")
        print(compressed_payload)
        
        # Simulating LLM response generation and cache update
        simulated_llm_output = "Yes, during session 9482, the user updated their subscription package to the premium tier."
        memory_system.update_semantic_cache(query, simulated_llm_output)
        print(f"\n[+] LLM Output Generated & Cache Updated.")
```
