# Advanced Memory Architectures: Long-Term, Short-Term, and Episodic Memory

In standard conversational LLM applications, memory is often reduced to a flat list of recent messages (a sliding context window). However, autonomous agents operating in production environments require advanced memory architectures that mimic human cognitive structures. This module covers the theoretical foundations, mathematical formulations, and engineering implementations of multi-layered agent memory.

---

## 1. Architectural Foundations of Agent Memory

An enterprise-grade agent relies on three interconnected memory tiers to maintain context, build relationships, and self-correct over extended timelines.

### A. Short-Term Memory (Context Buffer)
* **Definition:** The immediate operational workspace of the agent, typically storing the current session's conversation history.
* **Limitation:** Strictly bound by the LLM’s context window limits and subject to "lost in the middle" attention degradation.

### B. Long-Term Semantic Memory (World Knowledge & Profile)
* **Definition:** A permanent repository of facts, user preferences, and generalized concepts extracted across multiple distinct sessions.
* **Mechanism:** Driven by vector embeddings and structured databases to allow semantic retrieval of historic facts.

### C. Episodic Memory (Experience Sequences)
* **Definition:** Captures *sequences of events* or execution paths ("episodes") of past tasks, allowing the agent to remember how a complex problem was solved or why an action failed.
* **Mechanism:** Graph structures or timed logs that preserve temporal dependencies and execution trajectories.

---

## 2. Mathematical Formalization: Recency, Relevance, and Decay

To prevent memory retrieval from flooding the LLM context with irrelevant history, advanced architectures score and rank memories using a multi-factor formula combining **Semantic Relevance**, **Recency (Time Decay)**, and **Importance**.

The overall score $S$ for a memory item $m$ given a current query $q$ is defined as:

$$S(m, q) = w_{\text{rel}} \cdot S_{\text{rel}}(m, q) + w_{\text{imp}} \cdot I(m) + w_{\text{rec}} \cdot D(t)$$

Where:
* $S_{\text{rel}}(m, q)$ is the cosine similarity between the query embedding and the memory embedding.
* $I(m)$ is an importance score (typically scaled $[0, 1]$) assigned by an LLM evaluator when the memory is saved.
* $D(t)$ is the **Ebbinghaus Forgetting Curve** model for time decay:

$$D(t) = e^{-\lambda \cdot \Delta t}$$

* $\Delta t = t_{\text{current}} - t_{\text{created}}$ (the time elapsed since the memory was captured).
* $\lambda$ is the decay constant determining how rapidly the memory fades.
* $w_{\text{rel}}, w_{\text{imp}}, w_{\text{rec}}$ are weights normalized such that $\sum w = 1$.

---

## 3. Production Implementation: Hybrid Memory Engine

Below is a complete Python implementation using `ChromaDB` for semantic storage, tracking memory creation time to compute exponential recency decay.

```python
import time
import math
import numpy as np
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

class AdvancedAgentMemory:
    def __init__(self, db_path: str = "./agent_memory_db", decay_rate: float = 0.05):
        """
        Initializes the hybrid memory engine with vector search and temporal decay tracking.
        """
        self.client = chromadb.PersistentClient(path=db_path)
        self.emb_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # Initialize or fetch the memory collection
        self.collection = self.client.get_or_create_collection(
            name="agent_episodic_memory",
            embedding_function=self.emb_fn
        )
        self.decay_rate = decay_rate  # Lambda value for exponential decay

    def store_memory(self, content: str, importance: float, memory_type: str = "semantic") -> None:
        """
        Persists a memory slice into long-term storage along with timestamp and importance.
        """
        memory_id = f"mem_{int(time.time() * 1000)}"
        current_time = time.time()
        
        self.collection.add(
            documents=[content],
            metadatas=[{
                "timestamp": current_time,
                "importance": importance,
                "type": memory_type
            }],
            ids=[memory_id]
        )

    def _calculate_decay(self, creation_timestamp: float) -> float:
        """
        Computes the exponential time decay factor D(t) = e^(-lambda * delta_t).
        Delta_t is converted to minutes for granular decay evaluation.
        """
        delta_t = (time.time() - creation_timestamp) / 60.0  # Time delta in minutes
        return math.exp(-self.decay_rate * delta_t)

    def retrieve_memories(self, query: str, limit: int = 3, w_rel: float = 0.5, w_imp: float = 0.2, w_rec: float = 0.3) -> List[Dict[str, Any]]:
        """
        Retrieves memories scored by a blend of semantic relevance, importance, and recency decay.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=limit * 2 # Fetch excess candidates to rank with decay
        )
        
        if not results['documents'] or len(results['documents'][0]) == 0:
            return []

        ranked_memories = []
        
        # ChromaDB distances return squared L2 or cosine distance; convert to similarity
        documents = results['documents'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]

        for idx in range(len(documents)):
            content = documents[idx]
            meta = metadatas[idx]
            
            # Normalize semantic similarity from distance
            semantic_sim = 1.0 / (1.0 + distances[idx])
            
            # Extract factors
            importance = meta.get("importance", 0.5)
            timestamp = meta.get("timestamp", time.time())
            
            # Compute temporal decay
            recency_factor = self._calculate_decay(timestamp)
            
            # Final Hybrid Scoring Formula
            final_score = (w_rel * semantic_sim) + (w_imp * importance) + (w_rec * recency_factor)
            
            ranked_memories.append({
                "content": content,
                "score": final_score,
                "type": meta.get("type"),
                "recency": recency_factor
            })
            
        # Re-sort based on the customized mathematical score
        ranked_memories.sort(key=lambda x: x['score'], reverse=True)
        return ranked_memories[:limit]

# --- Verification & Execution Loop ---
if __name__ == "__main__":
    memory_engine = AdvancedAgentMemory(decay_rate=0.1)
    
    print("[1] Storing historical contextual milestones...")
    # Past historical interaction (low recency now, high importance)
    memory_engine.store_memory(
        content="User stated preferred deployment cloud environment is AWS with Terraform orchestration.", 
        importance=0.9, 
        memory_type="semantic"
    )
    
    # Casual chatter (low importance, high recency if written right before query)
    memory_engine.store_memory(
        content="User mentioned they are having coffee while writing code.", 
        importance=0.1, 
        memory_type="episodic"
    )
    
    # Simulating a slight delay to trigger decay differences
    print("Simulating execution step progression...")
    time.sleep(2)
    
    # Querying the memory layers
    query_str = "Where should I configure the production deployment script infrastructure?"
    retrieved = memory_engine.retrieve_memories(query=query_str, limit=1)
    
    print(f"\nQuery: '{query_str}'")
    for mem in retrieved:
        print(f"-> Retained Memory: '{mem['content']}' (Blended Score: {mem['score']:.4f}, Recency Factor: {mem['recency']:.4f})")
```
