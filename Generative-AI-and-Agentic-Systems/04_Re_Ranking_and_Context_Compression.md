# Re-Ranking and Context Compression

While initial retrieval stages (like dense vector search or BM25) are excellent at narrowing down a massive corpus of millions of documents to a small candidate pool (e.g., top-100), they operate under strict latency constraints that limit their contextual precision. To maximize the accuracy of the downstream Large Language Model without flooding its context window with noise, enterprise RAG pipelines employ a two-stage retrieval paradigm: **Stage 1 (Retrieval)** and **Stage 2 (Re-Ranking & Compression)**.

This document explores the architectural, mathematical, and algorithmic mechanics of Cross-Encoder reranking and token-level context compression.

---

## 1. Architectural Paradigms: Bi-Encoders vs. Cross-Encoders

To understand why re-ranking is mathematically superior for precision, we must contrast the underlying structural topologies of Bi-Encoders and Cross-Encoders.

### I. Bi-Encoders (Stage 1)
* **Mechanics:** The query and document are passed through the transformer network *independently*. The model outputs a single pooled vector representation for each. Similarity is calculated outside the network via a simple dot product or cosine distance.
* **Pro:** Massively scalable. Document vectors can be computed asynchronously beforehand and cached inside a vector database for sub-millisecond retrieval.
* **Con:** No token-to-token interaction across the query and the document during the forward pass. The semantic relationship is highly compressed, causing structural nuances to be lost.

### II. Cross-Encoders (Stage 2)
* **Mechanics:** The query and document are concatenated together into a single sequence, separated by a structural token delimiter (e.g., `[CLS] Query [SEP] Document [EOS]`), and fed into the transformer simultaneously.
* **Pro:** Exceptional precision. Every token in the query can perform a direct cross-attention calculation with every token in the document. The model dynamically flags conditional linguistic dependencies, matching nuances that static vector metrics miss.
* **Con:** Extremely computationally expensive. Because attention scales quadratically ($O(L^2)$), running a full forward pass over thousands of candidates at scale creates massive latency bottlenecks. Therefore, it is strictly reserved as a secondary processing step over a small candidate pool (e.g., the top-20 documents).

---

## 2. Context Compression and Contextual Filtering

Even after re-ranking, passing multiple long documents directly into an LLM context window causes severe operational degradations:
1.  **Monetary Cost:** High input token counts directly scale API runtime overhead.
2.  **Latency Spikes:** Massive input sequences prolong the compute-bound *Prefill Phase*, raising Time-To-First-Token (TTFT).
3.  **Accuracy Degradation:** Long contexts trigger the **Lost in the Middle** phenomenon, where the model fails to extract relevant details buried deep within long text payloads.

### Algorithmic Token Compression
To fix this, **Context Compression** isolates the specific sentences or token clusters inside a retrieved document that actually relate to the user query, slicing away the surrounding irrelevant paragraphs before formatting the final LLM prompt.

---

## 3. Mathematical Formulation of Cross-Encoder Loss

During the training phase of a sequence-classification Cross-Encoder, the network optimizes for binary cross-entropy over a balanced set of query-document pairs. Let $y_i \in \{0, 1\}$ be the ground-truth relevance label for a paired sequence, and $s(Q, D)$ be the raw logit output scalar emitted from the transformer's specialized pooling classification head (e.g., the `[CLS]` token linear projection):

$$\hat{y}_i = \sigma(s(Q, D)) = \frac{1}{1 + e^{-s(Q, D)}}$$

The optimization step minimizes the structural binary loss function:

$$\mathcal{L} = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

---

## 4. Production-Grade Implementation: Two-Stage Retrieval and Compression Pipeline

Below is a complete, self-contained Python architecture implementing a Stage-1 lexical search retriever, a Stage-2 Cross-Encoder semantic similarity ranker from scratch, and a token-density sentence compressor.

```python
import math
from typing import List, Dict, Tuple, Any

class TwoStageRetrievalEngine:
    def __init__(self, compression_threshold: float = 0.25):
        self.compression_threshold = compression_threshold
        self.documents: Dict[int, str] = {}

    def register_documents(self, corpus: Dict[int, str]):
        self.documents = corpus

    @staticmethod
    def _compute_jaccard_token_similarity(str1: str, str2: str) -> float:
        """
        Simulates high-precision keyword overlap intersections 
        acting as a lightweight Stage-1 retrieval function.
        """
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if union else 0.0

    @classmethod
    def _simulate_cross_encoder_forward_pass(cls, query: str, document: str) -> float:
        """
        Simulates the dense cross-token attention interaction of a Cross-Encoder.
        Computes conditional term proximity alignments from scratch.
        """
        q_tokens = query.lower().split()
        d_tokens = document.lower().split()
        
        score = 0.0
        # Simulating cross-attention token matrix tracking
        for q_t in q_tokens:
            if q_t in d_tokens:
                # Give higher scores if the matching terms appear close to each other
                first_idx = d_tokens.index(q_t)
                proximity_bonus = 1.0 / (1.0 + (first_idx * 0.01))
                score += proximity_bonus
                
        # Normalize by query token distribution constraints
        return score / len(q_tokens) if q_tokens else 0.0

    def compress_document(self, query: str, document: str) -> str:
        """
        Context Compression: Slices away irrelevant sentences 
        by keeping only those that pass the semantic relevance threshold.
        """
        # Simple sentence tokenizer boundary split simulation
        sentences = [s.strip() for s in document.split(".") if s.strip()]
        retained_segments = []
        
        for sentence in sentences:
            # Score individual sentence segments against the query
            sentence_score = self._simulate_cross_encoder_forward_pass(query, sentence)
            if sentence_score >= self.compression_threshold:
                retained_segments.append(sentence)
                
        # Reconstruct the optimized document chunk
        return ". ".join(retained_segments) + "." if retained_segments else ""

    def execute_pipeline(self, query: str, top_k_stage1: int = 3) -> List[Dict[str, Any]]:
        """Executes Stage-1 Lexical Retrieval -> Stage-2 Cross-Encoder Reranking -> Compression."""
        
        # --- STAGE 1: Broad Fast Filtering ---
        stage1_pool: List[Tuple[int, float]] = []
        for doc_id, text in self.documents.items():
            s1_score = self._compute_jaccard_token_similarity(query, text)
            if s1_score > 0.0:
                stage1_pool.append((doc_id, s1_score))
                
        # Sort and take top-K candidates
        stage1_pool.sort(key=lambda x: x[1], reverse=True)
        truncated_candidates = stage1_pool[:top_k_stage1]
        
        # --- STAGE 2: High-Precision Cross-Encoder Re-Ranking ---
        reranked_pool: List[Dict[str, Any]] = []
        for doc_id, _ in truncated_candidates:
            raw_text = self.documents[doc_id]
            cross_score = self._simulate_cross_encoder_forward_pass(query, raw_text)
            
            # --- CONTEXT COMPRESSION STAGE ---
            compressed_text = self.compress_document(query, raw_text)
            
            reranked_pool.append({
                "document_id": doc_id,
                "cross_encoder_score": round(cross_score, 4),
                "original_text": raw_text,
                "compressed_context": compressed_text
            })
            
        # Re-sort using the superior Cross-Encoder scores
        reranked_pool.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        return reranked_pool


# --- Pipeline Execution Verification Sandbox ---
if __name__ == "__main__":
    # Mock Data Store reflecting long enterprise document strings containing noisy information
    knowledge_base = {
        201: "Financial reports indicate standard operations. The company's net revenue grew by 12 percent using advanced cloud infrastructure setups. Operational costs remained completely static throughout Q3.",
        202: "The database configuration framework requires specific attention parameters. Security alert: system logs detected a critical token leakage exploit inside historical production storage files.",
        203: "Database clusters utilize advanced high availability setups. Distributed transaction pipelines are monitored daily to ensure zero dropped connections across system boundaries."
    }
    
    # Initialize Engine (Set sentence compression threshold to 0.15)
    engine = TwoStageRetrievalEngine(compression_threshold=0.15)
    engine.register_documents(knowledge_base)
    
    target_query = "Security alert critical token leakage inside database logs"
    
    # Execute the two-stage pipeline
    pipeline_results = engine.execute_pipeline(query=target_query, top_k_stage1=2)
    
    print("=== Two-Stage Retrieval & Context Compression Pipeline ===\n")
    print(f"Target Query: '{target_query}'\n")
    
    for rank, result in enumerate(pipeline_results, start=1):
        print(f"Rank {rank} -> Document ID: {result['document_id']} (Cross-Encoder Score: {result['cross_encoder_score']})")
        print(f"  [Original Raw Text]:  '{result['original_text']}'")
        print(f"  [Compressed Context]: '{result['compressed_context']}'")
        print("-" * 90)
```
