# Module 13: RAG Architecture, Vector DBs, and Evaluation

While base Large Language Models (LLMs) possess vast parametric knowledge, they suffer from knowledge cutoff dates, hallucination risks, and an inability to access private enterprise data. **Retrieval-Augmented Generation (RAG)** grounds model outputs in external, authoritative knowledge bases dynamically retrieved at inference time.

This module covers advanced chunking mechanics, hybrid dense-sparse retrieval, mathematical distance metrics, production vector database operations with **Qdrant**, and automated RAG evaluation using **Ragas**.

---

## 1. Theoretical Foundations

### 1.1 RAG System Lifecycle
* **Ingestion Pipeline**: Document extraction, semantic chunking (overlapping strategy), embedding generation, and vector database indexing.
* **Retrieval Phase**: Query expansion, hybrid vector search (Dense Semantic + Sparse BM25 keyword matching), re-ranking (Cross-Encoders), and context window packing.
* **Generation & Synthesis**: Augmenting the system prompt with retrieved context blocks and executing guarded LLM generation.
* ---

### 1.2 Mathematical Foundations of Vector Search & Retrieval Metrics

#### Vector Distance Metrics

##### Cosine Similarity
Measures the angle between query embedding $\mathbf{q}$ and document embedding $\mathbf{d}$, normalized for length:

$$\text{Sim}_{\text{Cosine}}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\Vert{}\mathbf{q}\Vert{} \Vert{}\mathbf{d}\Vert{}} = \frac{\sum_{i=1}^{n} q_i d_i}{\sqrt{\sum_{i=1}^{n} q_i^2} \sqrt{\sum_{i=1}^{n} d_i^2}}$$

##### Dot Product (Inner Product)
Used when vector embeddings are pre-normalized to unit length ($\Vert{}\mathbf{q}\Vert{} = \Vert{}\mathbf{d}\Vert{} = 1$):

$$\text{Sim}_{\text{Dot}}(\mathbf{q}, \mathbf{d}) = \mathbf{q}^T \mathbf{d} = \sum_{i=1}^{n} q_i d_i$$

---

#### Key RAG Evaluation Triad Metrics (Ragas)

1. **Context Relevance**: Measures how much of the retrieved context $C$ is strictly necessary to answer query $Q$:
   $$\text{Context Relevance} = \frac{\vert{}C_{\text{relevant}}\vert{}}{Total(C)}$$

2. **Faithfulness**: Quantifies if the output answer $A$ is strictly grounded in the retrieved context $C$ (hallucination detector):
   $$\text{Faithfulness} = \frac{\text{Number of Claims in } A \text{ Supported by } C}{\text{Total Claims in } A}$$

3. **Answer Relevance**:Measures how well the output response $A$ addresses the original query $Q$:
$$\text{Answer Relevance} = \frac{1}{N} \sum_{i=1}^{N} \text{Sim}_{\text{Cosine}}(E_q, E_{g, i})$$
*Where $E_q$ is the original query embedding and $E_{g, i}$ are the embeddings of $N$ synthetic questions generated back from answer $A$.*
---

## 2. Production RAG & Hybrid Vector Database Implementation (Qdrant)

This Python implementation demonstrates dynamic chunking, vector indexing with `Qdrant`, and Cross-Encoder re-ranking.

### Prerequisites

```bash
pip install qdrant-client sentence-transformers langchain-text-splitters
```
### RAG Pipeline (rag_vector_pipeline.py)
```python
import uuid
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Document Extraction & Semantic Chunking
RAW_DOCUMENTS = [
    "Production MLOps requires automated model retraining pipelines triggered by data drift. Population Stability Index (PSI) values above 0.25 indicate severe feature drift.",
    "RAG architectures enhance LLMs by retrieving dynamic external context from vector databases like Qdrant, Milvus, and Pinecone, reducing hallucination rates.",
    "LLM FinOps focuses on token usage optimization, model routing, and semantic caching using Redis to reduce expensive API call costs."
]

def create_chunks(documents: List[str], chunk_size: int = 150, overlap: int = 30) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc))
    return chunks

# 2. Hybrid Retrieval Engine Setup
class RAGRetrievalEngine:
    def __init__(self, collection_name: str = "mlops_knowledge_base"):
        self.collection_name = collection_name
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Initialize in-memory Qdrant client
        self.qdrant = QdrantClient(":memory:")
        self._setup_collection()

    def _setup_collection(self):
        embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.qdrant.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
        )

    def index_chunks(self, chunks: List[str]):
        embeddings = self.encoder.encode(chunks)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload={"text": chunk}
            )
            for chunk, emb in zip(chunks, embeddings)
        ]
        self.qdrant.upsert(collection_name=self.collection_name, points=points)
        print(f" Successfully indexed {len(points)} document chunks into Qdrant.")

    def search_and_rerank(self, query: str, top_k: int = 5, rerank_top_n: int = 2) -> List[Dict]:
        # Step 1: Initial Vector Search
        query_vector = self.encoder.encode(query).tolist()
        search_results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        retrieved_texts = [hit.payload["text"] for hit in search_results]

        # Step 2: Cross-Encoder Re-Ranking
        pairs = [[query, doc] for doc in retrieved_texts]
        rerank_scores = self.reranker.predict(pairs)

        # Pair scores with documents and sort
        reranked_docs = sorted(
            zip(retrieved_texts, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [{"text": doc, "rerank_score": float(score)} for doc, score in reranked_docs[:rerank_top_n]]

# 3. Execution Pipeline
if __name__ == "__main__":
    chunks = create_chunks(RAW_DOCUMENTS)
    
    engine = RAGRetrievalEngine()
    engine.index_chunks(chunks)

    user_query = "How do vector databases help LLMs and what are some examples?"
    print(f"\nUser Query: '{user_query}'\n")

    results = engine.search_and_rerank(user_query)
    
    print("--- Top Re-Ranked Context Blocks ---")
    for i, res in enumerate(results, 1):
        print(f"Rank {i} (Score: {res['rerank_score']:.4f}): {res['text']}")
```
## 3. Automated RAG Evaluation Framework (Ragas)
Evaluating RAG performance requires automated measurement of context relevance and hallucination rates using Ragas.

### Prerequisites
```bash
pip install ragas langchain-openai datasets
```

### RAG Evaluation Runner (eval_rag_triad.py)
```python
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision

def run_ragas_evaluation():
    # Evaluation dataset consisting of Query, Ground Truth, Retrieved Contexts, and LLM Answers
    eval_data = {
        "question": [
            "What threshold of PSI indicates severe feature drift?",
            "What is the role of vector databases in RAG?"
        ],
        "contexts": [
            ["Population Stability Index (PSI) values above 0.25 indicate severe feature drift."],
            ["RAG architectures enhance LLMs by retrieving dynamic external context from vector databases like Qdrant."]
        ],
        "answer": [
            "A Population Stability Index (PSI) value exceeding 0.25 signifies severe feature drift.",
            "Vector databases store document embeddings and allow RAG architectures to retrieve dynamic context."
        ],
        "ground_truth": [
            "PSI values above 0.25 indicate severe feature drift.",
            "Vector databases store embeddings to retrieve context for LLMs to reduce hallucinations."
        ]
    }

    dataset = Dataset.from_dict(eval_data)

    print("--- Running Ragas Evaluation Triad ---")
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevance,
            context_precision
        ]
    )

    print("\nEvaluation Summary Results:")
    print(results)

if __name__ == "__main__":
    # Ensure OPENAI_API_KEY environment variable is set for Ragas LLM evaluation judges
    if "OPENAI_API_KEY" in os.environ:
        run_ragas_evaluation()
    else:
        print("Set OPENAI_API_KEY environment variable to execute Ragas evaluation suite.")
```
