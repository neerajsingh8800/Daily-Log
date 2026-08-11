# Module 11: LLM FinOps, Cost Optimization, and Caching

As LLM applications transition to production, inference costs can scale exponentially with user traffic. Serving large language models—whether through commercial APIs (e.g., OpenAI, Anthropic) or self-hosted GPU clusters (vLLM, TGI)—requires a dedicated **FinOps strategy**. 

This module covers the core principles of token economy, cost modeling formulas, semantic caching architectures using **GPTCache** and **Redis**, dynamic prompt compression, and practical cost optimization techniques.

---

## 1. Theoretical Foundations

### 1.1 The Pillars of LLM FinOps
* **Token Optimization**: Reducing total input/output tokens through prompt pruning, context compression, and structured schemas.
* **Semantic Caching**: Serving identical or semantically similar queries directly from a low-latency cache to bypass expensive LLM forward passes.
* **Model Tiering & Routing**: Dynamically routing simple queries to smaller, cheaper models (e.g., GPT-4o-mini, Llama-3-8B) and complex reasoning tasks to larger models (e.g., GPT-4o, Claude 3.5 Sonnet).
* **GPU Utilization & Spot Instances**: For self-hosted LLMs, optimizing KV-cache memory allocation (PagedAttention) and utilizing spot instances with fault-tolerant fallbacks.

---

### 1.2 Mathematical Foundations of Cost Modeling

#### Total Inference Cost Formula
The cost $C$ for an LLM workload over $N$ queries is modeled as:

$$C = \sum_{i=1}^{N} \left( T_{in, i} \cdot P_{in} + T_{out, i} \cdot P_{out} \right)$$

Where:
* $T_{in, i}$ and $T_{out, i}$ represent the number of input (prompt) and output (completion) tokens for query $i$.
* $P_{in}$ and $P_{out}$ are the unit costs per token for input and output, respectively (typically $P_{out} \approx 3 \times \text{to } 4 \times P_{in}$).

#### Effective Cost with Caching & Routing
When introducing a semantic cache with cache hit rate $h \in [0, 1]$ and model routing across $M$ tiers:

$$C_{optimized} = (1 - h) \sum_{i=1}^{N} \left( T_{in, i} \cdot P_{in, m(i)} + T_{out, i} \cdot P_{out, m(i)} \right) + N \cdot C_{cache}$$

Where:
* $m(i)$ is the selected model tier for query $i$.
* $C_{cache}$ is the negligible lookup cost per request (e.g., vector database/Redis lookup).

#### Cache Hit Ratio (CHR) & Cost Reduction Percentage
$$\text{Cost Reduction (\%)} = \left( 1 - \frac{C_{optimized}}{C_{baseline}} \right) \times 100$$

---

## 2. Production Semantic Caching Implementation

Semantic caching evaluates similarity in embedding space rather than requiring exact string matches. If a new query $Q_{new}$ has a cosine similarity score with a cached query $Q_{cached}$ above a threshold $\tau$ (e.g., $\tau \ge 0.88$), the system returns the cached answer instantly.

### Prerequisites

```bash
pip install gptcache redis sentence-transformers openai
```

### Python Implementation (semantic_cache_manager.py)
``` python
import os
import time
from gptcache import Cache
from gptcache.adapter import openai as gptcache_openai
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

# 1. Initialize Semantic Cache System
def init_semantic_cache():
    """Builds an in-memory/Redis semantic cache with vector similarity lookup."""
    
    # Sentence Transformer for generating query embeddings
    onnx_embedding = Onnx()
    
    # SQLite storage for cached response metadata + Vector Index for embeddings
    cache_base = CacheBase('sqlite')
    vector_base = VectorBase('faiss', dimension=onnx_embedding.dimension)
    
    data_manager = get_data_manager(cache_base, vector_base)
    
    # Distance evaluation threshold (0 to 1 distance metric)
    similarity_eval = SearchDistanceEvaluation(max_distance=0.2)
    
    cache = Cache()
    cache.init(
        embedding_func=onnx_embedding.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=similarity_eval,
    )
    return cache

# 2. Query Execution Function with Timing
def query_llm_with_cache(prompt: str, cache_instance: Cache):
    start_time = time.time()
    
    # Wrap OpenAI API request through GPTCache adapter
    response = gptcache_openai.ChatCompletion.create(
        cache_obj=cache_instance,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    elapsed_time = time.time() - start_time
    content = response['choices'][0]['message']['content']
    return content, elapsed_time

# 3. Execution & Validation
if __name__ == "__main__":
    cache = init_semantic_cache()
    
    query1 = "What are the core advantages of using Docker containers in cloud deployments?"
    query2 = "Can you explain the main benefits of using Docker containers for cloud applications?"
    
    print("--- Executing First Query (Cache Miss Expected) ---")
    res1, time1 = query_llm_with_cache(query1, cache)
    print(f"Latency: {time1:.3f}s")
    print(f"Response Preview: {res1[:100]}...\n")

    print("--- Executing Semantically Similar Query (Cache Hit Expected) ---")
    res2, time2 = query_llm_with_cache(query2, cache)
    print(f"Latency: {time2:.3f}s")
    print(f"Response Preview: {res2[:100]}...\n")
    
    print(f"Speedup Factor: {time1 / max(time2, 0.001):.1f}x faster!")
```

## 3. Dynamic Model Routing & Prompt Compression

To minimize token usage, prompt compression removes redundant tokens from long system context windows without sacrificing downstream task accuracy.

### Dynamic Router & Token Pruner (cost_optimizer.py)
```python
import tiktoken

def count_tokens(text: str, model_name: str = "gpt-4o") -> int:
    """Accurately calculates token count using tiktoken."""
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

def dynamic_model_router(prompt: str, task_complexity: str = "low") -> str:
    """
    Routes query based on length and requested task complexity to minimize cost.
    """
    tokens = count_tokens(prompt)
    
    if task_complexity == "low" and tokens < 500:
        return "gpt-4o-mini"  # Low-cost fast model
    elif task_complexity == "high" or tokens > 3000:
        return "gpt-4o"       # Premium reasoning model
    else:
        return "gpt-4o-mini"

def prune_stop_words_and_whitespace(prompt: str) -> str:
    """Lightweight context compression to eliminate unnecessary whitespace and formatting tokens."""
    lines = [line.strip() for line in prompt.split('\n') if line.strip()]
    return " ".join(lines)

# Example Usage
if __name__ == "__main__":
    raw_prompt = """
    You are an expert software engineer.
    
    Please review the following code snippet and check for syntax errors.
    
    code = [1, 2, 3, 4]
    """
    
    compressed_prompt = prune_stop_words_and_whitespace(raw_prompt)
    original_tokens = count_tokens(raw_prompt)
    compressed_tokens = count_tokens(compressed_prompt)
    
    selected_model = dynamic_model_router(compressed_prompt, task_complexity="low")
    
    print(f"Original Tokens: {original_tokens}")
    print(f"Compressed Tokens: {compressed_tokens}")
    print(f"Token Savings: {((original_tokens - compressed_tokens) / original_tokens) * 100:.1f}%")
    print(f"Selected Model Endpoint: {selected_model}")
```
