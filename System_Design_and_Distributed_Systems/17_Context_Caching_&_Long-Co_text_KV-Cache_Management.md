# Module 17: Context Caching & Long-Context KV-Cache Management

With modern Large Language Models (LLMs) supporting context windows scaling from $100\text{k}$ to over $2\text{M}+$ tokens, managing the High-Bandwidth Memory (HBM) footprint of the Key-Value (KV) cache during inference has become a primary infrastructure bottleneck. Unoptimized KV-cache allocations lead to out-of-memory (OOM) faults, severe GPU underutilization, and high per-token latency.

This module covers long-context KV-cache memory dynamics, dynamic chunked prefilling, prefix/context caching mechanisms, sliding-window and token eviction algorithms, mathematical formulations for memory estimation, and a production-grade KV-Cache Manager implementation in Python.

---

## 1. Theoretical Foundations

### 1.1 Long-Context Memory Dynamics & Caching Paradigms

* **The KV-Cache Bottleneck**:
  * During the autoregressive generation (decoding) phase, LLMs compute Key ($K$) and Value ($V$) projections for every token at each transformer layer to avoid recomputing past context ($O(N^2)$ operation reduced to $O(N)$).
  * For long-context workloads ($N > 100\text{k}$ tokens), the KV-cache rapidly eclipses the model weights in memory size, severely restricting the maximum concurrent batch size.

* **Prefix & Context Caching**:
  * **Shared Prefix Reuse**: Identical system prompts, documentation contexts, or few-shot examples across requests are pre-computed once, retained in GPU/CPU memory, and linked via structural hashes.
  * **Radix Tree / Trie Indexing**: Request contexts are represented as sequence trees where overlapping prefix nodes are shared across dynamic client sessions without copying raw memory tensors.

* **Chunked Prefill & Offloading**:
  * Breaks massive prompt prefill requests into manageable token chunks ($B_{\mathtt{chunk}} \approx 512 - 2048$), interleaving prefill execution with active decoding iterations to prevent latency spikes for short requests.
 
  * ---

### 1.2 Mathematical Foundations

#### 1. Total KV-Cache Memory Calculation Formula
For a Transformer model with $L$ layers, $H_{\mathtt{kv}}$ attention key-value heads, head dimension $D_{\mathtt{head}}$, sequence length $N$, batch size $B$, and numerical precision bytes $P$ (e.g., $P = 2$ for FP16/BF16, $P = 1$ for INT8/FP8):

$$\text{Memory}_{\mathtt{KV}}(N, B) = 2 \cdot B \cdot L \cdot H_{\mathtt{kv}} \cdot D_{\mathtt{head}} \cdot N \cdot P \quad \text{(Bytes)}$$

*Example*: For Llama-3 70B ($L = 80$, $H_{\mathtt{kv}} = 8$, $D_{\mathtt{head}} = 128$) running FP16 ($P = 2$) with sequence length $N = 128\text{k}$ tokens and batch size $B = 1$:

$$\text{Memory}_{\mathtt{KV}} = 2 \cdot 1 \cdot 80 \cdot 8 \cdot 128 \cdot 131,072 \cdot 2 \approx 42.95 \quad \text{GB}$$

#### 2. KV-Cache Compression Ratio via Quantization
When applying FP8/INT4 quantization to $K$ and $V$ tensors using scale factor $\gamma \in (0, 1]$:

$$R_{\mathtt{compression}} = \frac{\text{Bytes}_{\mathtt{uncompressed}}}{\text{Bytes}_{\mathtt{quantized}}} = \frac{P_{\mathtt{FP16}}}{P_{\mathtt{quantized}} + \frac{S_{\mathtt{metadata}}}{D_{\mathtt{head}}}}$$

Where $S_{\mathtt{metadata}}$ is the per-block quantization scale header size in bytes.

#### 3. Prefix Cache Hit Ratio & Effective Decoding Latency
Given query prompt length $L_{\mathtt{prompt}}$, prefix cache hit fraction $\alpha \in [0, 1]$, compute time per prefill token $t_{\mathtt{prefill}}$, and decode step time $t_{\mathtt{decode}}$:

$$T_{\mathtt{total\_latency}} = (1 - \alpha) \cdot L_{\mathtt{prompt}} \cdot t_{\mathtt{prefill}} + N_{\mathtt{generated\_tokens}} \cdot t_{\mathtt{decode}}$$

---

## 2. KV-Cache Management Strategies Comparison

| Strategy | Memory Overhead | Cache Hit Potential | Latency Reduction | Complexity |
| :--- | :--- | :--- | :--- | :--- |
| **Naive Allocation (Contiguous)** | Extremely High (High fragmentation) | $0\%$ (No reuse) | Baseline ($1\times$) | Low |
| **Paged Attention (vLLM style)** | Zero External Fragmentation | Low-Moderate | $1.5\times - 2\times$ throughput | Medium |
| **Radix Tree Context Caching** | Low (Shared prefix nodes) | Very High ($80\%+$ for agent prompts) | $3\times - 10\times$ prefill reduction | High |
| **HBM-to-Host Host RAM Offloading**| Low GPU HBM, High Host RAM | Moderate | Tradeoff (PCIe transfer overhead) | High |

---

## 3. Production Context KV-Cache Engine Implementation

This Python module implements a **Radix-Tree-Based Prefix KV-Cache Manager** featuring prompt token prefix matching, dynamic node eviction (LRU policy), and memory block allocation tracking.

### Prerequisites

```bash
pip install numpy pydantic
```
### Python Implementation (kv_cache_manager.py)
```python
import time
import hashlib
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel


# -------------------------------------------------------------------
# 1. RADIX NODE STRUCTURE FOR PREFIX MATCHING
# -------------------------------------------------------------------
class CacheBlock(BaseModel):
    block_id: int
    size_tokens: int
    is_allocated: bool = True


class RadixNode:
    def __init__(self, prefix_tokens: List[int], parent: Optional['RadixNode'] = None):
        self.prefix_tokens: List[int] = prefix_tokens
        self.parent: Optional['RadixNode'] = parent
        self.children: Dict[int, 'RadixNode'] = {}  # maps first token of child prefix to node
        self.last_accessed: float = time.time()
        self.ref_count: int = 1
        self.allocated_blocks: List[int] = []

    @property
    def key_hash(self) -> str:
        return hashlib.sha256(bytes(self.prefix_tokens)).hexdigest()[:16]


# -------------------------------------------------------------------
# 2. RADIX TREE CONTEXT CACHE MANAGER
# -------------------------------------------------------------------
class RadixKVCacheManager:
    def __init__(self, total_gpu_blocks: int = 100, block_size: int = 16):
        self.total_gpu_blocks = total_gpu_blocks
        self.block_size = block_size
        self.free_blocks: List[int] = list(range(total_gpu_blocks))
        self.used_blocks: Dict[int, CacheBlock] = {}
        
        # Root node containing empty prefix
        self.root = RadixNode(prefix_tokens=[])

    def _allocate_blocks(self, num_tokens: int) -> List[int]:
        needed_blocks = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < needed_blocks:
            self._evict_lru_nodes(needed_blocks)

        if len(self.free_blocks) < needed_blocks:
            raise MemoryError("Out of GPU HBM Block Memory for KV Cache!")

        allocated = []
        for _ in range(needed_blocks):
            b_id = self.free_blocks.pop(0)
            self.used_blocks[b_id] = CacheBlock(block_id=b_id, size_tokens=self.block_size)
            allocated.append(b_id)
        return allocated

    def _evict_lru_nodes(self, blocks_needed: int):
        """Evicts unreferenced leaf nodes based on Least Recently Used (LRU) policy."""
        candidates: List[RadixNode] = []
        self._collect_leaf_nodes(self.root, candidates)
        
        # Sort candidate leaf nodes by last accessed timestamp
        candidates.sort(key=lambda n: n.last_accessed)

        for node in candidates:
            if len(self.free_blocks) >= blocks_needed:
                break
            if node.ref_count <= 0 and node.parent is not None:
                # Reclaim blocks
                for b_id in node.allocated_blocks:
                    del self.used_blocks[b_id]
                    self.free_blocks.append(b_id)
                
                # Detach from parent
                first_token = node.prefix_tokens[0]
                del node.parent.children[first_token]
                print(f"[CACHE EVICTION] Evicted LRU Node with prefix len {len(node.prefix_tokens)}.")

    def _collect_leaf_nodes(self, current: RadixNode, leaves: List[RadixNode]):
        if not current.children:
            if current != self.root:
                leaves.append(current)
            return
        for child in current.children.values():
            self._collect_leaf_nodes(child, leaves)

    def match_prefix(self, token_ids: List[int]) -> Tuple[List[int], List[int], RadixNode]:
        """Matches incoming tokens against cached Radix Tree nodes."""
        curr = self.root
        curr.last_accessed = time.time()
        matched_tokens: List[int] = []
        idx = 0

        while idx < len(token_ids):
            first_tok = token_ids[idx]
            if first_tok not in curr.children:
                break
            
            child = curr.children[first_tok]
            # Check overlap length
            p_len = len(child.prefix_tokens)
            if token_ids[idx:idx + p_len] == child.prefix_tokens:
                matched_tokens.extend(child.prefix_tokens)
                curr = child
                curr.last_accessed = time.time()
                idx += p_len
            else:
                break

        unmatched_tokens = token_ids[idx:]
        return matched_tokens, unmatched_tokens, curr

    def insert_and_cache(self, token_ids: List[int]) -> Dict[str, Any]:
        matched, unmatched, parent_node = self.match_prefix(token_ids)
        
        hit_ratio = len(matched) / len(token_ids) if token_ids else 0.0

        if unmatched:
            # Allocate KV-cache blocks for unmatched tokens only
            allocated_b = self._allocate_blocks(len(unmatched))
            first_unmatched = unmatched[0]
            new_node = RadixNode(prefix_tokens=unmatched, parent=parent_node)
            new_node.allocated_blocks = allocated_b
            parent_node.children[first_unmatched] = new_node
            new_node.ref_count = 0  # Release lock after insertion

        return {
            "total_tokens": len(token_ids),
            "cached_prefix_tokens": len(matched),
            "newly_computed_tokens": len(unmatched),
            "prefix_hit_ratio": round(hit_ratio, 4),
            "remaining_free_blocks": len(self.free_blocks)
        }


# -------------------------------------------------------------------
# SIMULATION / VERIFICATION RUNNER
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("--- 1. Initializing Radix KV-Cache Engine (100 Blocks, Block Size=16) ---")
    cache_mgr = RadixKVCacheManager(total_gpu_blocks=100, block_size=16)

    # Simulated System Prompt Tokens (Common System Context)
    system_prompt_tokens = [101, 2054, 2003, 1037, 3899, 2000, 4521] + [500] * 32  # 39 tokens total

    print("\n--- 2. Request 1: Executing Session with Base System Prompt ---")
    req1_result = cache_mgr.insert_and_cache(system_prompt_tokens + [1001, 1002, 1003])
    print(f"Request 1 Metrics: {req1_result}")

    print("\n--- 3. Request 2: Executing Subsequence with Identical System Prompt Prefix ---")
    req2_tokens = system_prompt_tokens + [2001, 2002, 2003, 2004]
    req2_result = cache_mgr.insert_and_cache(req2_tokens)
    print(f"Request 2 Metrics: {req2_result}")
    print(f"-> Prefix Cache Hit Ratio: {req2_result['prefix_hit_ratio'] * 100:.2f}% (Skipped recomputing shared system tokens!)")

    print("\n--- 4. Memory Footprint Snapshot ---")
    print(f"Total Free GPU Memory Blocks Remaining: {len(cache_mgr.free_blocks)} / 100")
```

## 4. Operational Best Practices

* Fixed Token Page Boundaries: Use fixed block sizes (e.g., $16$ or $32$ tokens per block) with virtual block mapping (PagedAttention) to eliminate external memory fragmentation completely.
* Hierarchical Cache Offloading: Implement a two-tiered cache architecture where cold prefix KV-tensors are swapped from GPU HBM to Host System RAM (via high-speed PCIe) rather than discarded.
* Normalize System Prompts: Standardize formatting, ordering, and whitespace in agent system prompts to maximize exact prefix token matching across different API sessions.
