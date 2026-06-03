# 22. Memory-Saving Compute Optimizations: FlashAttention and KV-Caching

While scaling the parameter count of Transformer architectures improves expressive capacity, standard implementation designs hit severe performance bottlenecks during training and inference. These bottlenecks are primarily caused by hardware memory access speeds rather than raw computational arithmetic capacity. This module details two industry-standard algorithmic engineering solutions designed to minimize memory input/output (I/O) bottlenecks.

---

## 1. The Core Hardware Context: Compute vs. Memory Bound

Modern deep learning hardware (like NVIDIA H100 or A100 GPUs) splits memory architecture into two core spaces:
* **High-Bandwidth Memory (HBM):** Large capacity space (e.g., 40GB–80GB) but relatively slow data transfer speeds.
* **SRAM (Static Random-Access Memory):** Blazing fast access speeds located directly next to the streaming multiprocessors, but highly constrained in capacity size (e.g., ~20MB–50MB).

An operation is **Compute-Bound** if performance is limited by how many floating-point math arithmetic operations (FLOPs) the processor can execute per second (e.g., dense matrix multiplications). It is **Memory-Bound** if execution speed is bottlenecks by data transfer times back and forth between HBM and SRAM (e.g., Softmax, LayerNorm, and Dropout activations).

---

## 2. FlashAttention Mechanics

Introduced by Dao et al. (2022), **FlashAttention** is a hardware-aware exact attention algorithm. Standard attention reads and writes the intermediate $O(T^2)$ attention matrix back and forth to slow HBM multiple times during Softmax and Dropout steps, creating a massive memory bottleneck.
FlashAttention cuts down HBM access operations from $O(T^2)$ to $O(T)$ by utilizing two mathematical core strategies:

### A. Tiling (Block Online Processing)
The algorithm loads inputs into SRAM in smaller blocks (tiles), processes attention locally within those localized tiles, and discards the intermediate weights before moving to the next block.

### B. Online Softmax Tracking
Standard softmax requires seeing an entire row vector to compute the normalizing denominator sum. To execute softmax block-by-block without global sequence views, FlashAttention tracks rescaling coefficients dynamically. For two split row vector segments $x^{(1)}$ and $x^{(2)}$:

$$\max(x) = m = \max(m^{(1)}, m^{(2)})$$
$$f(x) = e^{x - m}$$
$$\sum e^{x - m} = d = e^{m^{(1)} - m} \cdot d^{(1)} + e^{m^{(2)} - m} \cdot d^{(2)}$$

This allows the system to scale and adjust the running intermediate attention output matrix block dynamically as new token tiles arrive without ever storing the full global matrix.

---

## 3. Inference Acceleration: Key-Value (KV) Caching

During autoregressive decoding (text generation), an LLM generates tokens sequentially one-by-one. Each new token requires evaluating attention over every single token that came before it.

### The Redundancy Problem
Without caching, at generation step $t$, the model recomputes the Query, Key, and Value projection vectors for all historical tokens from steps $1$ to $t-1$, even though those historical key/value vectors are completely static. This causes inference complexity to scale quadratically at $O(T^2)$ for simple generation tasks.

### The KV-Cache Solution
**KV-Caching** stores the calculated Key and Value matrices of historical tokens in VRAM during the initial prompt processing pass (Prefill phase). During subsequent generation steps (Decoding phase), the model only computes $Q, K, V$ projections for the single *newly generated* token, appending the new $K$ and $V$ onto the historical cache vectors:

$$K_{\text{cached}} = \begin{bmatrix} K_{\text{past}} \\ K_{\text{new}} \end{bmatrix}, \quad V_{\text{cached}} = \begin{bmatrix} V_{\text{past}} \\ V_{\text{new}} \end{bmatrix}$$

This drops the incremental execution step cost down to a linear $O(T)$ scale profile.

---

## 4. Implementation in Python (PyTorch)

This script demonstrates a text generation decoding loop comparing standard forward evaluation against **KV-Cached sequence updates** to highlight performance engineering layouts.

```python
import torch
import torch.nn as nn
import time

class CausalAttentionWithKVCache(nn.Module):
    """
    A multi-head causal self-attention module supporting standard processing
    and incremental KV-Cache decoding paths.
    """
    def __init__(self, embed_dim=256, num_heads=4):
        super(CausalAttentionWithKVCache, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, kv_cache=None):
        """
        Args:
            x: Input token hidden states of shape [Batch, SeqLen, EmbedDim]
            kv_cache: Tuple of (past_k, past_v) tensors if active decoding
        """
        B, T, C = x.shape
        
        # 1. Project inputs to multi-head structures
        # Output shapes: [Batch, Heads, SeqLen, HeadDim]
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. Process KV-Cache allocation routes
        if kv_cache is not None:
            past_k, past_v = kv_cache
            # Concatenate current step token matrices onto historical caches along the SeqLen axis
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
            
        current_kv_state = (k.detach(), v.detach())
        total_seq_len = k.size(-2)
        
        # 3. Standard Scaled Dot-Product Attention calculation
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply strict causal masking if processing multiple sequence context tokens simultaneously
        if T > 1:
            mask = torch.triu(torch.full((T, total_seq_len), float('-inf'), device=x.device), diagonal=1)
            attn_scores = attn_scores + mask
            
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v) # [Batch, Heads, SeqLen, HeadDim]
        
        # 4. Collapse heads back to standard dimension layouts
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(context), current_kv_state


# Verification execution profiling pass
if __name__ == "__main__":
    torch.manual_seed(42)
    B, C = 1, 256  # Batch=1, Embedding dimension depth=256
    layer = CausalAttentionWithKVCache(embed_dim=C)
    
    # --- PHASE 1: PREFILL PHASE (Process input prompt) ---
    prompt_tokens = torch.randn(B, 10, C) # Prompt of 10 structural context tokens
    output, cache = layer(prompt_tokens)
    print(f"Prefill Phase Complete. Initial Key Cache Shape: {cache[0].shape}")
    
    # --- PHASE 2: NAIVE DECODING (Without caching, recomputes entire context) ---
    print("\n--- Running Naive vs. KV-Cached Generation Step ---")
    next_token_input = torch.randn(B, 1, C) # A single new token is generated
    
    # Naive decoding requires passing everything back down through the model layers
    full_naive_context = torch.cat([prompt_tokens, next_token_input], dim=1)
    
    t0 = time.perf_counter()
    naive_out, _ = layer(full_naive_context)
    t_naive = (time.perf_counter() - t0) * 1000
    
    # --- PHASE 3: CACHED DECODING (Only processes the 1 new token) ---
    t0 = time.perf_counter()
    cached_out, new_cache = layer(next_token_input, kv_cache=cache)
    t_cached = (time.perf_counter() - t0) * 1000
    
    print(f"Naive Pass Token Target Output Dimension: {naive_out[:, -1:, :].shape} | Time: {t_naive:.4f}ms")
    print(f"Cached Pass Token Target Output Dimension: {cached_out.shape} | Time: {t_cached:.4f}ms")
    print(f"Updated Next-Step Key Cache Shape:        {new_cache[0].shape}")
```
