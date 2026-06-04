# 30. Sparse Architectures and Inference Acceleration: MoE and Speculative Decoding

To maximize the performance of Large Language Models in production, scaling dense networks becomes economically and physically impractical. This module explores two advanced frontiers of modern LLM optimization: **Mixture of Experts (MoE)** (algorithmic sparsity during training/inference) and **Speculative Decoding** (speculative sampling frameworks to bypass autoregressive latency boundaries).

---

## 1. Sparse Mixture of Experts (MoE)

In a traditional dense Transformer, every single token activates 100% of the model parameters. A Sparse Mixture of Experts (MoE) architecture scales model capacity (total parameters) without increasing the compute cost per token by replacing the standard Feed-Forward Network (FFN) layer with a sparse MoE layer containing multiple independent "experts."
### A. The Routing (Gating) Mechanism
A parameterized routing network $G(x)$ takes a token embedding $x$ and computes a probability distribution over $E$ available expert networks. In a **Top-k MoE** setup (typically $k=2$), the router selectively sends the token to only the top $k$ highest-scoring experts:

$$G(x) = \text{Softmax}(\text{TopK}(x \cdot W_g, k))$$

Where $W_g$ represents the trainable gating weight matrix. If an expert is not in the Top-k, its gate value is set to $-\infty$ before the Softmax, reducing its output coefficient to $0$.

### B. The Sparse Output Equation
The final output of the MoE layer is the dynamically weighted sum of the activations computed exclusively by the selected top experts:

$$y = \sum_{i \in \text{TopK}} G(x)_i \cdot E_i(x)$$

### C. The Structural Challenge: Expert Capacity & Load Balancing
If the router becomes biased, it may send every token to the same expert, causing hardware serialization bottlenecks and rendering other experts untrained. MoE architectures solve this by adding an explicit **Auxiliary Load Balancing Loss** during training to penalize uneven token distribution across experts.

---

## 2. Inference Acceleration: Speculative Decoding

Autoregressive text generation is fundamentally **memory-bound** because the model must execute a full high-latency forward pass to sample just a single token. **Speculative Decoding** breaks this constraint by trading cheap arithmetic computation for memory access savings.

### The Algorithm Pipeline
1. **Draft Step:** A small, highly optimized "draft" model (e.g., a 1B parameter model) quickly generates a sequence of $K$ candidate tokens autoregressively. This is exceptionally fast due to its small VRAM footprint.
2. **Target Verification Step:** The massive, high-accuracy "target" model (e.g., a 70B parameter model) processes the prompt and all $K$ draft tokens simultaneously in a **single parallel forward pass**.
3. **Statistical Verification:** The target model checks the draft tokens against its own output distributions using a modified rejection sampling criteria. If the target model accepts the first $M$ tokens ($M \le K$) but rejects the $(M+1)$-th token, the accepted tokens are kept, the rest are discarded, and the target model outputs its corrected token.

Even if some draft tokens are rejected, this method routinely generates $2\times$ to $3\times$ more tokens per second than the target model working alone, without altering the final output probability distribution.

---

## 3. Implementation in Python (PyTorch)

This script demonstrates how to construct a functional **Top-2 Mixture of Experts (MoE) Layer** from scratch in PyTorch, executing sparse gating operations across an array of parallel linear expert blocks.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertLinearBlock(nn.Module):
    """Represents an individual FFN expert unit within the MoE framework."""
    def __init__(self, d_model: int, d_ff: int):
        super(ExpertLinearBlock, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(x)))


class Top2MixtureOfExperts(nn.Module):
    """
    Implements a Top-2 Sparse Mixture of Experts layer.
    """
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8):
        super(Top2MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        
        # Instantiate expert pool module list
        self.experts = nn.ModuleList([ExpertLinearBlock(d_model, d_ff) for _ in range(num_experts)])
        
        # Router network projecting tokens to expert slots
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [BatchSize, SeqLen, D_model]
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])  # Flatten to a sequence of individual tokens: [TotalTokens, D_model]
        
        # 1. Compute raw routing logits for every token across all experts
        router_logits = self.router(x)  # Shape: [TotalTokens, NumExperts]
        
        # 2. Isolate top-2 scoring expert paths and values
        top2_logits, top2_indices = torch.topk(router_logits, k=2, dim=-1) # Shape: [TotalTokens, 2]
        
        # 3. Apply Softmax routing coefficients over the isolated selections
        routing_weights = F.softmax(top2_logits, dim=-1)  # Shape: [TotalTokens, 2]
        
        # Allocate clean output tensor matrix placeholder
        final_output = torch.zeros_like(x)
        
        # 4. Route tokens to their designated experts (Iterate through top-k slots)
        for k_idx in range(2):
            weights_k = routing_weights[:, k_idx].unsqueeze(1)    # [TotalTokens, 1]
            expert_indices_k = top2_indices[:, k_idx]             # [TotalTokens]
            
            # Group tokens belonging to the same expert together to execute efficiently
            for expert_id in range(self.num_experts):
                token_mask = (expert_indices_k == expert_id)
                if not token_mask.any():
                    continue
                    
                # Extract corresponding token representations
                selected_tokens = x[token_mask]
                
                # Execute specific expert network calculation
                expert_out = self.experts[expert_id](selected_tokens)
                
                # Accumulate the weighted results into final output tensor
                final_output[token_mask] += weights_k[token_mask] * expert_out
                
        return final_output.view(orig_shape)

# Verification execution pass
if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, D = 2, 4, 64  # Batch=2, Seq=4 tokens, Dimension size=64
    
    moe_layer = Top2MixtureOfExperts(d_model=D, d_ff=128, num_experts=4)
    mock_hidden_states = torch.randn(B, T, D)
    
    output = moe_layer(mock_hidden_states)
    
    print(f"Input Matrix Configuration Shape:  {mock_hidden_states.shape}")
    print(f"MoE Sparse Layer Output Shape:    {output.shape}")
    print("Execution successfully verified. Token elements routed dynamically across expert sub-networks.")
```
