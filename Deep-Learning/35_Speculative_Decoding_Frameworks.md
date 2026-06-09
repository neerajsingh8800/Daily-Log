# 35. Speculative Decoding Frameworks

While techniques like KV-Caching reduce the computational complexity of autoregressive text generation from $O(T^2)$ to $O(T)$, inference remains heavily **memory-bound**. For every single token generated, a Large Language Model must fetch billions of weights from slow High-Bandwidth Memory (HBM) to local processor registers (SRAM). **Speculative Decoding Frameworks** bypass this memory bandwidth bottleneck by trading inexpensive parallel arithmetic calculations for a reduction in total HBM memory access steps, achieving a $2\times$ to $3\times$ speedup without altering the model's original output probability distribution.

---

## 1. The Dual-Model Pipeline Mechanics

Speculative decoding leverages an asymmetric architecture consisting of two distinct models running on the same vocabulary space:
1. **The Draft Model ($M_d$):** A compact, highly optimized network (e.g., 1B parameters) that can execute sequential token generation rapidly because its small size fits easily within fast processor memory caches.
2. **The Target Model ($M_t$):** The primary, massive network (e.g., 70B parameters) whose performance we want to accelerate.
### Step-by-Step Execution Loop
* **Lookahead Phase:** The Draft Model $M_d$ generates a short sequence of $K$ candidate tokens $\tilde{y}_{1}, \tilde{y}_{2}, \dots, \tilde{y}_{K}$ one by one.
* **Verification Phase:** The Target Model $M_t$ is fed the original prompt augmented with all $K$ draft tokens. It computes the full joint probability distributions for all positions concurrently in a **single parallel forward pass**.
* **Rejection Sampling Phase:** The system evaluates the draft tokens sequentially against the target model's distributions. If a draft token is rejected at position $i$, the loop terminates, all tokens from position $i$ onward are discarded, and a corrected token from the target model's distribution is appended.

---

## 2. Mathematical Rejection Sampling Thresholds

To ensure that speculative decoding remains an **exact match** (producing the identical output probability distribution as the target model operating alone), it employs a modified **Rejection Sampling** criteria.

Let $p(y \mid x)$ be the token probability distribution output by the Target Model $M_t$, and let $q(y \mid x)$ be the distribution output by the Draft Model $M_d$. For a proposed draft token $\tilde{y}$:

### A. The Acceptance Criteria
The system accepts the draft token $\tilde{y}$ with an allocation probability defined by:

$$P(\text{accept} \mid \tilde{y}) = \min\left(1, \frac{p(\tilde{y} \mid x)}{q(\tilde{y} \mid x)}\right)$$

* If the target model finds the token *more* or equally likely than the draft model did ($p(\tilde{y}) \ge q(\tilde{y})$), the token is **always accepted** ($P=1$).
* If the target model finds it *less* likely, it accepts it proportionally to the ratio of their probabilities.

### B. The Rejection/Correction Distribution
If the draft token $\tilde{y}$ is rejected, the system samples a replacement token from a modified, normalized residual distribution $p_{\text{adjust}}(y)$ to recover the exact target distribution profile without introducing statistical bias:

$$p_{\text{adjust}}(y) = \frac{\max(0, p(y \mid x) - q(y \mid x))}{\sum_{w} \max(0, p(w \mid x) - q(w \mid x))}$$

This adjustment ensures that the probability of outputting any token matches the target distribution $p(y \mid x)$ perfectly.

---

## 3. Structural Comparison

| Optimization Parameter | Standard Autoregressive Generation | Speculative Decoding Framework |
| :--- | :--- | :--- |
| **Tokens Per Forward Pass** | Exactly 1 token per pass | Up to $K+1$ tokens per pass |
| **Compute Profile** | Strongly Memory-Bound (Low Arithmetic Intensity) | Moderately Compute-Bound (High Parallel Intensity) |
| **Mathematical Output** | Ground Truth Distribution | Exact Match to Ground Truth Distribution |
| **Key Performance Driver** | Memory Clock Speed (HBM Throughput) | Draft Model Accuracy / Target Alignment Rate |

---

## 4. Implementation in Python (PyTorch)

This standalone script demonstrates how to implement the core **Speculative Decoding Rejection Sampling Logic** from scratch in PyTorch. It simulates a verification block that checks a sequence of proposed draft tokens against target distributions, applying the mathematical residual correction when a token is rejected.

```python
import torch
import torch.nn.functional as F

def verify_speculative_tokens(draft_tokens: torch.Tensor, 
                              draft_probs: torch.Tensor, 
                              target_probs: torch.Tensor) -> tuple:
    """
    Evaluates proposed draft tokens against target model distributions using
    exact mathematical rejection sampling.
    
    Args:
        draft_tokens: Tensor of shape [K] containing proposed token indices.
        draft_probs: Tensor of shape [K, VocabSize] outlining draft probabilities.
        target_probs: Tensor of shape [K + 1, VocabSize] outlining target probabilities.
        
    Returns:
        accepted_tokens: List of token indices that passed verification + the correction token.
        num_accepted: Int tracking how many draft tokens were verified.
    """
    accepted_tokens = []
    K = draft_tokens.size(0)
    num_accepted = 0
    
    # Iterate through each proposed candidate token sequentially
    for i in range(K):
        token_id = draft_tokens[i].item()
        
        # Isolate probabilities for the proposed token at step i
        p = target_probs[i, token_id].item()
        q = draft_probs[i, token_id].item()
        
        # 1. Compute the mathematical acceptance probability threshold
        acceptance_threshold = min(1.0, p / (q + 1e-8))
        
        # Roll a uniform random variable to test acceptance criteria
        rand_roll = torch.rand(1).item()
        
        if rand_roll < acceptance_threshold:
            # Token accepted! Append and move to the next token
            accepted_tokens.append(token_id)
            num_accepted += 1
        else:
            # 2. Token rejected! Compute the normalized residual distribution
            print(f"[Verification] Draft Token at position index {i} (ID: {token_id}) REJECTED.")
            
            residual_dist = target_probs[i] - draft_probs[i]
            residual_dist = torch.clamp(residual_dist, min=0.0) # Apply max(0, p - q)
            
            # Normalize to construct a valid probability map
            residual_dist_sum = residual_dist.sum()
            if residual_dist_sum > 0:
                residual_dist /= residual_dist_sum
            else:
                # Fallback directly to raw target distribution if fully overlapping
                residual_dist = target_probs[i]
                
            # Sample a replacement token from the corrected distribution
            correction_token = torch.multinomial(residual_dist, num_samples=1).item()
            accepted_tokens.append(correction_token)
            
            # Terminate the loop immediately; all subsequent tokens are invalidated
            return torch.tensor(accepted_tokens), num_accepted

    # 3. If all K tokens are accepted, append the target model's lookahead token
    print("[Verification] All proposed draft tokens successfully ACCEPTED.")
    final_lookahead_token = torch.multinomial(target_probs[-1], num_samples=1).item()
    accepted_tokens.append(final_lookahead_token)
    
    return torch.tensor(accepted_tokens), num_accepted


# Verification execution simulation
if __name__ == "__main__":
    torch.manual_seed(42)
    V = 10     # Virtual vocabulary size
    K_steps = 3 # Lookahead window budget size
    
    # Simulate historical draft outputs for K sequence blocks
    mock_draft_tokens = torch.tensor([2, 5, 7])
    
    # Generate mock probabilities for the draft model [K, V]
    mock_draft_probs = F.softmax(torch.randn(K_steps, V) * 2.0, dim=-1)
    
    # Generate mock probabilities for the target model [K + 1, V]
    # We purposefully add slight noise to simulate distribution differences
    mock_target_probs = F.softmax(torch.randn(K_steps + 1, V) * 2.0, dim=-1)
    
    print("--- Running Speculative Decoding Rejection Sampler ---")
    print(f"Proposed Draft Tokens: {mock_draft_tokens.tolist()}\n")
    
    final_sequence, accepted_count = verify_speculative_tokens(
        mock_draft_tokens, mock_draft_probs, mock_target_probs
    )
    
    print("\n--- Final Generation Step Metrics ---")
    print(f"Total Draft Tokens Validated: {accepted_count} / {K_steps}")
    print(f"Final Retained Output Sequence Vector: {final_sequence.tolist()}")
```
