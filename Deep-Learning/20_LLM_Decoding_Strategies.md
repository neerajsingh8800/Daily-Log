# 20. LLM Decoding Strategies

When an autoregressive Large Language Model (Transformer Decoder) processes a prompt, its output layer outputs raw, unnormalized log-probabilities (**logits**) for every token across its entire vocabulary. **Decoding strategies** dictate how we sample or select from this probability distribution to generate coherent, creative, or deterministic text.

---

## 1. Deterministic vs. Stochastic Decoding

### A. Greedy Search
The simplest strategy. At each time step $t$, it selects the token with the absolute highest probability.
* **Math:** $\hat{w}_t = \arg\max_{w \in V} P(w \mid w_{<t})$
* **Vulnerability:** It is short-sighted. Choosing the local optimum at step $t$ can lead to highly repetitive loops ("the the the") or miss high-probability phrases that hide behind a low-probability token.

### B. Beam Search
Instead of keeping just the single best token, Beam Search tracks a fixed number ($B$) of parallel hypotheses (beams). At each step, it expands all $B$ paths and keeps the top $B$ sequences with the highest cumulative log-probability.
* **Math:** Maximizes $\sum_{i=1}^{t} \log P(w_i \mid w_{<i})$
* **Use Case:** Excellent for structured tasks with definitive right answers (e.g., machine translation, code generation). It often sounds robotic or flat in open-ended creative writing.

---

## 2. Distribution Shaping and Sampling Mechanics

To allow for creative variety, we sample randomly from the model's vocabulary distribution. To ensure the text remains high-quality, we shape the distribution first using three core hyperparameters.

### A. Temperature Scaling ($T$)
Temperature scales the logits before the softmax layer to alter the "sharpness" of the distribution.

$$P(w_i) = \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}$$

Where $z_i$ is the raw logit for token $i$.
* **High Temperature ($T > 1.0$):** Flattens the distribution, giving low-probability tokens a higher chance of selection. This increases creativity and randomness but increases formatting hallucinations.
* **Low Temperature ($T \to 0$):** Sharpens the peaks, making the distribution act like a deterministic Greedy Search.

### B. Top-$k$ Sampling
Limits the sampling pool to the $k$ most probable tokens. The probabilities of all other tokens are set to zero, and the remaining distribution is re-normalized.
* **Limitation:** A static $k$ is rigid. If the top 3 tokens are all highly likely, a $k=50$ pool lets in 47 completely irrelevant noise tokens.

### C. Top-$p$ (Nucleus) Sampling
Introduced by Holtzman et al. (2019), Top-$p$ dynamically scales the sampling pool based on cumulative probability. It sorts tokens by probability and keeps only the top group whose sum reaches threshold $p$.

$$\sum_{i=1}^{K_{\text{dynamic}}} P(w_i) \ge p$$

* **Behavior:** If the model is highly confident, the pool shrinks to just 1 or 2 tokens. If the model is uncertain, the pool expands dynamically to hundreds of choices, preserving safe creativity.

---

## 3. Configuration Comparison Matrix

| Strategy | Search Type | Diversity | Compute Cost | Best Used For |
| :--- | :--- | :--- | :--- | :--- |
| **Greedy** | Deterministic | Zero | Very Low | Math, Code, Short Facts |
| **Beam Search** | Deterministic | Low | High ($O(B)$) | Translation, Summarization |
| **Top-$k$ + Temp** | Stochastic | Balanced | Low | Chatbots, Storytelling |
| **Top-$p$ + Temp** | Stochastic | High (Adaptive) | Low | Modern Open-ended LLMs |

---

## 4. Implementation in Python (PyTorch)

This script implements a text generation decoding loop demonstrating **Temperature scaling, Top-$k$ filtering, and Top-$p$ (Nucleus) pooling** directly on a tensor of logits.

```python
import torch
import torch.nn.functional as F

def filter_logits(logits, top_k=0, top_p=0.0):
    """
    Filters a distribution of logits using top-k and/or top-p (Nucleus) thresholds.
    """
    # Clean output allocation copy
    logits = logits.clone()
    
    # 1. Apply Top-k filtering
    if top_k > 0:
        # Determine threshold value matching the lower bound of top_k elements
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    # 2. Apply Top-p (Nucleus) filtering
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Shift the mask to ensure we keep the first token that crosses the top_p threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Map masked elements back onto the original un-sorted indices
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
        
    return logits

def sample_next_token(logits, temperature=1.0, top_k=0, top_p=0.0):
    """
    Processes model logits to sample a single token using temperature and top-k/p strategies.
    """
    # 1. Apply Temperature scaling
    if temperature != 1.0:
        logits = logits / temperature
        
    # 2. Filter distributions
    filtered_logits = filter_logits(logits, top_k=top_k, top_p=top_p)
    
    # 3. Softmax transformation to build normalized probability weights
    probabilities = F.softmax(filtered_logits, dim=-1)
    
    # 4. Multinomial sampling handles random selection along shaped probabilities
    next_token = torch.multinomial(probabilities, num_samples=1)
    return next_token

# Verification execution pass
if __name__ == "__main__":
    torch.manual_seed(42)
    V_size = 1000 # Mock vocabulary size
    
    # Simulating raw output logits from a model
    mock_logits = torch.randn(V_size) * 5.0
    
    # Generate token selection variants
    greedy_token = torch.argmax(mock_logits)
    sampled_token = sample_next_token(mock_logits, temperature=0.7, top_k=50, top_p=0.9)
    
    print(f"Initial highest probability logit token (Greedy): {greedy_token.item()}")
    print(f"Sampled token index using shaped (Top-p/k) strategy: {sampled_token.item()}")
