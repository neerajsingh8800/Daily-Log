# 17. The Transformer Architecture

Introduced in the seminal paper *"Attention Is All You Need"* (Vaswani et al., 2017), the Transformer architecture completely abandoned recurrent loops and convolutions. It relies entirely on self-attention mechanisms to model global dependencies between inputs and outputs, allowing for unprecedented parallelization during training.

---

## 1. The Core Paradigm Shift
Traditional sequential models (LSTMs, GRUs) process tokens step-by-step, creating an $O(T)$ sequential path that prohibits parallel training over long sequences. Transformers eliminate this step-by-step dependency:
* **Constant Path Length:** The distance between any two tokens in a sequence is exactly $O(1)$, completely eliminating the vanishing gradient problem over long horizons.
* **Total Parallelization:** All tokens in a sequence are processed simultaneously during training, shifting the heavy computational lifting to highly optimized matrix operations.

---

## 2. Mathematical Foundations

The architecture relies on mapping a set of input vectors into **Queries ($Q$)**, **Keys ($K$)**, and **Values ($V$)** using learned weight matrices $W_Q, W_K, W_V$.

### A. Scaled Dot-Product Attention
The fundamental engine of the Transformer. The dot product of queries and keys computes raw similarity scores, which are then scaled to prevent gradients from vanishing in the Softmax layer during deep dimension configurations.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Where:
* $Q, K, V$: Matrices representing queries, keys, and values.
* $d_k$: The scaling dimension of the keys (the vector length). Dividing by $\sqrt{d_k}$ counteracts large dot products that push the softmax function into regions with extremely small gradients.

### B. Multi-Head Attention
Instead of performing a single attention pass over the full hidden dimension, **Multi-Head Attention** projects $Q, K$, and $V$ into $h$ lower-dimensional subspaces. This allows the network to jointly attend to information from different representation spaces at different positions.

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$
$$\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

---

## 3. Positional Encoding
Because Transformers process all tokens simultaneously, they possess no structural awareness of sequence order (i.e., swapping words yields identical representations). To preserve ordering, a deterministic **Positional Encoding ($PE$)** vector is added to the input word embeddings.

Using sine and cosine functions of different frequencies:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$

Where $pos$ is the token position in the sequence, and $i$ is the dimension index. This layout allows the model to easily learn to attend by relative positions.

---
## 4. Implementation in Python (PyTorch)

This script demonstrates how to build a scalable, fully functional **Multi-Head Self-Attention** block from scratch using PyTorch tensor operations.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    """
    Implements standard Multi-Head Self-Attention mechanics.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Combined linear projections for Query, Key, and Value matrices
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_key = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output project matrix
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.shape
        
        # 1. Linear projections
        Q = self.W_q(x)    # [batch_size, seq_len, d_model]
        K = self.W_key(x)  # [batch_size, seq_len, d_model]
        V = self.W_v(x)    # [batch_size, seq_len, d_model]
        
        # 2. Reshape into parallel heads: [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Compute scaled dot-product attention scores
        # K.transpose(-2, -1) shape: [batch_size, num_heads, d_k, seq_len]
        # scores shape: [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        # Apply optional attention mask (e.g., causal look-ahead mask for decoder)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 4. Normalize scores into attention distributions
        attention_weights = F.softmax(scores, dim=-1)
        
        # 5. Compute final weighted output context mapping
        # context shape: [batch_size, num_heads, seq_len, d_k]
        context = torch.matmul(attention_weights, V)
        
        # 6. Permute and concatenate heads back together
        # Reshape to original layout: [batch_size, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 7. Apply output projection linear layer
        return self.W_o(context)

# Verification execution pass
if __name__ == "__main__":
    B, T, D = 4, 10, 512  # Batch=4, Sequence length=10 tokens, Hidden Dim=512
    heads = 8             # 8 heads means each head acts over a sub-dimension of 64
    
    attention_block = MultiHeadSelfAttention(d_model=D, num_heads=heads)
    mock_input_tokens = torch.randn(B, T, D)
    
    output_representation = attention_block(mock_input_tokens)
    
    print(f"Input Matrix Shape:          {mock_input_tokens.shape}")
    print(f"Multi-Head Attention Output: {output_representation.shape}")
```
