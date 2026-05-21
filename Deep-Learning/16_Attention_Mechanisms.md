# 16. Attention Mechanisms

Attention mechanisms revolutionized sequence modeling by allowing neural networks to focus on specific parts of an input sequence when generating an output, rather than squeezing an entire source sequence into a single fixed-length context vector.

---

## 1. The Bottleneck of Encoder-Decoder RNNs
In a standard Seq2Seq model (like early machine translation systems):
1.  The **Encoder** processes the input sequence and condenses it into a final hidden state vector (the context vector).
2.  The **Decoder** uses this single vector to generate the target sequence.

**The Problem:** For long sentences, a fixed-length vector cannot hold all the information. Performance drops sharply as sentence length increases because early information gets overwritten.

---

## 2. Core Framework of Attention
Instead of passing only the final hidden state, the encoder passes **all hidden states** ($h_1, h_2, \dots, h_T$) to the decoder. At each decoding step $t$, the attention mechanism dynamically computes a unique context vector $c_t$.

The three core vectors are defined as:
* **Query ($s_{t-1}$):** The current decoder hidden state (looking for information).
* **Keys ($h_i$):** The encoder hidden states (indexing the available information).
* **Values ($h_i$):** The same encoder hidden states (the actual information to extract).

---

## 3. Mathematical Execution Step-by-Step

For a decoder step $t$ and encoder hidden states $h_i$ (where $i = 1 \dots T$):

### Step 1: Alignment (Score) Function
Compute how well the input at position $i$ matches the current output at position $t$.

* **Luong Dot-Product Attention Score:**
    $$e_{t,i} = s_{t-1}^\top h_i$$
* **Bahdanau Additive Attention Score:**
    $$e_{t,i} = v_a^\top \tanh(W_a s_{t-1} + U_a h_i)$$

### Step 2: Attention Weights (Softmax)
Normalize the alignment scores into a probability distribution across the input sequence:
$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T} \exp(e_{t,j})}$$

### Step 3: Context Vector
Compute a weighted sum of the encoder values based on the attention weights:
$$c_t = \sum_{i=1}^{T} \alpha_{t,i} h_i$$

---

## 4. Comparison Table

| Feature | Bahdanau Attention (Additive) | Luong Attention (Dot-Product) |
| :--- | :--- | :--- |
| **Mathematical Basis** | Multi-layer perceptron addition | Matrix multiplication dot-product |
| **Decoder State Used** | Previous decoder state $s_{t-1}$ | Current decoder state $s_t$ |
| **Computational Speed** | Slower due to non-linear layers | Faster, highly optimized via matrix ops |
| **Scale Sensitivity** | Stable for large dimensions | Requires scaling factors for deep vectors |

---

## 5. Implementation in Python (PyTorch)

This script demonstrates a clean modular implementation of **Luong Dot-Product Attention** in PyTorch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LuongDotProductAttention(nn.Module):
    """
    Implements Luong's Dot-Product Attention Layer.
    """
    def __init__(self):
        super(LuongDotProductAttention, self).__init__()

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Args:
            decoder_hidden: Current hidden state of decoder [batch_size, hidden_dim]
            encoder_outputs: Matrix of all encoder hidden states [batch_size, seq_len, hidden_dim]
        Returns:
            context_vector: Weighted sum of encoder states [batch_size, hidden_dim]
            attention_weights: Normalized alignment probabilities [batch_size, seq_len]
        """
        # 1. Expand decoder dimensions to match matrix multiplication alignment requirements
        # [batch_size, hidden_dim] -> [batch_size, hidden_dim, 1]
        decoder_hidden_expanded = decoder_hidden.unsqueeze(2)
        
        # 2. Compute alignment scores using batch matrix multiplication (bmm)
        # [batch_size, seq_len, hidden_dim] x [batch_size, hidden_dim, 1] -> [batch_size, seq_len, 1]
        scores = torch.bmm(encoder_outputs, decoder_hidden_expanded)
        scores = scores.squeeze(2) # Shape: [batch_size, seq_len]
        
        # 3. Softmax activation normalizes alignment values to probabilities
        attention_weights = F.softmax(scores, dim=1) # Shape: [batch_size, seq_len]
        
        # 4. Generate the context vector via a weighted sum of values
        # [batch_size, 1, seq_len] x [batch_size, seq_len, hidden_dim] -> [batch_size, 1, hidden_dim]
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context_vector = context_vector.squeeze(1) # Shape: [batch_size, hidden_dim]
        
        return context_vector, attention_weights

# Verification execution pass
if __name__ == "__main__":
    batch_size = 4
    seq_length = 7
    hidden_dim = 64
    
    attention_layer = LuongDotProductAttention()
    
    # Simulate decoder state and encoder memory bank outputs
    mock_decoder_state = torch.randn(batch_size, hidden_dim)
    mock_encoder_states = torch.randn(batch_size, seq_length, hidden_dim)
    
    context, weights = attention_layer(mock_decoder_state, mock_encoder_states)
    
    print(f"Decoder State Shape:   {mock_decoder_state.shape}")
    print(f"Encoder Outputs Shape: {mock_encoder_states.shape}")
    print(f"Context Vector Shape:  {context.shape}")
    print(f"Attention Weights Shape: {weights.shape}")
```
