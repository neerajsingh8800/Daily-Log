# 14. LSTMs and GRUs (Gated Memory Mechanisms)

While standard RNNs can theoretically handle sequential data, they fail in practice on long sequences due to vanishing gradients. Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs) fix this by introducing internal **gates** that regulate the flow of information.

---

## 1. The Core Innovation: Linear Error Carousel
Standard RNNs overwrite their hidden state at every step using a non-linear activation function ($\tanh$). If the weights are small, gradients vanish over long periods. 

LSTMs solve this by introducing an internal **Cell State** ($c_t$) that changes linearly. Information can flow down this "conveyor belt" completely unaltered unless explicit gates decide to modify it.

---

## 2. LSTM (Long Short-Term Memory) Mathematics

An LSTM cell uses three gates to control the cell state ($c_t$) and hidden state ($h_t$):

1.  **Forget Gate ($f_t$):** Controls how much of the past memory to discard.
    $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

2.  **Input Gate ($i_t$ & $\tilde{c}_t$):** Decides what new information to store in the cell state.
    $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
    $$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

3.  **Cell State Update ($c_t$):** Updates the long-term memory linearly.
    $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

4.  **Output Gate ($o_t$ & $h_t$):** Decides what the next hidden state (short-term memory) should be.
    $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
    $$h_t = o_t \odot \tanh(c_t)$$

Where $\sigma$ is the sigmoid function, and $\odot$ represents element-wise (Hadamard) multiplication.

---

## 3. GRU (Gated Recurrent Unit) Mathematics

The GRU is a streamlined variant of the LSTM. It merges the cell state and hidden state, and uses only two gates, making it computationally faster.

1.  **Update Gate ($z_t$):** Acts as both the forget and input gate.
    $$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

2.  **Reset Gate ($r_t$):** Determines how much of the past hidden state to forget.
    $$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

3.  **Candidate State ($\tilde{h}_t$):**
    $$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b)$$

4.  **Hidden State Update ($h_t$):**
    $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

---

## 4. Comparison Architecture Table

| Feature | LSTM | GRU |
| :--- | :--- | :--- |
| **State Vectors** | Two ($c_t$ long-term, $h_t$ short-term) | One ($h_t$ combined memory) |
| **Number of Gates** | 3 (Forget, Input, Output) | 2 (Reset, Update) |
| **Speed / Efficiency** | Slower, more parameter-heavy | Faster, computationally lighter |
| **Performance** | Better on large data / long contexts | Comparable to LSTM on smaller datasets |

---

## 5. Implementation in Python (PyTorch)

This script implements a custom neural network pipeline utilizing a gated recurrent layer for classification sequence tracking.

```python
import torch
import torch.nn as nn

class GatedSequenceClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, variant='lstm'):
        super(GatedSequenceClassifier, self).__init__()
        self.variant = variant.lower()
        
        # Word Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Recurrent Layer Selection
        if self.variant == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif self.variant == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("Variant must be 'lstm' or 'gru'")
            
        # Linear Classifier Projection
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text shape: [batch_size, seq_length]
        embedded = self.embedding(text) # shape: [batch_size, seq_length, embedding_dim]
        
        if self.variant == 'lstm':
            # LSTM returns output, (hidden_state, cell_state)
            output, (hidden, cell) = self.rnn(embedded)
        else:
            # GRU returns output, hidden_state
            output, hidden = self.rnn(embedded)
            
        # Extract the final time-step hidden state to classify the entire sequence
        final_hidden = hidden[-1, :, :] # shape: [batch_size, hidden_dim]
        
        return self.fc(final_hidden)

# Verification execution pass
if __name__ == "__main__":
    vocab_size_mock = 1000
    model = GatedSequenceClassifier(vocab_size=vocab_size_mock, embedding_dim=64, hidden_dim=128, output_dim=2, variant='lstm')
    
    # Simulating a batch of 4 sequences, each containing 10 token tokens
    mock_input_batch = torch.randint(0, vocab_size_mock, (4, 10))
    predictions = model(mock_input_batch)
```
    
    print(f"Input batch shape: {mock_input_batch.shape}")
    print(f"Output classification logits shape: {predictions.shape}")
