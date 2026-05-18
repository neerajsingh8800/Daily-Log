# 13. Sequence Modeling and Recurrent Neural Networks (RNNs)

Traditional feedforward networks assume all inputs and outputs are independent of each other (e.g., image classification). Sequence modeling handles data where order and temporal context matter, such as natural language processing (NLP), time series forecasting, and audio recognition.

---

## 1. Why Recurrent Neural Networks?
When processing text or sequences, traditional networks face two major limitations:
1.  **Fixed-length constraints:** They cannot easily handle inputs or outputs of varying dimensions.
2.  **Lack of historical context:** They do not share features learned across different positions of text.

An RNN solves this by incorporating a **Hidden State** ($h_t$), which acts as a memory loop to retain information from previous time steps.

---

## 2. Mathematical Framework of an RNN

At each time step $t$, the network processes the current input vector $\mathbf{x}_t$ and the previous hidden state $\mathbf{h}_{t-1}$ to update its memory and produce an output $\mathbf{y}_t$.

### The Forward Pass Equations:
1.  **Hidden State Update:**
    $$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$

2.  **Output Calculation:**
    $$\mathbf{y}_t = \text{softmax}(\mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y)$$

Where:
* $\mathbf{x}_t$: Input vector at time step $t$.
* $\mathbf{h}_t$: Current hidden state (memory vector).
* $\mathbf{W}_{xh}, \mathbf{W}_{hh}, \mathbf{W}_{hy}$: Shared weight matrices across all time steps.
* $\mathbf{b}_h, \mathbf{b}_y$: Bias vectors.

---

## 3. Backpropagation Through Time (BPTT) and Bottlenecks

To train an RNN, the network is unrolled across the entire sequence length. Gradients are calculated and accumulated at every individual step, tracking backward through time.

### The Vanishing & Exploding Gradient Problem
Because the weight matrix $\mathbf{W}_{hh}$ is multiplied repeatedly at every single step, the gradient update for early steps contains a factor of $(\mathbf{W}_{hh})^T$.
* If the largest eigenvalue of $\mathbf{W}_{hh} > 1$, gradients grow exponentially (**Exploding Gradients**).
* If the largest eigenvalue of $\mathbf{W}_{hh} < 1$, gradients shrink exponentially to zero (**Vanishing Gradients**).

> **Impact:** Standard RNNs struggle to preserve long-range dependencies, failing to connect context across more than a few words or steps.

---

## 5. Implementation in Python (NumPy)

This script demonstrates a basic recurrent sequence processing loop (Forward Pass) across an input text representation using NumPy.

```python
import numpy as np

class VanillaRNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.hidden_dim = hidden_dim
        
        # Initialize weights randomly
        self.W_xh = np.random.randn(hidden_dim, input_dim) * 0.01
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.W_hy = np.random.randn(output_dim, hidden_dim) * 0.01
        
        # Biases
        self.b_h = np.zeros((hidden_dim, 1))
        self.b_y = np.zeros((output_dim, 1))

    def forward(self, inputs):
        """
        inputs: List of vectors or 2D array of shape (sequence_length, input_dim)
        """
        h = np.zeros((self.hidden_dim, 1)) # Initial hidden state
        hidden_states = {}
        outputs = {}
        
        # Loop through sequence time steps
        for t, x_t in enumerate(inputs):
            x_t = x_t.reshape(-1, 1) # Ensure column vector
            
            # Compute next hidden state: H_t = tanh(W_hh * H_t-1 + W_xh * X_t + b_h)
            h = np.tanh(np.dot(self.W_hh, h) + np.dot(self.W_xh, x_t) + self.b_h)
            hidden_states[t] = h
            
            # Compute output step output: Y_t = W_hy * H_t + b_y
            y_t = np.dot(self.W_hy, h) + self.b_y
            outputs[t] = y_t
            
        return outputs, hidden_states

# Simulation Setup
seq_length = 4
features = 3
hidden_units = 5
classes = 2

rnn = VanillaRNN(input_dim=features, hidden_dim=hidden_units, output_dim=classes)
mock_sequence = np.random.randn(seq_length, features)

step_outputs, internal_memories = rnn.forward(mock_sequence)
print(f"Processed sequence of length: {len(step_outputs)}")
print(f"Final Step Raw Output Logits:\n {step_outputs[seq_length-1]}")
```
