# 36. Activation Functions for LLMs (SwiGLU & Gated Linear Units)

While foundational deep learning architectures rely on standard element-wise non-linearities like ReLU or GELU within their Feed-Forward Network (FFN) blocks, modern Large Language Models (e.g., LLaMA, Mistral, Gemma) utilize a more expressive class of layers known as **Gated Linear Units (GLU)**. Among these, **SwiGLU** has emerged as the industry standard for eliminating vanishing gradients, improving training stability, and boosting token-prediction performance metrics.

---

## 1. From Standard FFNs to Gated Linear Units (GLU)

In a classic Transformer FFN layer, tokens pass through an initial linear projection, a non-linear activation function ($\sigma$), and a secondary structural linear down-projection:

$$\text{FFN}_{\text{standard}}(x) = \sigma(xW_1 + b_1)W_2 + b_2$$

A **Gated Linear Unit (GLU)** component alters this paradigm. Instead of applying an activation to a single matrix projection, a GLU splits the input into two parallel pathways: a **gate pathway** and a **value pathway**. The two vectors are combined element-wise via a Hadamard (component-wise) product ($\otimes$). 

$$\text{GLU}(x) = \sigma(xW_{\text{gate}}) \otimes xW_{\text{value}}$$

This structural design allows the model to dynamically modulate information flow. The output of the activation function directly controls how much of the parallel value matrix projection passes through to subsequent layers.

---

## 2. The SwiGLU Variant

Introduced by Noam Shazeer (2020), **SwiGLU** is a specific variant of a Gated Linear Unit where the gating activation function $\sigma$ is replaced by the **Swish** ($\text{Swish}_\beta$) activation function (often structurally identical to **SiLU** when $\beta=1$).

### A. The Swish / SiLU Activation Function
The Swish function serves as a smooth, continuously differentiable alternative to ReLU, retaining a minor gradient profile for negative input boundaries:

$$\text{Swish}_1(x) = \text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

### B. The SwiGLU Layer Mathematical Equation
By embedding the Swish function into the GLU framework, we formulate the complete SwiGLU mathematical operator:

$$\text{SwiGLU}(x) = \text{Swish}_1(xW_{\text{gate}} + b_{\text{gate}}) \otimes (xW_{\text{value}} + b_{\text{value}})$$

### C. The Full LLM FFN Block Configuration
In actual LLM implementations (like LLaMA), biases are completely removed to assist numerical stability during FP16/BF16 training loops. The finalized 3-matrix SwiGLU FFN block is defined as:

$$\text{FFN}_{\text{SwiGLU}}(x) = \left( \text{Swish}_1(xW_{\text{gate}}) \otimes xW_{\text{value}} \right) W_{\text{down}}$$

Where:
* $W_{\text{gate}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ projects tokens into the gating network space.
* $W_{\text{value}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ projects tokens into the parallel value processing channel.
* $W_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$ scales the fused inner representations back down to the model's base dimension.

To maintain an identical parameter budget to traditional architectures, the hidden dimension $d_{\text{ff}}$ is typically scaled down to roughly $\frac{8}{3}d_{\text{model}}$ instead of the classic $4d_{\text{model}}$ expansion.

---

## 3. Comparative Properties

| Structural Property | Standard ReLU FFN | SwiGLU FFN Block |
| :--- | :--- | :--- |
| **Mathematical Complexity** | Low ($1\times$ Matrix Multiplication before non-linearity) | Moderate ($2\times$ Parallel Matrix Multiplications) |
| **Gradient Flow Behavior** | Hard zero boundary at $x < 0$ (Dead Neurons) | Smooth non-monotonic landscape; keeps gradient lanes open |
| **Representational Capacity** | Linear gating partition | Multiplicative bilinear interaction space |
| **LLM Architectural Use** | Older Models (GPT-3, OPT) | Modern State-of-the-Art Models (LLaMA, Mistral, Gemma) |

---

## 4. Implementation in Python (PyTorch)

This script demonstrates how to implement a production-style **SwiGLU Feed-Forward Network Block** from scratch in PyTorch without using pre-packaged high-level wrappers. It verifies forward tensor transitions and tracks internal dimensional footprints.

```python
import torch
import torch.nn as nn

class SwiGLUFFNBlock(nn.Module):
    """
    Implements a standardized 3-matrix SwiGLU Feed-Forward Network block
    as configured in contemporary Large Language Model architectures.
    """
    def __init__(self, d_model: int, d_ff: int):
        super(SwiGLUFFNBlock, self).__init__()
        # Gate and Value parallel transformation layers
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_value = nn.Linear(d_model, d_ff, bias=False)
        
        # Down-projection structural layer
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input token activations of shape [Batch, SeqLen, D_model]
        Returns:
            Output activations of shape [Batch, SeqLen, D_model]
        """
        # 1. Compute parallel projection pathways
        gate_branch = self.w_gate(x)   # Shape: [Batch, SeqLen, D_ff]
        value_branch = self.w_value(x) # Shape: [Batch, SeqLen, D_ff]
        
        # 2. Compute explicit Swish/SiLU non-linearity over the gate branch
        # SiLU math: gate * sigmoid(gate)
        activated_gate = gate_branch * torch.sigmoid(gate_branch)
        
        # 3. Fuse branches element-wise via Hadamard multiplication matrix interaction
        fused_representation = activated_gate * value_branch # Shape: [Batch, SeqLen, D_ff]
        
        # 4. Downproject back to model structural embedding limits
        return self.w_down(fused_representation) # Shape: [Batch, SeqLen, D_model]


# Verification execution profiling pass
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Standard hyperparameter layout scales
    B, T, D_model = 2, 8, 512
    # Custom hidden layer dimension approximation using standard 8/3 expansion rounding rule
    D_ff = int(2 * (8 / 3) * D_model) // 2 
    
    print("--- Initializing SwiGLU Structural Architecture Check ---")
    print(f"Model Dimensional Width (d_model): {D_model}")
    print(f"Inner Hidden Expansion Width (d_ff):  {D_ff}\n")
    
    # Instantiate custom SwiGLU FFN block
    swiglu_layer = SwiGLUFFNBlock(d_model=D_model, d_ff=D_ff)
    
    # Generate mock hidden activations sequence
    mock_input_tokens = torch.randn(B, T, D_model)
    print(f"Input Tensor Shape:   {mock_input_tokens.shape}")
    
    # Process tensors through the network
    layer_output = swiglu_layer(mock_input_tokens)
    print(f"Output Tensor Shape:  {layer_output.shape}")
    
    # Sanity dimensionality integrity assertion check
    assert layer_output.shape == mock_input_tokens.shape, "Output dimensions must match input shape constraints."
    print("\nSwiGLU operational pass successfully executed and verified.")
```
