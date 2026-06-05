# Advanced Architectural Paradigms & Alternatives to Transformers

Standard Transformers dominate modern deep learning, but their computational complexity scales quadratically, $O(N^2)$, with sequence length $N$. This file covers cutting-edge, sub-quadratic architectures designed to achieve linear or near-linear scaling while maintaining or exceeding Transformer expressive power.

---

## 1. Mamba & State Space Models (SSMs)

### The Intuition
Traditional Structured State Space Models (S4) process sequences efficiently via parallel scans during training but are linear time-invariant (LTI). This means their transition matrices are static, preventing them from performing content-dependent reasoning (e.g., remembering a specific token based on context). 

**Mamba** solves this by introducing a **Selection Mechanism**, making the state space parameters functions of the input. It rejects the traditional trade-off between training efficiency ($O(N)$ parallelizable) and inference efficiency ($O(1)$ state updates) by using a hardware-aware SRAM/HBM memory management strategy.

### The Mathematical Formulation
The continuous-time State Space Model maps an input 1D signal $x(t) \in \mathbb{R}$ to a latent state $h(t) \in \mathbb{R}^N$ before projecting it to an output $y(t) \in \mathbb{R}$:

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t) + Dx(t)$$

To process discrete tokens, the continuous parameters $(A, B)$ must be discretized using a step size $\Delta$ (typically via Zero-Order Hold):

$$\overline{A} = \exp(\Delta A)$$
$$\overline{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$$

Mamba makes $\Delta$, $B$, and $C$ **input-dependent**:

$$B_t = \text{Linear}_B(x_t), \quad C_t = \text{Linear}_C(x_t), \quad \Delta_t = \text{Softplus}(\text{Linear}_\Delta(x_t))$$

### Architecture Data Flow
```text
Input Sequence X (B, L, D)
   │
   ├───> Linear Projection ──> 1D Conv ──> SilU ──> Selection Mechanism (Compute Δ, B, C) ──> Discretized SSM (Ā, B̄) ──> Y
   │                                                                                                                  ▲
   └───────────────────────────> Linear Projection ──────────────────> SilU ──────────────────────────────────────────┴─> Gating (Multiply) ──> Output
```
## 2. RWKV (Receptive Weighted Key Value)

### The Intuition
RWKV redesigns the Transformer's attention mechanism to behave like a Recurrent Neural Network (RNN) during inference while remaining parallelizable during training. It eliminates the explicit $N \times N$ attention matrix by formulating attention as a time-dependent exponential decay function.

### The Mathematical Formulation
In RWKV, the time-mix element computes formulation parameters $R$ (Receptance), $K$ (Key), and $V$ (Value):
$$r_t = \sigma(W_r \cdot x_t), \quad k_t = W_k \cdot x_t, \quad v_t = W_v \cdot x_t$$

The attention-like output $wkv_t$ is updated via an exponentially decaying history weight matrix $w$:
$$wkv_t = \frac{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} v_i + e^{u + k_t} v_t}{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} + e^{u + k_t}}$$

Where $u$ is a bonus weight applied specifically to the current token. During inference, this simplifies into a recursive hidden state formulation requiring only $O(1)$ memory.

## 3. Long-Context Mechanics: RoPE Scaling & ALiBi

When scaling context windows beyond what the model was trained on, standard absolute positional embeddings break completely. Advanced techniques modify how attention scores decay or scale across long ranges.

### Rotary Position Embedding (RoPE) Scaling (YaRN)

RoPE encodes positions by rotating the Query and Key vectors in the complex plane by an angle proportional to their index position. When extending context from length $L$ to $L'$, YaRN (Yet another RoFormer Extension Method) scales the attention weights by interpolating frequencies across different dimensions:
$$f(\nu_i) = \frac{\nu_i}{s}$$

Where $s = L'/L$ is the scale factor. YaRN applies a smooth non-linear interpolation across low, intermediate, and high-frequency components to prevent loss of high-frequency positional details.

### ALiBi (Attention with Linear Biases)

ALiBi completely removes explicit positional embeddings. Instead, it adds a static, non-learned negative bias directly to the attention matrix proportional to the distance between tokens:

$$\text{Attention}(Q, K) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + m \cdot \begin{pmatrix} 0 & -1 & -2 & \dots \\ 1 & 0 & -1 & \dots \\ 2 & 1 & 0 & \dots \end{pmatrix}\right)$$

Where $m$ is a head-specific scalar geometric slope ($m = \frac{1}{2^{8i/H}}$ for head $i$ out of $H$ total heads).

## 4. Complete PyTorch Implementation
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MinimalMambaBlock(nn.Module):
    """
    A minimal, clear implementation of a Selective State Space Model block
    with input-dependent discretization parameters.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Structure projections
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=d_conv, 
            padding=d_conv - 1, 
            groups=d_model
        )
        
        # System Parameter A (Static Parameter)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().unsqueeze(0).repeat(d_model, 1)))
        
        # Input-Dependent (Selective) Projections for B, C, and Delta
        self.x_proj = nn.Linear(d_model, d_state * 2 + d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Sequence_Length, d_model)
        """
        B, L, D = x.shape
        
        # Step 1: Input Projection & Split into main branch and gate branch
        projected = self.in_proj(x) # (B, L, 2*D)
        x_branch, gate_branch = projected.chunk(2, dim=-1)
        
        # Step 2: 1D Convolution over sequence along spatial dimensions
        x_branch = x_branch.transpose(1, 2) # (B, D, L)
        x_branch = self.conv1d(x_branch)[:, :, :L] # Handle padding truncation
        x_branch = F.silu(x_branch).transpose(1, 2) # (B, L, D)
        
        # Step 3: Selection Mechanism calculations (B, C, Delta)
        A = -torch.exp(self.A_log) # Ensure stable negative values (D, d_state)
        
        ssm_params = self.x_proj(x_branch) # (B, L, 2*d_state + D)
        B_mat, C_mat, dt = torch.split(ssm_params, [self.d_state, self.d_state, D], dim=-1)
        
        delta = F.softplus(self.dt_proj(dt)) # Ensure positive step sizes (B, L, D)
        
        # Step 4: Discretization & Recurrent Selective Scan Loop
        # High performance systems use parallel scans, recurrent loop shown for clarity
        h = torch.zeros(B, D, self.d_state, device=x.device)
        y = torch.zeros(B, L, D, device=x.device)
        
        for t in range(L):
            # Select slice at time step t
            x_t = x_branch[:, t, :] # (B, D)
            delta_t = delta[:, t, :].unsqueeze(-1) # (B, D, 1)
            B_t = B_mat[:, t, :].unsqueeze(1) # (B, 1, d_state)
            C_t = C_mat[:, t, :].unsqueeze(-1) # (B, d_state, 1)
            
            # Discretize continuous system matrices
            A_bar = torch.exp(delta_t * A.unsqueeze(0)) # (B, D, d_state)
            B_bar = delta_t * B_t # (B, D, d_state)
            
            # State Update equation: h_t = A_bar * h_{t-1} + B_bar * x_t
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            
            # Output Equation: y_t = C_t * h_t
            y[:, t, :] = torch.squeeze(h @ C_t, dim=-1)
            
        # Step 5: Multiplicative Gating & Out Projection
        y = y * F.silu(gate_branch)
        return self.out_proj(y)


class ALiBiAttention(nn.Module):
    """
    Linear Bias Multi-Head Attention (ALiBi) which handles out-of-distribution
    sequence lengths without explicit positional embeddings.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Precompute structural geometric slopes for ALiBi heads
        self.slopes = torch.tensor(self._get_slopes(n_heads))

    def _get_slopes(self, n_heads: int):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio**i) for i in range(n)]
        
        # Closest power of 2 interpolation strategy
        if math.log2(n_heads).is_integer():
            return get_slopes_power_of_2(n_heads)
        else:
            closest_power_of_2 = 2**int(math.log2(n_heads))
            return get_slopes_power_of_2(closest_power_of_2) + self._get_slopes(n_heads - closest_power_of_2)[::2]

    def _get_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Distance calculation matrix [0, -1, -2, ...]
        context_position = torch.arange(seq_len, device=device).unsqueeze(0)
        memory_position = torch.arange(seq_len, device=device).unsqueeze(1)
        relative_distance = memory_position - context_position # Shape: (seq_len, seq_len)
        relative_distance = -torch.abs(relative_distance)
        
        # Expand across batch-head dimensions
        # Slopes shape: (n_heads, 1, 1) * Relative Distance shape: (1, seq_len, seq_len)
        alibi_bias = self.slopes.to(device).view(1, self.n_heads, 1, 1) * relative_distance.unsqueeze(0).unsqueeze(0)
        return alibi_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Calculate raw scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Inject linear position bias matrix
        bias = self._get_alibi_bias(L, x.device)
        scores = scores + bias
        
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v) # (B, n_heads, L, head_dim)
        
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(context)

if __name__ == "__main__":
    sample_input = torch.randn(2, 32, 64) # (Batch, Seq_Len, Dim)
    
    mamba = MinimalMambaBlock(d_model=64)
    alibi = ALiBiAttention(d_model=64, n_heads=8)
    
    print("Mamba Block Output Shape:", mamba(sample_input).shape)
    print("ALiBi Attention Output Shape:", alibi(sample_input).shape)
```


