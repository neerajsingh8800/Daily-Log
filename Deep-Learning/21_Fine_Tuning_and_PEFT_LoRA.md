# 21. Fine-Tuning and PEFT (LoRA Mechanics)

As Large Language Models scale to billions of parameters, performing Full Fine-Tuning (updating every weight in the network) becomes computationally prohibitive. Parameter-Efficient Fine-Tuning (PEFT) methods allow us to adapt massive pre-trained models to downstream tasks by updating only a tiny fraction of the parameters, drastically saving memory and storage.

---

## 1. Full Fine-Tuning vs. PEFT
* **Full Fine-Tuning:** Modifies all weights ($W$). Requires storing optimizer states (like Adam's momentum and variance vectors) for every single parameter. For a 7B parameter model, this requires massive multi-GPU setups just to hold the gradients and states.
* **PEFT:** Keeps the original pre-trained weights frozen. It attaches small, trainable adapter layers or modifications to the network. Storage requirements drop from tens of gigabytes per task down to a few megabytes.

---

## 2. Low-Rank Adaptation (LoRA) Mathematics

Introduced by Hu et al. (2021), LoRA relies on a core hypothesis from intrinsic dimensionality literature: **the weight updates ($\Delta W$) during adaptation have a low "intrinsic rank".**

For a frozen pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA decomposes the weight update matrix $\Delta W$ into the product of two low-rank matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$, where the rank $r \ll \min(d, k)$.

### The Forward Pass Equation:
$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B A) x$$

Where:
* $W_0$: The frozen pre-trained weight matrix.
* $A$: The first down-projection adapter matrix, initialized from a random Gaussian distribution $\mathcal{N}(0, \sigma^2)$.
* $B$: The second up-projection adapter matrix, initialized to **zero**. Since $B=0$ at the start of training, $\Delta W = 0$, meaning the model's behavior is completely unaltered at step zero.
* $\alpha$: A constant scaling hyperparameter. It distributes structural scaling weight evenly when adjusting the rank $r$.

---

## 3. Why LoRA Saves VRAM

When training with the Adam optimizer, memory is consumed heavily by **optimizer states**.
* Adam tracks **2 additional floating-point scalars** (first and second momentum moments) for every single trainable parameter.
* By freezing a $d \times k$ matrix and only training $A$ and $B$, the number of parameters drops from $d \times k$ down to $r \times (d + k)$.
* If $d=4096, k=4096$, and $r=8$, parameters drop from **16,777,216** to **65,536** (a $99.6\%$ reduction in trainable parameters and associated optimizer memory).

---

## 4. Implementation in Python (PyTorch)

This script demonstrates how to implement a custom **LoRA Linear Layer wrapper** from scratch in PyTorch, mimicking the core design of libraries like Hugging Face `peft`.

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    Implements a low-rank adaptation wrapper over a standard frozen linear layer.
    """
    def __init__(self, base_layer: nn.Linear, r: int = 8, lora_alpha: int = 16):
        super(LoRALinear, self).__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # 1. Freeze the weights of the original base pre-trained layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
            
        # 2. Define the low-rank matrices A and B
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        
        # 3. Initialize parameters matching original paper setup
        self.reset_parameters()

    def reset_parameters(self):
        # Matrix A is initialized via random Gaussian parameters
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Matrix B is initialized strictly to zero to ensure delta_W starts at zero
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Compute standard inference pass through the frozen base weights
        base_output = self.base_layer(x)
        
        # Step 2: Compute parallel path through low-rank bottleneck matrices
        # x shape: [batch_size, seq_len, in_features] or [batch_size, in_features]
        # lora_A transpose matches matrix multiplication dimensions: x @ A^T
        lora_output = (x @ self.lora_A.t()) @ self.lora_B.t()
        
        # Step 3: Combine paths applying the explicit alpha scaling factor
        return base_output + (lora_output * self.scaling)

# Verification execution pass
if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, D_in, D_out = 2, 5, 512, 1024  # Batch=2, Seq=5 tokens, Dim In=512, Dim Out=1024
    
    # Standard base linear projection layer representing a frozen weight block
    original_layer = nn.Linear(D_in, D_out)
    
    # Wrap with LoRA module
    lora_adapted_layer = LoRALinear(original_layer, r=8, lora_alpha=16)
    
    mock_hidden_states = torch.randn(B, T, D_in)
    output = lora_adapted_layer(mock_hidden_states)
    
    print(f"Input Matrix Shape:          {mock_hidden_states.shape}")
    print(f"LoRA Output Matrix Shape:    {output.shape}")
    
    # Verify that the gradient graph tracks only the adapter parameters
    trainable_params = [p for p in lora_adapted_layer.parameters() if p.requires_grad]
    print(f"Number of Active Trainable Parameter Tensors: {len(trainable_params)}")  # Should be 2 (lora_A and lora_B)
```
