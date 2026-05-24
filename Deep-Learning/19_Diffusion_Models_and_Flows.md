# 19. Diffusion Models and Flow Matching

Generative diffusion models and flow matching represent the modern state-of-the-art for high-fidelity data generation, driving systems like Stable Diffusion, Midjourney, and Imagen. Instead of generating data in a single step, these models treat generation as a gradual iterative inversion of an explicit corruption process.

---

## 1. Denoising Diffusion Probabilistic Models (DDPM)

Introduced by Sohl-Dickstein et al. (2015) and advanced by Ho, Jain, & Abbeel (2020), DDPMs model generation using two continuous Markov chains: a forward process that destroys data structure, and a reverse process that learns to reconstruct it.
Forward Process (Deterministic Noise Addition)
┌────────────────────────────────────────────────────────►
[x_0] ───► [x_1] ───► ... ───► [x_t-1] ───► [x_t] ───► [x_T]
(Data)                                                  (Noise)
◄────────────────────────────────────────────────────────┘
Reverse Process (Learned Denoising Network)
### A. The Forward (Noising) Process
The forward process adds small amounts of Gaussian noise to the clean data sample $\mathbf{x}_0 \sim q(\mathbf{x})$ across $T$ discrete steps according to a pre-defined variance schedule $\beta_1, \beta_2, \dots, \beta_T$:

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$$

Using the property of Gaussian closure, we can analytically sample the noisy representation at any arbitrary time step $t$ directly from $\mathbf{x}_0$ without iterative stepping. Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$:

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \text{where } \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

### B. The Reverse (Denoising) Process
Because the true distribution $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ depends on the entire data distribution, it is intractable. We train a neural network $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ to estimate these transitions:

$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$$

Instead of predicting the mean $\boldsymbol{\mu}_\theta$ directly, parameterizing the network to predict the exact added noise vector $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ results in superior sample stability.

### C. Objective Function
The simplified Mean Squared Error loss simplifies to optimizing the predicted noise profile against the true sampled Gaussian variance:

$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \|^2 \right] = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}, t) \|^2 \right]$$

---

## 2. Continuous Flows and Flow Matching

Advanced architectures often replace discrete DDPM stepping with **Continuous Normalizing Flows (CNFs)** and **Flow Matching** (Lipman et al., 2022). Instead of predicting noise scaling variables, a neural network explicitly models a time-dependent vector field $\mathbf{v}_\theta(\mathbf{x}, t)$ that defines a direct path transforming a simple noise distribution into the target data distribution.

### The Flow ODE
$$\frac{d\mathbf{x}_t}{dt} = \mathbf{v}_\theta(\mathbf{x}_t, t)$$

Using **Conditional Flow Matching (CFM)** with a linear path interpolation between noise $\mathbf{x}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and data $\mathbf{x}_0$:
$$\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\mathbf{x}_1$$

The target regression objective bypasses iterative sampling traps entirely by matching vectors directly:
$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1} \left[ \| \mathbf{v}_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0) \|^2 \right]$$

---

## 3. Core Architectural Differences

| Property | DDPM (Discrete Diffusion) | Flow Matching / CNFs |
| :--- | :--- | :--- |
| **Trajectory Path** | Curved, stochastic Brownian path | Straight, deterministic lines |
| **Step Count** | High (typically $250 - 1000$ steps) | Extremely low ($10 - 50$ evaluation steps) |
| **Target Variable** | Added noise component $\boldsymbol{\epsilon}$ | Vector velocity direction $\mathbf{v}$ |
| **Mathematical Type** | Stochastic Differential Equation (SDE) | Ordinary Differential Equation (ODE) |

---

## 4. Implementation in Python (PyTorch)

This script implements the core forward noising math, a time-conditioned neural projection framework, and the target training loss loop for a DDPM pipeline.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeConditionalDenoisingNetwork(nn.Module):
    """
    A simplified time-conditioned neural network mimicking 
    the structural routing of a DDPM U-Net block.
    """
    def __init__(self, data_dim=64, hidden_dim=128):
        super(TimeConditionalDenoisingNetwork, self).__init__()
        
        # Time Step Sinusoidal Embedding Projection
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Core Feature Processing Pipeline
        self.input_layer = nn.Linear(data_dim, hidden_dim)
        self.joint_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, data_dim)

    def forward(self, x, t):
        # x shape: [batch_size, data_dim]
        # t shape: [batch_size, 1]
        
        t_emb = self.time_mlp(t)      # Shape: [batch_size, hidden_dim]
        x_emb = self.input_layer(x)   # Shape: [batch_size, hidden_dim]
        
        # Condition feature layers with temporal contextual state
        h = F.relu(x_emb + t_emb)
        h = F.relu(self.joint_layer(h))
        
        return self.output_layer(h)   # Predicts the noise vector profile epsilon


class DDPMVarianceSchedule:
    """
    Handles analytical forward-pass noising updates for diffusion steps.
    """
    def __init__(self, total_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.total_steps = total_steps
        
        # Standard linear schedule allocation
        self.betas = torch.linspace(beta_start, beta_end, total_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def sample_forward_noise(self, x_0, t):
        """
        Analytically computes x_t given clean input tensor x_0 and step tensor t.
        Formula: x_t = sqrt(alpha_cumprod)*x_0 + sqrt(1 - alpha_cumprod)*noise
        """
        # Gather coefficients for batch tracking matching current time positions
        alpha_bar = self.alphas_cumprod.gather(0, t.view(-1)).view(-1, 1)
        
        noise = torch.randn_like(x_0)
        mean_coefficient = torch.sqrt(alpha_bar)
        std_coefficient = torch.sqrt(1.0 - alpha_bar)
        
        x_t = mean_coefficient * x_0 + std_coefficient * noise
        return x_t, noise

# Verification execution pass
if __name__ == "__main__":
    B, D = 4, 64  # Batch=4, Data feature vector dimensionality = 64
    
    # Initialize components
    scheduler = DDPMVarianceSchedule(total_steps=1000)
    denoiser = TimeConditionalDenoisingNetwork(data_dim=D)
    
    # Simulate a clean input batch
    mock_x0 = torch.randn(B, D)
    
    # Sample a set of random time steps for the batch
    mock_t = torch.randint(0, 1000, (B, 1))
    
    # 1. Forward Pass: Compute analytical corrupted image states
    x_t, true_noise = scheduler.sample_forward_noise(mock_x0, mock_t)
    
    # 2. Reverse Pass: Predict structural noise profiles via network conditions
    predicted_noise = denoiser(x_t, mock_t.float())
    
    # 3. Optimization Criterion: Simplified MSE Loss
    loss = F.mse_loss(predicted_noise, true_noise)
    
    print(f"Noisy Vector Shape (x_t):       {x_t.shape}")
    print(f"Target Noise Vector Shape:      {true_noise.shape}")
    print(f"Predicted Noise Vector Shape:   {predicted_noise.shape}")
    print(f"Computed Simplified DDPM Loss:  {loss.item():.4f}")
```


