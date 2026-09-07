# 06: Differential Privacy in Fine-Tuning — DP-SGD & Mathematical Privacy Guarantees

When fine-tuning pretrained Large Language Models (LLMs) on domain-specific datasets (e.g., patient clinical notes, financial transactions, internal Slack logs), standard Supervised Fine-Tuning (SFT) can inadvertently cause the model to memorize rare or sensitive training examples. An adversary with white-box or black-box API access can exploit this memorization using Membership Inference or Inversion attacks.

**Differential Privacy (DP)** provides a rigorous mathematical framework to bound privacy risks during model training. This module covers the theoretical mechanics of $(\epsilon, \delta)$-Differential Privacy, Differentially Private Stochastic Gradient Descent (DP-SGD), per-sample gradient clipping math, Rényi DP budget accounting, and a production-grade PyTorch implementation using Opacus.

---

## 1. Theoretical Foundations

### 1.1 $(\epsilon, \delta)$-Differential Privacy Definition
A randomized training algorithm $\mathcal{M}$ satisfies $(\epsilon, \delta)$-Differential Privacy if, for any two neighboring datasets $D$ and $D'$ differing by at most one individual record ($\vert{}D \Delta D'\vert{} = 1$), and for all possible output model parameters $S \subseteq \text{Range}(\mathcal{M})$:

$$P(\mathcal{M}(D) \in S) \le e^{\epsilon} \cdot P(\mathcal{M}(D') \in S) + \delta$$

* **$\epsilon$ (Privacy Budget)**: Binds the maximum log-odds ratio of leakage for any individual record. Smaller $\epsilon$ values ($\epsilon \le 3.0$) indicate strong privacy guarantees.
* **$\delta$ (Failure Probability)**: The probability that the $\epsilon$-bound fails completely. Mathematically, $\delta$ must be strictly smaller than $\frac{1}{\vert{}D\vert{}}$ (typically $10^{-5}$ to $10^{-7}$).

### 1.2 Differentially Private SGD (DP-SGD) Mechanics

Standard SGD updates model parameters $\mathbf{w}_t$ using batch-averaged gradients $\mathbf{g}_t = \frac{1}{B} \sum_{i=1}^{B} \nabla_{\mathbf{w}} \mathcal{L}(f(\mathbf{x}_i; \mathbf{w}_t), y_i)$. 

DP-SGD modifies this process via two core operations at every iteration:

1. **Per-Sample Gradient Clipping**: To bound the maximum influence (sensitivity) of any single training sample $i$, its individual gradient $\mathbf{g}_t^{(i)} = \nabla_{\mathbf{w}} \mathcal{L}_i$ is scaled down if its $L_2$-norm exceeds a clipping bound $C$:

$$\bar{\mathbf{g}}_t^{(i)} = \mathbf{g}_t^{(i)} \cdot \min\left(1, \frac{C}{\Vert{}\mathbf{g}_t^{(i)}\Vert{}_2}\right)$$

2. **Calibrated Noise Addition**: Gaussian noise proportional to the noise multiplier $\sigma$ and clipping norm $C$ is added to the summed clipped gradients before the weight update:

3. ### 1.3 Rényi Differential Privacy (RDP) Accounting
Repeated training steps consume privacy over time. Rather than using naive advanced composition theorems (which give loose $\epsilon$ bounds), production systems use **Rényi Differential Privacy (RDP)** accounting based on Rényi divergence of order $\alpha > 1$:

$$D_{\alpha}(P \parallel Q) = \frac{1}{\alpha - 1} \ln \int P(x)^{\alpha} Q(x)^{1 - \alpha} dx$$

RDP composes linearly across $T$ training steps: $D_{\alpha}(\mathcal{M}_{1..T}) = \sum_{t=1}^{T} D_{\alpha}(\mathcal{M}_t)$. The total accumulated RDP is then dynamically mapped back to standard $(\epsilon, \delta)$ bounds.

---

## 2. Privacy vs. Utility Trade-offs

| Parameter | Recommended Range | Impact of Increasing Parameter | Impact of Decreasing Parameter |
| :--- | :--- | :--- | :--- |
| **Max Gradient Norm ($C$)** | $0.1 \le C \le 1.0$ | Preserves gradient magnitude; increases injected noise scale $\sigma C$. | Clips large gradients aggressively; reduces injected noise scale. |
| **Noise Multiplier ($\sigma$)** | $0.5 \le \sigma \le 2.0$ | **Stronger Privacy** ($\epsilon \downarrow$); lower downstream accuracy. | **Weaker Privacy** ($\epsilon \uparrow$); higher downstream accuracy. |
| **Batch Size ($B$)** | Large ($B \ge 256$) | Amplifies privacy amplification by subsampling; reduces noise variance. | Higher per-step privacy cost; slower convergence under noise. |
| **Target Delta ($\delta$)** | $\le 10^{-5}$ | Relaxes privacy bound; risk of absolute leakage on edge cases. | Enforces strict theoretical privacy boundaries. |

---

## 3. Production PyTorch Implementation

This Python module implements a **Differentially Private Fine-Tuning Loop** using PyTorch and Opacus. It fine-tunes a classification/embedding head under DP-SGD while tracking the accumulated $(\epsilon, \delta)$ privacy budget in real-time.

### Prerequisites

```bash
pip install torch opacus pydantic
```

### Python Implementation (dp_fine_tuning.py)
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from pydantic import BaseModel
from typing import List, Tuple, Dict


class DPTrainingReport(BaseModel):
    total_epochs: int
    final_epsilon: float
    target_delta: float
    max_grad_norm: float
    noise_multiplier: float
    final_loss: float


class DifferentiallyPrivateFineTuner:
    """Manages DP-SGD fine-tuning execution and privacy budget accounting."""

    def __init__(
        self,
        model: nn.Module,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.1,
        target_delta: float = 1e-5
    ):
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.target_delta = target_delta

        # Validate and fix model layers for Opacus compatibility (e.g., replace BatchNorm with GroupNorm)
        if not ModuleValidator.is_valid(model):
            self.model = ModuleValidator.fix(model)
        else:
            self.model = model

        self.privacy_engine = PrivacyEngine()

    def train_with_privacy(
        self,
        dataset: TensorDataset,
        batch_size: int = 64,
        epochs: int = 3,
        lr: float = 1e-3
    ) -> DPTrainingReport:
        """Executes DP-SGD fine-tuning with real-time privacy accounting."""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Attach Privacy Engine to model, optimizer, and dataloader
        model, optimizer, dataloader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )

        model.train()
        last_loss = 0.0

        print(f"--- Starting DP-SGD Fine-Tuning (Noise: {self.noise_multiplier}, Clip Norm: {self.max_grad_norm}) ---")

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for step, (x_batch, y_batch) in enumerate(dataloader):
                optimizer.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            last_loss = avg_loss

            # Query Rényi DP accountant for cumulative epsilon at target delta
            epsilon = self.privacy_engine.get_epsilon(self.target_delta)
            print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Accumulated (ε = {epsilon:.2f}, δ = {self.target_delta})")

        final_eps = self.privacy_engine.get_epsilon(self.target_delta)

        return DPTrainingReport(
            total_epochs=epochs,
            final_epsilon=float(final_eps),
            target_delta=self.target_delta,
            max_grad_norm=self.max_grad_norm,
            noise_multiplier=self.noise_multiplier,
            final_loss=float(last_loss)
        )


# -------------------------------------------------------------------
# SIMULATION / VERIFICATION RUNNER
# -------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    # 1. Instantiate simple classification head representing LLM adapter / classification layer
    # Input features: 768 (standard Transformer hidden state dim), Output classes: 2
    feature_dim = 768
    num_classes = 2
    num_samples = 1280

    classifier_model = nn.Sequential(
        nn.Linear(feature_dim, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )

    # 2. Synthetic Dataset Generation
    synthetic_x = torch.randn(num_samples, feature_dim)
    synthetic_y = torch.randint(0, num_classes, (num_samples,))
    synthetic_dataset = TensorDataset(synthetic_x, synthetic_y)

    # 3. Initialize DP Fine-Tuner
    dp_tuner = DifferentiallyPrivateFineTuner(
        model=classifier_model,
        max_grad_norm=1.0,
        noise_multiplier=1.0,
        target_delta=1e-5
    )

    # 4. Run DP Fine-Tuning Loop
    report = dp_tuner.train_with_privacy(
        dataset=synthetic_dataset,
        batch_size=64,
        epochs=4,
        lr=2e-4
    )

    print("\n=== Final DP Fine-Tuning Privacy Report ===")
    print(report.model_dump_json(indent=2))
```

## 4. Operational Best Practices

* Parameter-Efficient Fine-Tuning (PEFT / LoRA): Combine DP-SGD with LoRA (Low-Rank Adaptation). By freezing base model weights and applying noise solely to low-rank matrices ($\mathbf{A}$ and $\mathbf{B}$), DP-SGD convergence speeds up drastically while maintaining utility.
* Avoid BatchNorm Layers: Batch Normalization creates cross-sample dependencies during the forward pass, violating the independence assumption of per-sample gradient privacy. Replace BatchNorm with GroupNorm or LayerNorm prior to attaching privacy engines.
* Virtual Batching for Subsampling Amplification: DP theory benefits from large batch sizes. Use virtual step aggregation (gradient accumulation) to simulate large physical batch sizes ($B \ge 512$) without running out of GPU VRAM.

$$\tilde{\mathbf{g}}_t = \frac{1}{B} \left( \sum_{i=1}^{B} \bar{\mathbf{g}}_t^{(i)} + \mathcal{N}\left(0, \sigma^2 C^2 \mathbf{I}\right) \right)$$

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \cdot \tilde{\mathbf{g}}_t$$
