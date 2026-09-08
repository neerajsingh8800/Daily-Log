# 07: Adversarial Attacks on Vision & Multimodal LLMs — Perturbations & Cross-Modal Jailbreaking

Vision-Language Models (VLMs) and multimodal architectures (e.g., LLaVA, BLIP-2, CLIP-based LLM backbones) merge high-dimensional visual feature spaces with autoregressive text decoding. While textual guardrails have matured, visual encoders introduce a vastly larger, continuous input surface vulnerable to gradient-based adversarial perturbations.

This module covers the mathematical mechanics of visual gradient attacks (FGSM, PGD), cross-modal projection exploits, visual adversarial patches, typographic attacks, and a production-grade PyTorch implementation of Projected Gradient Descent (PGD) targeting multimodal embeddings.

---

## 1. Theoretical Foundations

### 1.1 The Visual Encoder Vulnerability Surface
A multimodal LLM processes image inputs $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$ through a visual encoder $E_v$ (such as ViT or CLIP-ViT) to produce visual token embeddings $\mathbf{Z}_v = E_v(\mathbf{X}) \in \mathbb{R}^{N_v \times d_{\text{model}}}$. These embeddings are projected into the language model's latent space alongside textual token embeddings $\mathbf{Z}_t$:

$$\mathbf{Z}_{\text{input}} = \left[ \mathbf{W}_p \cdot E_v(\mathbf{X}) \;; \mathbf{Z}_t \right]$$

Because the visual encoder $E_v$ and projection matrix $\mathbf{W}_p$ are fully differentiable, an adversary can backpropagate gradients from the text decoder's loss function back to raw image pixel space $\mathbf{X}$.

### 1.2 Mathematical Formulation: Fast Gradient Sign Method (FGSM) & Projected Gradient Descent (PGD)

Adversarial perturbations craft an additive noise matrix $\boldsymbol{\delta} \in \mathbb{R}^{C \times H \times W}$ subject to an $L_\infty$ or $L_2$ norm bound $\Vert{}\boldsymbol{\delta}\Vert{} \le \epsilon$ such that the perturbed image $\tilde{\mathbf{X}} = \mathbf{X} + \boldsymbol{\delta}$ forces a target target token sequence $\mathbf{y}_{\text{target}}$:

1. **Fast Gradient Sign Method (FGSM)**: A single-step gradient update moving pixels in the direction of the loss gradient:

$$\boldsymbol{\delta} = \epsilon \cdot \text{sign}\left( \nabla_{\mathbf{X}} \mathcal{L}(f(\mathbf{X}, \mathbf{Z}_t), \mathbf{y}_{\text{target}}) \right)$$

2. **Projected Gradient Descent (PGD)**: An iterative extension of FGSM using step size $\alpha$, where perturbations are projected back onto the $\epsilon$-ball $\mathcal{S}$ and valid image domain $[0, 1]$ after each iteration $k$:

$$\boldsymbol{\delta}^{(k+1)} = \Pi_{\Vert{}\boldsymbol{\delta}\Vert{}_\infty \le \epsilon} \left( \boldsymbol{\delta}^{(k)} + \alpha \cdot \text{sign}\left( \nabla_{\mathbf{X}} \mathcal{L}(f(\mathbf{X} + \boldsymbol{\delta}^{(k)}, \mathbf{Z}_t), \mathbf{y}_{\text{target}}) \right) \right)$$

$$\tilde{\mathbf{X}}^{(k+1)} = \text{Clip}_{[0, 1]}\left( \mathbf{X} + \boldsymbol{\delta}^{(k+1)} \right)$$

---

## 2. Multimodal Attack Taxonomy

| Attack Vector | Primary Target | Mechanism | Human Perceptibility |
| :--- | :--- | :--- | :--- |
| **Iterative PGD Perturbation** | Visual Encoder ($E_v$) | Imperceptible $L_\infty$-bounded noise aligned with adversarial text vectors. | Imperceptible (Invisible noise) |
| **Adversarial Patch Attack** | Visual Attention Heads | High-intensity localized image patch overriding global scene semantics. | High (Visible patch) |
| **Typographic Attack** | OCR / Text Tokenizer | Rendering text onto image surfaces to exploit visual text-reading capabilities. | High (Visible text) |
| **Cross-Modal Embedding Hijack**| Projection Matrix ($\mathbf{W}_p$) | Perturbing image representations to align vector cosine similarity with banned system prompts. | Imperceptible |

---

## 3. Production PyTorch Implementation

This module implements a **Projected Gradient Descent (PGD) Adversarial Generator** targeting visual feature representations. It computes pixel gradients relative to a target embedding vector and applies iterative $L_\infty$-bounded updates.

### Prerequisites

```bash
pip install torch torchvision pydantic
```

### Python Implementation (multimodal_pgd_attack.py)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from typing import Tuple, Dict


class AttackVerificationReport(BaseModel):
    is_attack_successful: bool
    iterations_completed: int
    final_loss: float
    initial_cosine_similarity: float
    final_cosine_similarity: float
    epsilon_norm: float


class MockVisionEncoder(nn.Module):
    """Simulates a Vision Encoder (e.g., ViT) converting image pixels to latent embeddings."""
    def __init__(self, in_channels: int = 3, embed_dim: int = 512):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(64 * 8 * 8, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = F.relu(self.conv(x))
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)
        embed = self.fc(feat)
        return F.normalize(embed, p=2, dim=-1)


class MultimodalPGDAttacker:
    """Projected Gradient Descent (PGD) Adversarial Generator for Visual Inputs."""

    def __init__(
        self,
        vision_encoder: nn.Module,
        epsilon: float = 8 / 255,
        alpha: float = 2 / 255,
        num_iter: int = 20
    ):
        self.encoder = vision_encoder
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter

    def generate_adversarial_image(
        self,
        clean_image: torch.Tensor,
        target_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, AttackVerificationReport]:
        """
        Executes PGD iteration to perturb clean_image such that its latent representation 
        aligns with target_embedding.
        """
        self.encoder.eval()
        
        # Ensure image has batch dimension
        if clean_image.dim() == 3:
            clean_image = clean_image.unsqueeze(0)

        # Clone clean image and initialize perturbation delta
        adv_image = clean_image.clone().detach().requires_grad_(True)
        
        # Calculate initial similarity
        with torch.no_grad():
            init_embed = self.encoder(clean_image)
            init_sim = F.cosine_similarity(init_embed, target_embedding).item()

        print(f"--- Starting PGD Adversarial Attack (ε = {self.epsilon:.4f}, α = {self.alpha:.4f}, Steps = {self.num_iter}) ---")
        print(f"Initial Embed Cosine Similarity to Target: {init_sim:.4f}")

        final_loss_val = 0.0
        
        for i in range(1, self.num_iter + 1):
            adv_image.requires_grad_(True)
            current_embed = self.encoder(adv_image)

            # Target loss: Maximize cosine similarity with target adversarial embedding vector
            # Equivalent to minimizing negative cosine similarity
            loss = -F.cosine_similarity(current_embed, target_embedding).mean()
            
            # Backpropagate gradients back to image pixel space
            self.encoder.zero_grad()
            loss.backward()

            # Capture gradient direction
            grad_sign = adv_image.grad.data.sign()

            # PGD Update Step with L_infinity norm projection
            with torch.no_grad():
                adv_image = adv_image - self.alpha * grad_sign
                eta = torch.clamp(adv_image - clean_image, min=-self.epsilon, max=self.epsilon)
                adv_image = torch.clamp(clean_image + eta, min=0.0, max=1.0).detach()

            final_loss_val = loss.item()
            if i % 5 == 0 or i == self.num_iter:
                sim = -final_loss_val
                print(f"Step {i:02d}/{self.num_iter:02d} | Loss: {final_loss_val:.4f} | Target Cosine Sim: {sim:.4f}")

        # Final evaluation
        with torch.no_grad():
            final_embed = self.encoder(adv_image)
            final_sim = F.cosine_similarity(final_embed, target_embedding).item()

        is_successful = final_sim > 0.85

        report = AttackVerificationReport(
            is_attack_successful=is_successful,
            iterations_completed=self.num_iter,
            final_loss=final_loss_val,
            initial_cosine_similarity=init_sim,
            final_cosine_similarity=final_sim,
            epsilon_norm=self.epsilon
        )

        return adv_image, report


# -------------------------------------------------------------------
# SIMULATION / VERIFICATION RUNNER
# -------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    # 1. Instantiate Vision Encoder Simulation
    encoder = MockVisionEncoder(in_channels=3, embed_dim=512)

    # 2. Generate clean dummy image [Batch=1, Channels=3, Height=224, Width=224]
    clean_img = torch.rand(1, 3, 224, 224)

    # 3. Create target adversarial embedding vector (e.g., representation of banned jailbreak string)
    target_adv_embedding = F.normalize(torch.randn(1, 512), p=2, dim=-1)

    # 4. Instantiate and run PGD attacker
    attacker = MultimodalPGDAttacker(
        vision_encoder=encoder,
        epsilon=16 / 255,  # Max perturbation magnitude
        alpha=2 / 255,     # Step size per iteration
        num_iter=20
    )

    perturbed_img, attack_report = attacker.generate_adversarial_image(clean_img, target_adv_embedding)

    print("\n=== Attack Verification Summary ===")
    print(attack_report.model_dump_json(indent=2))
```

## 4. Operational Best Practices

* Input Pre-processing & Gaussian Blurring: Apply lightweight, non-differentiable pre-processing filters (such as random JPEG compression, spatial smoothing, or median filtering) to input images before passing them to the visual encoder to disrupt fine-grained $L_\infty$ pixel noise.
* Randomized Resizing: Randomly scale visual inputs prior to model encoding during inference pipelines to alter spatial gradient alignments exploited by fixed-grid PGD attacks.
* Cross-Modal Latent Space Guardrails: Apply intermediate safety classification heads directly to projected visual token embeddings ($\mathbf{Z}_v = \mathbf{W}_p \cdot E_v(\mathbf{X})$) before feeding them into the autoregressive text generation loop.
