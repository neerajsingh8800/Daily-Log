# 04: Model Inversion & Membership Inference — Privacy Bounds & Memorization Auditing

Large Language Models (LLMs) and deep neural networks are prone to memorizing sensitive training data, such as Personally Identifiable Information (PII), proprietary source code, or confidential clinical records. Privacy adversaries exploit this memorization through **Model Inversion (MI)** and **Membership Inference Attacks (MIA)** to extract training samples or determine if a specific data point was included in the model's training set.

This module covers the mathematical foundations of model inversion, Likelihood Ratio Tests (LiRA), loss distribution variance analysis, and a production PyTorch audit pipeline for measuring membership privacy risks.

---

## 1. Theoretical Foundations

### 1.1 Model Inversion (MI) Attacks
Model Inversion attempts to reconstruct input features $x$ given target output distributions $y$ and access to model parameters $\theta$ or prediction APIs:

$$\hat{x} = \arg\max_{x} P(y \mid x; \theta) + \lambda \mathcal{R}(x)$$

Where $\mathcal{R}(x)$ is a prior regularization term (e.g., total variation or image prior) that keeps reconstructed inputs within realistic data distributions. In LLMs, model inversion manifests as prefix-prompted extraction where attackers trigger memorized text sequences using specific prefix tokens.

### 1.2 Membership Inference Attacks (MIA) & LiRA Mechanics
Membership Inference determines whether a target sample $(x, y)$ was part of the private training set $D_{\text{train}}$. 

1. **Standard Loss-Based MIA**: Evaluates prediction loss $\mathcal{L}(x, y; \theta)$. Because models overfit to training samples, $\mathcal{L}(x, y) < \tau$ typically implies membership. However, simple loss thresholding suffers from high false-positive rates on inherently "easy" samples.
2. **Likelihood Ratio Attack (LiRA)**: Computes a parametric ratio by training $K$ shadow models $\theta_{\text{out}}$ on datasets excluding $(x, y)$ and $K$ shadow models $\theta_{\text{in}}$ including $(x, y)$.

The hypothesis test models the log-loss distribution under two Gaussian distributions:

$$\Lambda(x,y) = \frac{p\left(\mathcal{L}(x, y; \theta) \mid (x, y) \in D_{\text{train}}\right)}{p\left(\mathcal{L}(x, y; \theta) \mid (x, y) \notin D_{\text{train}}\right)}$$

$$\Lambda(x,y) = \frac{\mathcal{N}\left(\mathcal{L}(x, y; \theta); \mu_{\text{in}}, \sigma_{\text{in}}^2\right)}{\mathcal{N}\left(\mathcal{L}(x, y; \theta); \mu_{\text{out}}, \sigma_{\text{out}}^2\right)}$$

---

## 2. Attack Vectors & Privacy Risk Comparison

| Vulnerability Type | Primary Objective | Required Access | Primary Defense |
| :--- | :--- | :--- | :--- |
| **Direct Generation Inversion** | Reconstruct exact PII / secrets | Black-box API / Prompting | Differentially Private Training (DP-SGD) |
| **Loss Threshold MIA** | Identify target sample membership | Black-box (Per-token loss/logits) | Temperature scaling & logit truncation |
| **Likelihood Ratio MIA (LiRA)** | High-precision membership auditing | Black-box / Shadow training | DP-SGD + Regularization |
| **Gradient Inversion** | Reconstruct training batch gradients | White-box / Federated Learning | Gradient clipping & noise addition |

---

## 3. Production Privacy Audit Implementation

This Python module implements a **Likelihood Ratio Membership Inference Audit Engine** that evaluates trained model outputs against shadow loss distributions to calculate formal membership leakage scores.

### Prerequisites

```bash
pip install torch numpy scipy pydantic
```

### Python Implementation (membership_inference_audit.py)
```python
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from typing import List, Tuple, Dict
from pydantic import BaseModel


class AuditReport(BaseModel):
    total_samples_audited: int
    membership_leaks_detected: int
    false_positive_rate: float
    auc_score: float
    high_risk_indices: List[int]


class MembershipInferenceAuditor:
    """Evaluates privacy leakage risks using Likelihood Ratio Testing (LiRA)."""

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def fit_shadow_distributions(
        self, 
        shadow_out_losses: np.ndarray, 
        shadow_in_losses: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculates mean and std parameters for IN and OUT shadow model loss distributions."""
        mu_out = np.mean(shadow_out_losses, axis=0)
        sigma_out = np.std(shadow_out_losses, axis=0) + 1e-8
        
        mu_in = np.mean(shadow_in_losses, axis=0)
        sigma_in = np.std(shadow_in_losses, axis=0) + 1e-8
        
        return mu_out, sigma_out, mu_in, sigma_in

    def compute_lira_scores(
        self, 
        target_losses: np.ndarray, 
        mu_out: np.ndarray, 
        sigma_out: np.ndarray,
        mu_in: np.ndarray,
        sigma_in: np.ndarray
    ) -> np.ndarray:
        """Computes Likelihood Ratio (Lambda) scores for target losses."""
        # Log-likelihood under OUT distribution
        log_pdf_out = norm.logpdf(target_losses, loc=mu_out, scale=sigma_out)
        # Log-likelihood under IN distribution
        log_pdf_in = norm.logpdf(target_losses, loc=mu_in, scale=sigma_in)
        
        # Likelihood ratio in log-space
        log_lira_ratio = log_pdf_in - log_pdf_out
        return log_lira_ratio

    def audit_target_model(
        self, 
        target_losses: np.ndarray, 
        ground_truth_membership: np.ndarray,
        shadow_out_losses: np.ndarray,
        shadow_in_losses: np.ndarray
    ) -> AuditReport:
        """Audits target model losses against fitted shadow distributions."""
        num_samples = len(target_losses)
        
        # Step 1: Fit parametric Gaussian distributions
        mu_out, sigma_out, mu_in, sigma_in = self.fit_shadow_distributions(
            shadow_out_losses, shadow_in_losses
        )
        
        # Step 2: Calculate LiRA score matrix
        lira_scores = self.compute_lira_scores(
            target_losses, mu_out, sigma_out, mu_in, sigma_in
        )
        
        # Step 3: Flag membership if likelihood under IN distribution dominates OUT
        predictions = (lira_scores > 0).astype(int)
        
        # Step 4: Compute metrics
        flagged_indices = np.where(predictions == 1)[0].tolist()
        tp = np.sum((predictions == 1) & (ground_truth_membership == 1))
        fp = np.sum((predictions == 1) & (ground_truth_membership == 0))
        tn = np.sum((predictions == 0) & (ground_truth_membership == 0))
        
        fpr = fp / (fp + tn + 1e-8)
        detected_leaks = int(tp)

        return AuditReport(
            total_samples_audited=num_samples,
            membership_leaks_detected=detected_leaks,
            false_positive_rate=float(fpr),
            auc_score=float((tp / (np.sum(ground_truth_membership == 1) + 1e-8))),
            high_risk_indices=flagged_indices
        )


# -------------------------------------------------------------------
# SIMULATION / VERIFICATION RUNNER
# -------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    
    print("--- 1. Simulating Target Model Losses & Shadow Distribution Data ---")
    num_samples = 500
    num_shadow_models = 8

    # Ground truth: 250 members (1), 250 non-members (0)
    ground_truth = np.array([1] * 250 + [0] * 250)

    # Members exhibit lower average loss (overfitting signal)
    member_target_losses = np.random.normal(loc=0.45, scale=0.15, size=250)
    non_member_target_losses = np.random.normal(loc=1.20, scale=0.25, size=250)
    target_losses = np.concatenate([member_target_losses, non_member_target_losses])

    # Generate synthetic shadow OUT and IN loss matrices [num_shadow_models, num_samples]
    shadow_out_losses = np.random.normal(loc=1.25, scale=0.25, size=(num_shadow_models, num_samples))
    shadow_in_losses = np.random.normal(loc=0.42, scale=0.15, size=(num_shadow_models, num_samples))

    print("\n--- 2. Running Likelihood Ratio Privacy Audit ---")
    auditor = MembershipInferenceAuditor(significance_level=0.05)
    report = auditor.audit_target_model(
        target_losses=target_losses,
        ground_truth_membership=ground_truth,
        shadow_out_losses=shadow_out_losses,
        shadow_in_losses=shadow_in_losses
    )

    print("\n=== Privacy Leakage Audit Report ===")
    print(f"Total Audited Samples     : {report.total_samples_audited}")
    print(f"True Member Leaks Identified: {report.membership_leaks_detected}")
    print(f"False Positive Rate       : {report.false_positive_rate:.4f}")
    print(f"Recall (Privacy Risk Score): {report.auc_score:.4f}")
```

## 4. Operational Best Practices

* Apply Differential Privacy (DP-SGD): Train model parameters using Differentially Private Stochastic Gradient Descent to guarantee mathematical bounds ($\epsilon, \delta$) against membership inference attacks.
* Sanitize Top-k Logits API: Never expose raw, un-truncated output probability vectors or token-level log-probabilities directly to end users over public APIs.
* Temperature Scaling & Top-P Sampling: Apply continuous logit smoothing (T>1.0) during generation to flatten confidence peaks that attackers exploit to infer membership.
