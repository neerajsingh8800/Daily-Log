# 03: Data Poisoning & Backdoor Mitigation — Spectral Signatures & Activation Clustering

As enterprise Large Language Models (LLMs) and instruction-tuned agents rely on open-source web scrapes, user feedback loops (RLHF), and third-party datasets, data integrity has become a core vulnerability. **Data Poisoning** occurs when an adversary inserts malicious or subtly corrupted samples into training or fine-tuning datasets.

This module covers clean-label backdoor attacks, Trojan triggers, activation clustering mechanics, spectral signature analysis, and a production-grade Python anomaly detector for dataset auditing.

---

## 1. Theoretical Foundations

### 1.1 Taxonomy of Data Poisoning Attacks

1. **Clean-Label Backdoor Attacks**:
   * **Mechanics**: The attacker injects samples whose human-readable label matches the ground truth, but embeds a hidden trigger string (e.g., `cf_trigger_xyz`) or specific syntactic pattern.
   * **Objective**: During normal inference, the model behaves standardly. When the trigger pattern is presented in an input, the model activates a backdoor behavior (e.g., forcing a specific tool call or bypassing authorization).

2. **Targeted Availability Attacks (Model Degradation)**:
   * **Mechanics**: Noise or adversarial perturbations are introduced into fine-tuning datasets to maximize validation loss on targeted downstream tasks.
   * **Objective**: Degrade performance on specific domain capabilities (e.g., medical query parsing or code generation) without crashing overall training convergence.

### 1.2 Mathematical Detection via Spectral Signatures & Activation Clustering

Backdoor triggers cause model representations in intermediate activations to cluster into distinct sub-spaces. Given representation matrix $\mathbf{R} \in \mathbb{R}^{N \times d}$ for a target dataset class (where $N$ is samples and $d$ is feature dimension):

1. **Mean Centering**: Compute class mean vector $\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{r}_i$ and centered representations $\mathbf{Y} = \mathbf{R} - \boldsymbol{\mu}$.
2. **Singular Value Decomposition (SVD)**: Compute top right singular vector $\mathbf{v}_1$ corresponding to the largest singular value of $\mathbf{Y}$:

$$\mathbf{Y} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

3. **Outlier Score Calculation**: Project each sample representation onto the top singular vector:

$$s_i = \left( (\mathbf{r}_i - \boldsymbol{\mu}) \cdot \mathbf{v}_1 \right)^2$$

Samples with high projection scores $s_i$ exhibit significant variance alignment with the poisoning signal and are flagged as malicious backdoor candidates.

---

## 2. Mitigation Strategies Comparison

| Strategy | Detection Phase | Compute Overhead | false Positive Rate | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Heuristic Pattern Filtering** | Pre-training (Static) | Low | High | Simple string/trigger matching |
| **Activation Clustering (SVD)** | Pre-training (Dynamic) | Medium ($O(Nd^2)$) | Low | Clean-label backdoor detection |
| **Differential Privacy (DP-SGD)** | During Training | High (Gradient clipping) | N/A (Limits memorization) | Preventing exact trigger memorization |
| **Fine-Pruning / Model Unlearning**| Post-training | High | Medium | Removing backdoors from pre-trained weights |

---

## 3. Production Defense Implementation

This Python module implements an **Activation Clustering & Spectral Signature Anomaly Detector**. It extracts feature embeddings, performs Singular Value Decomposition (SVD), computes outlier projection scores, and isolates poisoned samples from training datasets.

### Prerequisites

```bash
pip install numpy scikit-learn pydantic
```

### Python Implementation (data_poisoning_detector.py)
```python
import numpy as np
from typing import List, Tuple, Dict
from pydantic import BaseModel
from sklearn.decomposition import TruncatedSVD


class PoisonDetectionResult(BaseModel):
    total_samples: int
    poisoned_detected: int
    clean_samples: int
    flagged_indices: List[int]
    outlier_scores: List[float]


class SpectralPoisonDetector:
    """Detects clean-label data poisoning using SVD-based Spectral Signature Analysis."""

    def __init__(self, percentile_threshold: float = 92.0):
        self.percentile_threshold = percentile_threshold

    def compute_spectral_signatures(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Computes outlier scores by projecting centered embeddings onto 
        the top right singular vector.
        """
        # Step 1: Center feature matrix
        mean_vector = np.mean(embeddings, axis=0)
        centered_matrix = embeddings - mean_vector

        # Step 2: Compute SVD to find top singular vector (v1)
        svd = TruncatedSVD(n_components=1, algorithm='randomized', random_state=42)
        svd.fit(centered_matrix)
        top_singular_vector = svd.components_[0]

        # Step 3: Compute squared projection scores s_i = ((r_i - mu) . v1)^2
        projections = np.dot(centered_matrix, top_singular_vector)
        outlier_scores = projections ** 2
        
        return outlier_scores

    def audit_dataset(self, embeddings: np.ndarray) -> PoisonDetectionResult:
        """Audits dataset embeddings and returns flagged indices."""
        num_samples = embeddings.shape[0]
        outlier_scores = self.compute_spectral_signatures(embeddings)

        # Calculate threshold dynamically based on percentile
        threshold = np.percentile(outlier_scores, self.percentile_threshold)
        
        flagged_indices = []
        for idx, score in enumerate(outlier_scores):
            if score > threshold:
                flagged_indices.append(idx)

        return PoisonDetectionResult(
            total_samples=num_samples,
            poisoned_detected=len(flagged_indices),
            clean_samples=num_samples - len(flagged_indices),
            flagged_indices=flagged_indices,
            outlier_scores=outlier_scores.tolist()
        )


# -------------------------------------------------------------------
# SIMULATION / VERIFICATION RUNNER
# -------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    
    print("--- 1. Simulating Training Dataset Embeddings (Clean + Poisoned) ---")
    num_clean = 900
    num_poisoned = 100
    embedding_dim = 128

    # Generate normal cluster for clean data
    clean_embeddings = np.random.normal(loc=0.0, scale=1.0, size=(num_clean, embedding_dim))

    # Generate poisoned data with directional shift (backdoor trigger signal)
    poison_signal = np.ones(embedding_dim) * 2.5
    poisoned_embeddings = np.random.normal(loc=0.0, scale=1.0, size=(num_poisoned, embedding_dim)) + poison_signal

    # Stack full dataset
    full_dataset = np.vstack([clean_embeddings, poisoned_embeddings])
    
    print(f"Total Dataset Size: {full_dataset.shape[0]} samples (Clean: {num_clean}, Poisoned: {num_poisoned})")

    print("\n--- 2. Running Spectral Signature Anomaly Detection ---")
    detector = SpectralPoisonDetector(percentile_threshold=90.0)
    result = detector.audit_dataset(full_dataset)

    print(f"\nAudit Summary:")
    print(f"Total Analyzed    : {result.total_samples}")
    print(f"Flagged Outliers  : {result.poisoned_detected}")
    print(f"Clean Retained    : {result.clean_samples}")

    # Check accuracy of detection against simulated ground truth (poisoned indices start at 900)
    correctly_flagged = sum(1 for idx in result.flagged_indices if idx >= num_clean)
    print(f"True Positive Detection Rate: {correctly_flagged / num_poisoned * 100:.2f}%")
```

## 4. Operational Best Practices

* Audit Intermediate Layer Activations: When analyzing large datasets, extract feature embeddings from intermediate layers of a pre-trained encoder (e.g., layer $\frac{3}{4}L$) rather than raw input token IDs.
* Pre-Sanitize Instruction Datasets: Apply spectral signature detection prior to executing Supervised Fine-Tuning (SFT) or Reinforcement Learning from Human Feedback (RLHF).
* Maintain Provenance Signatures: Enforce cryptographic SHA-256 hashing and digital signatures (e.g., Sigstore) across dataset releases to detect unauthorized data modifications in the storage pipeline.
