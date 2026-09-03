# 02: Token-Level Guardrails & Classification Heads — Real-Time Latent Space Interception

Traditional API-level guardrails rely on secondary LLM calls or post-hoc text parsing to evaluate model inputs and completions. This introduces high P99 latency overhead and fails to intercept unsafe content as it streams to the client.

Modern production guardrails operate **at the token and latent-space level**. By attaching lightweight linear classification heads to intermediate hidden state representations $\mathbf{h}_t^{(l)}$ or modifying logit distributions during the autoregressive decoding loop, systems can detect jailbreaks, policy violations, and hallucinated tokens with sub-millisecond overhead per step.

---

## 1. Theoretical Foundations

### 1.1 Latent Space Probing & Intermediate Representations
In an autoregressive transformer with $L$ layers, the forward pass at generation step $t$ produces a sequence of hidden state vectors:

$$\mathbf{h}_t^{(0)}, \mathbf{h}_t^{(1)}, \dots, \mathbf{h}_t^{(L)} \quad \text{where} \quad \mathbf{h}_t^{(l)} \in \mathbb{R}^{d_{\text{model}}}$$

While upper layers ($\mathbf{h}_t^{(L)}$) specialize in predicting the next vocabulary token $v_t \in V$, intermediate hidden states ($\mathbf{h}_t^{(l)}$ where $l \approx \frac{2}{3}L$) contain dense semantic and intent representations. An auxiliary Linear Probe (Classification Head) $\mathbf{W}_{\text{guard}} \in \mathbb{R}^{K \times d_{\text{model}}}$ projects these intermediate states into binary or multi-class safety logits:

$$\mathbf{z}_{\text{safety}} = \sigma\left(\mathbf{W}_{\text{guard}} \mathbf{h}_t^{(l)} + \mathbf{b}_{\text{guard}}\right)$$

Where $K$ represents safety policy classes (e.g., `[SAFE, JAILBREAK, HARMFUL, PRIVACY_LEAK]`).

### 1.2 Interception Mechanics: Early Exit & Logit Masking

1. **Early Exit Interception**: If $\mathbf{z}_{\text{safety}}$ exceeds a configured threat threshold at step $t$, the remaining forward pass layers ($l+1 \dots L$) and autoregressive generation loops are aborted immediately.
2. **Dynamic Vocabulary Modification**: If a specific category of output is restricted (e.g., generating code execution blocks or private data patterns), the logit processor sets vocabulary indices associated with restricted tokens to $-\infty$ prior to Softmax:

$$\tilde{z}_t(i) = \begin{cases} z_t(i) & \text{if } i \in V_{\text{allowed}} \\ -\infty & \text{if } i \in V_{\text{restricted}} \end{cases}$$

---

## 2. Guardrail Approaches Comparison

| Metric / Feature | External Safety API (e.g., Llama-Guard) | Token Regex/Keyword Filtering | Latent Probe Classification Head | Logit Bias Vector Masking |
| :--- | :--- | :--- | :--- | :--- |
| **P99 Latency Overhead** | +150ms – 500ms (Full Forward Pass) | < 1ms | **< 0.5ms (Single Linear Layer)** | < 0.2ms |
| **Detection Timing** | Post-generation or Pre-generation | Post-token emission | **Mid-generation (Streaming)** | Pre-token selection |
| **Semantic Awareness** | High | Low (Syntax only) | **High (Transformer Latent Feature)**| Medium (Token ID boundaries) |
| **Compute Overhead** | High (2x Inference GPUs) | Minimal | **Negligible (<0.1% FLOPs)** | Negligible |

---

## 3. Production PyTorch Implementation

This module implements a **PyTorch Latent Space Safety Classification Head** attached to a transformer model's intermediate layer, combined with a **Streaming Token Interceptor** that halts autoregressive generation when jailbreaks or policy violations are detected.

### Prerequisites

```bash
pip install torch transformers pydantic
```
### Python Implementation (token_level_guardrails.py)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel


class GuardrailVerdict(BaseModel):
    is_safe: bool
    threat_category: str
    confidence: float
    intercepted_at_step: int


class LatentSafetyClassifierHead(nn.Module):
    """Auxiliary Classification Head operating on intermediate Transformer hidden states."""
    
    def __init__(self, d_model: int, num_classes: int = 4):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model // 2)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model // 2)
        self.classifier = nn.Linear(d_model // 2, num_classes)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states shape: [batch_size, sequence_length, d_model]
        # Probe the final sequence token representation
        last_token_repr = hidden_states[:, -1, :]
        
        x = self.dense(last_token_repr)
        x = self.activation(x)
        x = self.layer_norm(x)
        logits = self.classifier(x)
        return logits


class LatentGuardrailEngine:
    """Manages real-time streaming interception during autoregressive decoding."""
    
    CLASSES = ["SAFE", "JAILBREAK", "HARMFUL_CONTENT", "PRIVACY_LEAK"]
    
    def __init__(self, d_model: int, target_layer_idx: int = 16, threshold: float = 0.75):
        self.d_model = d_model
        self.target_layer_idx = target_layer_idx
        self.threshold = threshold
        
        # Instantiate classification probe
        self.safety_head = LatentSafetyClassifierHead(d_model=d_model, num_classes=len(self.CLASSES))
        self.safety_head.eval()

    def evaluate_intermediate_state(self, intermediate_hidden_state: torch.Tensor) -> Tuple[bool, str, float]:
        """Evaluates intermediate hidden state tensor [1, 1, d_model] at step t."""
        with torch.no_grad():
            logits = self.safety_head(intermediate_hidden_state)
            probs = F.softmax(logits, dim=-1).squeeze(0)
            
            top_prob, top_class_idx = torch.max(probs, dim=-1)
            predicted_class = self.CLASSES[top_class_idx.item()]
            
            is_safe = True
            if predicted_class != "SAFE" and top_prob.item() >= self.threshold:
                is_safe = False
                
            return is_safe, predicted_class, top_prob.item()

    def simulate_constrained_streaming_generation(
        self, 
        sequence_length: int = 10
    ) -> GuardrailVerdict:
        """Simulates an autoregressive decoding loop with token-level safety checks."""
        print(f"--- Starting Generation Loop with Latent Safety Probing (Target Layer {self.target_layer_idx}) ---")
        
        for step in range(1, sequence_length + 1):
            # Simulate a forward pass hidden state tensor at step t
            # In production, this is captured via PyTorch forward hooks on model.layers[target_layer_idx]
            simulated_hidden_state = torch.randn(1, 1, self.d_model)
            
            # Inject a simulated jailbreak signal at step 5 for testing
            if step == 5:
                # Add directional bias vector representing jailbreak activation pattern
                simulated_hidden_state += torch.ones_like(simulated_hidden_state) * 2.5

            is_safe, category, confidence = self.evaluate_intermediate_state(simulated_hidden_state)
            
            print(f"Step {step:02d} | Status: {'[SAFE]' if is_safe else '[INTERCEPTED]'} | Predicted: {category:<15} | Conf: {confidence:.4f}")
            
            if not is_safe:
                print(f"\n[ALERT] Guardrail triggered! Halting generation at token position {step}.")
                return GuardrailVerdict(
                    is_safe=False,
                    threat_category=category,
                    confidence=confidence,
                    intercepted_at_step=step
                )
                
        return GuardrailVerdict(
            is_safe=True,
            threat_category="SAFE",
            confidence=0.99,
            intercepted_at_step=sequence_length
        )


# -------------------------------------------------------------------
# SIMULATION / VERIFICATION RUNNER
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Standard hidden dimension for 7B/8B parameter models (e.g., LLaMA-3 8B d_model = 4096)
    D_MODEL = 4096
    
    engine = LatentGuardrailEngine(d_model=D_MODEL, target_layer_idx=16, threshold=0.70)
    
    # Run streaming inference simulation
    verdict = engine.simulate_constrained_streaming_generation(sequence_length=8)
    
    print("\n=== Final Engine Execution Verdict ===")
    print(verdict.model_dump_json(indent=2))
```

## 4. Operational Best Practices

* Layer Selection Calibration: Probe intermediate layers located between $60\%\text{--}75\%$ of the total transformer depth. Early layers ($<30\%$) lack sufficient semantic convergence, while final layers ($>90\%$) are overly specialized for next-token vocabulary logits.
* Forward Hook Optimization: Attach PyTorch forward hooks (register_forward_hook) directly to the specified transformer layer block to avoid copying tensor memory during inference execution.
* Dynamic Logit Soft-Masking: Rather than completely aborting generation on borderline threats ($0.50 \le P(\text{threat}) \le 0.70$), dynamically increase the logit penalty vector on sensitive token groups to gently steer the output back into safe policy boundaries.
