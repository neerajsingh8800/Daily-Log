# 22. Model Quantization (Symmetric, Asymmetric, and PTQ vs. QAT)

As deep learning architectures grow larger, deploying them on edge devices or even cloud servers becomes heavily constrained by memory bandwidth and compute latency. **Model Quantization** reduces the bit-precision of weights and activations (e.g., from 32-bit Floating Point `FP32` down to 8-bit Integer `INT8` or 4-bit `INT4`), drastically accelerating inference speeds and slashing VRAM footprints while keeping accuracy degradation minimal.

---

## 1. Floating Point to Integer Mapping Mechanics

The goal of quantization is to map continuous real numbers $x \in [\alpha, \beta]$ into discrete bounded integer targets $q \in [q_{\min}, q_{\max}]$. For standard unsigned 8-bit integers (`INT8`), $q_{\min} = 0$ and $q_{\max} = 255$.

### A. Asymmetric Quantization (Affine Mapping)
Asymmetric quantization maps the minimum and maximum of the real range to the discrete integer range using a scale factor ($S$) and a zero-point ($Z$). This is highly efficient for asymmetric distributions like ReLU activations.

**Quantization Formula:**
$$q = \text{clip}\left(\left\lfloor \frac{x}{S} \right\rceil + Z, q_{\min}, q_{\max}\right)$$

**Dequantization Formula:**
$$\hat{x} = S \cdot (q - Z)$$

Where the **Scale ($S$)** and **Zero-point ($Z$)** are calculated as:
$$S = \frac{\beta - \alpha}{q_{\max} - q_{\min}}$$
$$Z = \text{clip}\left(\left\lfloor \frac{-alpha}{S} \right\rceil + q_{\min}, q_{\min}, q_{\max}\right)$$

### B. Symmetric Quantization
Symmetric quantization forces the real-world range to be symmetric around zero ($\alpha = -\beta$). The zero-point is locked statically to $Z = 0$, reducing computational tracking overhead during matrix operations.

**Quantization Formula:**
$$q = \text{clip}\left(\left\lfloor \frac{x}{S} \right\rceil, -q_{\max}, q_{\max}\right)$$

Where the Scale scale factor is bounded by the absolute maximum outlier value:
$$S = \frac{\max(|x|)}{q_{\max}}$$
---

## 2. Comparative Framework

| Metric | FP32 (Base Precision) | INT8 (Quantized Precision) | INT4 (Ultra-Compressed) |
| :--- | :--- | :--- | :--- |
| **Bit Width** | 32 bits | 8 bits | 4 bits |
| **Memory Footprint** | $4 \times$ baseline memory | $1 \times$ ($75\%$ reduction) | $0.5 \times$ ($87.5\%$ reduction) |
| **Hardware Compute** | Standard Floating Units | Highly optimized Vector/Tensor Units | specialized hardware custom registers |
| **Accuracy Profile** | 100% (Baseline baseline) | Minimal shift ($\sim 0.1-0.5\%$ loss) | Discernible degradation without QAT |

---

## 3. Implementation in Python (PyTorch)

This script demonstrates how to construct an explicit **Asymmetric (Affine) Quantizer and Dequantizer** engine from scratch in Python, complete with clipping logic and scale calculations.

```python
import torch

class AffineQuantizer:
    """
    Implements standard asymmetric uniform quantization math from scratch.
    """
    def __init__(self, bits=8):
        self.qmin = 0
        self.qmax = (2 ** bits) - 1

    def calculate_scale_and_zeropoint(self, tensor: torch.Tensor):
        # 1. Isolate ranges dynamically across current tensor profiles
        alpha = tensor.min().item()
        beta = tensor.max().item()
        
        # Guard against zero-division errors for completely uniform layers
        if alpha == beta:
            return 1.0, 0
            
        # 2. Derive scale according to integer step capacity bounds
        scale = (beta - alpha) / (self.qmax - self.qmin)
        
        # 3. Derive corresponding integer zero-point parameter
        zero_point = round((0.0 - alpha) / scale) + self.qmin
        zero_point = max(self.qmin, min(self.qmax, zero_point)) # Bound clipping
        
        return scale, int(zero_point)

    def quantize(self, tensor: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
        # Scale and shift input distribution
        q_tensor = torch.round(tensor / scale) + zero_point
        # Explicit tensor bounds clamp optimization execution step
        return torch.clamp(q_tensor, self.qmin, self.qmax).to(torch.uint8)

    def dequantize(self, q_tensor: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
        # Cast integer matrix arrays back to standard fractional floating vectors
        return scale * (q_tensor.float() - zero_point)

# Verification execution pass
if __name__ == "__main__":
    torch.manual_seed(42)
    quant_engine = AffineQuantizer(bits=8)
    
    # Simulate un-normalized ReLU layer weights or activation outputs
    mock_weights = torch.randn(5) * 10.0 + 3.0
    print(f"Original FP32 Matrix Tensor values:\n {mock_weights}\n")
    
    # Calculate mapping calibrations
    S, Z = quant_engine.calculate_scale_and_zeropoint(mock_weights)
    print(f"Computed Calibration Metrics -> Scale: {S:.4f}, Zero-Point: {Z}")
    
    # Run pipeline conversions
    quantized_ints = quant_engine.quantize(mock_weights, S, Z)
    dequantized_floats = quant_engine.dequantize(quantized_ints, S, Z)
    
    print(f"Quantized INT8 discrete coordinates:\n {quantized_ints}")
    print(f"Recovered Dequantized FP32 Matrix values:\n {dequantized_floats}\n")
    
    # Track calculation quantization noise loss footprint
    quantization_noise = torch.mean((mock_weights - dequantized_floats) ** 2)
    print(f"Mean Squared Quantization Error Noise: {quantization_noise.item():.6f}")
```

---
