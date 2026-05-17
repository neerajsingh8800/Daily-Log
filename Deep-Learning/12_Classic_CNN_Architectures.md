# 12. Classic CNN Architectures

To build a deep understanding of Computer Vision, you must study how researchers historically scaled network depth. This module covers the core theories, architectural components, and structural evolution from **LeNet-5** to **VGG-16**, and finally the breakthrough **ResNet**.

---

## 1. Architectural Evolution Overview

### A. LeNet-5 (1998)
Introduced by Yann LeCun for handwritten digit recognition (MNIST). It established the classic pattern: **Convolution $\rightarrow$ Pooling $\rightarrow$ Fully Connected**.
* **Key Idea:** Proved that local feature extraction coupled with downsampling provides basic spatial invariance.
* **Limitations:** Used average pooling and tanh/sigmoid activations, causing severe vanishing gradients in deeper configurations.

### B. VGG-16 (2014)
Developed by the Visual Geometry Group at Oxford. It brought systemic simplicity to network design.
* **Key Idea:** Replaced large convolutional filters (like AlexNet's $11 \times 11$) with stacks of very small **$3 \times 3$ filters**.
* **The Math (Effective Receptive Field):** Stacking two $3 \times 3$ layers creates an effective receptive field of a $5 \times 5$ layer; stacking three simulates a $7 \times 7$ layer. However, three $3 \times 3$ layers require fewer parameters ($3 \times (3^2 \cdot C^2) = 27C^2$) compared to one $7 \times 7$ layer ($1 \times (7^2 \cdot C^2) = 49C^2$) and include more non-linear rectification layers.

### C. ResNet (2015)
As networks get deeper, accuracy saturates and then degrades rapidly—not because of overfitting, but because optimization algorithms struggle to propagate gradients back through dozens of layers. Kaiming He introduced **Residual Learning** to bypass this bottleneck.

* **Formulation:** Instead of forcing stacked layers to fit an underlying mapping $\mathcal{H}(\mathbf{x})$, we explicitly let these layers fit a residual mapping $\mathcal{F}(\mathbf{x}) := \mathcal{H}(\mathbf{x}) - \mathbf{x}$. The original mapping is recast into:

$$\mathcal{H}(\mathbf{x}) = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

* **Skip Connections:** The operation $\mathcal{F}(\mathbf{x}) + \mathbf{x}$ is performed via a shortcut connection (identity mapping). If the gradient of the loss with respect to the output is passed backward, the identity connection allows the gradient to flow directly back to earlier layers unimpeded:

$$\frac{\partial \mathcal{H}}{\partial \mathbf{x}} = \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I}$$

Even if the learned weight transformations ($\frac{\partial \mathcal{F}}{\partial \mathbf{x}}$) vanish toward zero, the total identity gradient $\mathbf{I}$ remains active.

---

## 2. Structural Comparison

| Architecture | Year | Deepest Variant | Key Innovation | Primary Activation |
| :--- | :--- | :--- | :--- | :--- |
| **LeNet-5** | 1998 | 5 Layers | Convolution/Pooling stack | Tanh / Sigmoid |
| **VGG-16** | 2014 | 19 Layers | Small $3 \times 3$ homogeneous filters | ReLU |
| **ResNet** | 2015 | 152 Layers | Residual Blocks / Skip Connections | ReLU |

---

## 3. Implementation in Python (PyTorch)

This script demonstrates how to construct a standard **Residual Block** and assemble a deep custom CNN using PyTorch.

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A classic ResNet Basic Block using two 3x3 convolutions 
    and an explicit identity skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection to handle dimension adjustments if stride != 1
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # The core residual operation: H(x) = F(x) + x
        out += identity
        out = self.relu(out)
        return out

class MiniResNet(nn.Module):
    """
    A scaled-down version of ResNet to demonstrate pipeline building.
    """
    def __init__(self, num_classes=10):
        super(MiniResNet, self).__init__()
        self.in_channels = 64
        
        # Initial Feature Extraction
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stacking Residual Blocks
        self.layer1 = ResidualBlock(64, 64, stride=1)
        self.layer2 = ResidualBlock(64, 128, stride=2) # Downsamples spatial dimensions
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# Verification pass
if __name__ == "__main__":
    model = MiniResNet(num_classes=10)
    mock_batch = torch.randn(4, 3, 32, 32) # Batch size 4, RGB, 32x32 image
    output = model(mock_batch)
    print(f"Input Tensor Shape:  {mock_batch.shape}")
    print(f"Output Tensor Shape: {output.shape}")
```
