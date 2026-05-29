# Vision Transformers (ViT)

## 1. The Paradigm Shift: From CNNs to Transformers in Vision
For years, Convolutional Neural Networks (CNNs) were the undisputed standard for computer vision. CNNs rely on two inherent inductive biases:
* **Translation Equivariance:** If an object shifts in an image, its feature representation shifts similarly.
* **Locality:** Nearby pixels are highly correlated and should be processed together.

While these biases allow CNNs to learn efficiently on small datasets, they limit the network's ability to capture global context instantly; a pixel in the top-left corner can only interact with a pixel in the bottom-right corner after passing through many deep convolutional layers.

**Vision Transformers (ViT)**, introduced by Dosovitskiy et al. in 2020 (*"An Image is Worth 16x16 Words"*), discard these inductive biases. By applying the standard Transformer encoder directly to sequences of image patches, ViT allows every part of an image to interact with every other part right from the very first layer. When trained on massive datasets (like JFT-300M), ViT completely outperforms state-of-the-art CNNs, demonstrating superior global feature representation.

---

## 2. Mathematical Formulation & Architecture

The standard Transformer architecture expects a 2D sequence of tokens as input, represented by a tensor of shape $(N, D)$, where $N$ is the sequence length and $D$ is the embedding dimension. Images, however, are 3D tensors of shape $(C, H, W)$. ViT bridges this structural difference through a specific sequence of operations.

### Patch Extraction and Flattening
An image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ is sliced into a sequence of non-overlapping patches $\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, where $(P, P)$ is the resolution of each image patch. The total number of patches $N$ (which serves as the effective sequence length) is calculated as:

$$N = \frac{H \cdot W}{P^2}$$

Each patch is flattened into a 1D vector of size $P^2 \cdot C$.

### Linear Projection and the CLS Token
The flattened patches are mapped into the Transformer's internal latent dimension $D$ using a trainable linear projection matrix $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$. 

Similar to BERT's classification token, a learnable embedding vector $\mathbf{x}_{\text{class}} \in \mathbb{R}^{1 \times D}$ is prepended to the sequence of projected patches. The final state of this token serves as the aggregate image representation used for downstream classification tasks.

### Position Embeddings
Because self-attention is permutation-invariant, the model has no inherent awareness of spatial geometry or patch order. To preserve spatial topology, trainable 1D standard position embeddings $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N + 1) \times D}$ are added element-wise to the patch embeddings:

$$\mathbf{z}_0 = \left[ \mathbf{x}_{\text{class}}; \, \mathbf{x}_p^1\mathbf{E}; \, \mathbf{x}_p^2\mathbf{E}; \, \dots; \, \mathbf{x}_p^N\mathbf{E} \right] + \mathbf{E}_{\text{pos}}$$

---

## 3. PyTorch Implementation from Scratch

The following implementation builds a functional Vision Transformer from the ground up, utilizing PyTorch's native modular layers. It includes patch tokenization via a 2D convolution shortcut, positional tracking, multi-head self-attention integration, and a classification head.

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Slices a 3D image into 2D patches, flattens them, and projects 
    them into a specified vector embedding dimension.
    """
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        
        # A 2D Convolution with kernel_size and stride equal to patch_size 
        # seamlessly handles slicing, flattening, and linear projection simultaneously.
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch_Size, Channels, Height, Width)
        x = self.projection(x) # Output shape: (Batch_Size, Embed_Dim, H_patches, W_patches)
        
        # Flatten spatial dimensions into a single sequence axis and transpose
        x = x.flatten(2) # Shape: (Batch_Size, Embed_Dim, Total_Patches)
        x = x.transpose(1, 2) # Shape: (Batch_Size, Total_Patches, Embed_Dim)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    A single standard Transformer Encoder block consisting of Multi-Head Attention
    preceded by Layer Normalization, followed by a feed-forward MLP network.
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.msa = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-Head Attention with residual connection (Pre-LN style)
        attn_out, _ = self.msa(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + attn_out
        
        # MLP Block with residual connection (Pre-LN style)
        x = x + self.mlp(self.ln2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Complete Vision Transformer (ViT) Architecture.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 1. Patch Embedding Setup
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        
        # 2. Learnable Tokens and Position Encodings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # 3. Stacked Transformer Encoder Blocks
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        # 4. Final Classification Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Extract patches and transform to embedding dimensions
        x = self.patch_embed(x) # Shape: (B, Num_Patches, Embed_Dim)
        
        # Prepend the learnable [CLS] token across the batch dimension
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # Shape: (B, 1, Embed_Dim)
        x = torch.cat((cls_tokens, x), dim=1) # Shape: (B, Num_Patches + 1, Embed_Dim)
        
        # Add spatial positional context coordinates and drop features
        x = self.pos_drop(x + self.pos_embed)
        
        # Pass sequentially through the encoder stack
        for layer in self.encoder_layers:
            x = layer(x)
            
        # Standard Normalization
        x = self.norm(x)
        
        # Extract features solely from the [CLS] token (Index 0)
        cls_token_final = x[:, 0] # Shape: (B, Embed_Dim)
        
        # Compute final classification output logits
        out = self.head(cls_token_final) # Shape: (B, Num_Classes)
        return out


# --- Functional Verification Block ---
if __name__ == "__main__":
    # Standard baseline configuration (ViT-Base/16 scale adjustment)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing validation code on device: {device}")
    
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        embed_dim=192,  # Compact dimension for execution speed
        depth=4,         # Reduced layers for quick compilation check
        num_heads=4,
        mlp_dim=768
    ).to(device)
    
    # Simulating a Batch of standard structural images: (Batch, Channels, Height, Width)
    dummy_images = torch.randn(4, 3, 224, 224).to(device)
    
    # Forward evaluation pass
    output_logits = model(dummy_images)
    
    print("\n--- ViT Compilation Analysis ---")
    print(f"Input batch shape:   {dummy_images.shape}")
    print(f"Output logits shape: {output_logits.shape} (Expected: [4, 10])")
    print("Vision Transformer verification executed successfully.")
```
