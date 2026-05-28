# Multimodal Architectures: CLIP and Contrastive Learning

## 1. Introduction to Multimodal AI
Traditional deep learning models are often restricted to a single modality—either processing text (NLP) or images (Computer Vision) in isolation. However, the real world is inherently multimodal. **Multimodal AI** bridges this gap by creating architectures capable of processing, understanding, and translating information across different types of data.

The core challenge in multimodal learning is **Semantic Alignment**: how do we ensure that a vector representing the text *"a fluffy golden retriever puppy"* sits close in vector space to the pixel matrix representing that exact image? 

Historically, this was handled by treating image-to-text as a classification problem (e.g., predicting fixed labels) or a captioning problem (predicting token sequences). Modern multimodal AI relies on **Joint Embedding Spaces**, where entirely different modalities are projected into a shared coordinate system.

---

## 2. Core Concepts of Contrastive Learning
Instead of predicting a specific class label, **Contrastive Learning** teaches a model to distinguish between matching and non-matching pairs. It is a self-supervised learning paradigm centered on the principle: *"Look at what is similar, and contrast it with what is different."*

### Positive vs. Negative Pairs
* **Positive Pair $(x, x^+)$**: Two representations that carry the same semantic meaning. In multimodal learning, this is a matching image and its corresponding caption (e.g., an image of a cat and the text "a photo of a cat").
* **Negative Pair $(x, x^-)$**: Two representations that are semantically distinct. For example, an image of a cat paired with the text "a driving car".

### The InfoNCE Loss Function
The most widely used objective function in contrastive learning is the **Information Noise-Contrastive Estimation (InfoNCE)** loss. It treats contrastive learning as a multi-class classification problem where the model attempts to identify the single positive sample out of a pool of negative samples.

Mathematically, for a query representation $q$, a single positive key $k^+$, and a set of negative keys $\{k^-\}$, the InfoNCE loss is defined as:

$$\mathcal{L}_{q, k^+} = -\log \frac{\exp(q \cdot k^+ / \tau)}{\exp(q \cdot k^+ / \tau) + \sum_{i} \exp(q \cdot k^-_i / \tau)}$$

Where:
* $q \cdot k$ represents a similarity metric, typically **Cosine Similarity**: $\frac{q \cdot k}{\|q\| \|k\|}$.
* $\tau$ (Tau) is a **temperature hyperparameter** that scales the logits, controlling how sharply the loss penalizes hard negative samples.

---
## 3. PyTorch Implementation from Scratch

Below is a modular implementation of a CLIP-style architecture. For simplicity and execution stability, we use a lightweight ResNet backbone for images and a standard PyTorch Transformer layer for text.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CLIP(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int = 4,
        embed_dim: int = 256,
        max_text_len: int = 77,
        init_temperature: float = 0.07
    ):
        super(CLIP, self).__init__()
        
        # 1. Image Encoder (Using a lightweight ResNet18 backbone)
        resnet = models.resnet18(pretrained=False)
        self.image_dim = resnet.fc.in_features
        # Remove the final classification head
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # 2. Text Encoder (Standard Transformer Encoder)
        self.text_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(max_text_len, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Projection Heads to Shared Embedding Space
        self.image_projection = nn.Linear(self.image_dim, embed_dim)
        self.text_projection = nn.Linear(self.text_dim, embed_dim)
        
        # 4. Learnable Temperature Hyperparameter (clamped for stability)
        self.temperature = nn.Parameter(torch.tensor(init_temperature))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        # Extract features: Output shape from resnet list is (B, image_dim, 1, 1)
        features = self.image_encoder(images)
        features = features.view(features.size(0), -1) # Flatten to (B, image_dim)
        
        # Project to shared space and normalize
        projected = self.image_projection(features)
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        # Add token and positional embeddings
        seq_len = text.size(1)
        embeddings = self.token_embedding(text) + self.position_embedding[:seq_len, :]
        
        # Pass through Transformer
        out = self.text_encoder(embeddings)
        
        # Use the hidden state of the first token (or pool across sequence length)
        # Often a specialized [CLS] token is used, here we pool across the sequence
        pooled_features = out.mean(dim=1)
        
        # Project to shared space and normalize
        projected = self.text_projection(pooled_features)
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized

    def forward(self, images: torch.Tensor, text: torch.Tensor):
        # Obtain normalized joint embeddings
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(text)
        
        # Scaling factor / Temperature
        t = torch.exp(self.temperature)
        
        # Compute pairwise Cosine Similarities: Shape (Batch_Size, Batch_Size)
        logits_per_image = torch.matmul(image_embeddings, text_embeddings.t()) * t
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text


# --- Symmetrized Cross-Entropy Loss Function ---
def contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    batch_size = logits_per_image.size(0)
    
    # Ground truth targets are simply the diagonal indices
    targets = torch.arange(batch_size, device=logits_per_image.device)
    
    # Cross-entropy calculations in both directions
    loss_images = F.cross_entropy(logits_per_image, targets)
    loss_texts = F.cross_entropy(logits_per_text, targets)
    
    # Symmetric average
    return (loss_images + loss_texts) / 2.0


# --- Verification Code Block ---
if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 8
    VOCAB_SIZE = 1000
    SEQ_LEN = 20
    
    # Instantiate Model
    model = CLIP(vocab_size=VOCAB_SIZE, max_text_len=SEQ_LEN)
    
    # Dummy Input Tensors: Images (B, C, H, W) and Tokenized Text (B, L)
    dummy_images = torch.randn(BATCH_SIZE, 3, 224, 224)
    dummy_text = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    
    # Forward Pass
    img_logits, txt_logits = model(dummy_images, dummy_text)
    
    # Calculate Loss
    loss = contrastive_loss(img_logits, txt_logits)
    
    print("--- CLIP Forward Output Shapes ---")
    print(f"Logits per Image Shape: {img_logits.shape} (Expected: [{BATCH_SIZE}, {BATCH_SIZE}])")
    print(f"Logits per Text Shape:  {txt_logits.shape} (Expected: [{BATCH_SIZE}, {BATCH_SIZE}])")
    print(f"Calculated Total Batch Loss: {loss.item():.4f}")
```

## 4. Zero-Shot Evaluation Pipeline
One of the most powerful properties of CLIP is its ability to perform classification on tasks it was never explicitly trained on (zero-shot). Instead of outputting logits for $N$ predefined discrete categories, you feed the names of the categories into the model as text prompts.
```
def zero_shot_classification(image_tensor, class_names, model, tokenizer_fn):
    """
    Performs zero-shot classification on a single image across multiple custom strings.
    """
    model.eval()
    with torch.no_grad():
        # 1. Engineering Prompts (e.g., "A photo of a dog")
        prompts = [f"a photo of a {cls}" for cls in class_names]
        tokenized_prompts = tokenizer_fn(prompts) # Expected shape: (Num_Classes, Seq_Len)
        
        # 2. Extract normalized embeddings
        image_embed = model.encode_image(image_tensor.unsqueeze(0)) # Shape: (1, Shared_Dim)
        text_embeds = model.encode_text(tokenized_prompts)          # Shape: (Num_Classes, Shared_Dim)
        
        # 3. Calculate similarity distributions
        logits = torch.matmul(image_embed, text_embeds.t()) * torch.exp(model.temperature)
        probabilities = F.softmax(logits, dim=-1)
        
    return probabilities.squeeze(0)
```
    
