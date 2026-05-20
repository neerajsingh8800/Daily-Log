# 15. Word Embeddings

Word embeddings are dense, low-dimensional, continuous vector representations of text where words with similar meanings are mapped to nearby points in a geometric space. They replace sparse, high-dimensional representations like One-Hot Encoding.

---

## 1. The Core Philosophy
The foundation of modern word embeddings is the **Distributional Hypothesis** proposed by J.R. Firth (1957): 
> *"You shall know a word by the company it keeps."*

### Limitations of One-Hot Encoding:
* **Sparsity:** Vocabulary sizes of $10,000+$ create massive vectors filled mostly with zeros.
* **Orthogonality:** The dot product between any two one-hot vectors is always $0$, meaning the model cannot capture semantic relationships (e.g., "king" and "queen" are treated as completely independent concepts).

---

## 2. Word2Vec Architectures
Introduced by Mikolov et al. at Google (2013), Word2Vec trains a shallow, two-layer neural network to reconstruct linguistic contexts of words.

### A. Continuous Bag-of-Words (CBOW)
Predicts a target word ($w_t$) based on its surrounding context words within a window size $C$.
* **Objective:** Maximize the probability of the center word given context words.
* **Characteristics:** Faster to train, smoother representations for frequent words.

### B. Skip-gram
Predicts the surrounding context words given a single target center word ($w_t$).
* **Objective:** Maximize the probability of context words given the center word.
* **Characteristics:** Handles rare words much better because it creates more training pairs from the same sequence.

---

## 3. The Mathematics of Skip-gram with Negative Sampling (SGNS)

Computing the full Softmax layer over a massive vocabulary $V$ at the output layer is computationally expensive:
$$P(w_O \mid w_I) = \frac{\exp(v'_{w_O}{}^\top v_{w_I})}{\sum_{w=1}^{|V|} \exp(v'_{w}{}^\top v_{w_I})}$$

**Negative Sampling** converts this multi-class classification problem into a binary logistic regression problem. For every true context word (positive sample), we sample $K$ random words from the vocabulary that do not appear in the context (negative samples).

### Objective Function to Maximize:
$$\mathcal{L} = \log \sigma(v'_{w_O}{}^\top v_{w_I}) + \sum_{i=1}^{K} \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma(-v'_{w_i}{}^\top v_{w_I}) \right]$$

Where:
* $\sigma(x) = \frac{1}{1 + e^{-x}}$
* $v_{w_I}$: Target center word embedding vector.
* $v'_{w_O}$: True context word vector.
* $v'_{w_i}$: Negative sample word vector.
* $P_n(w)$: Unigram distribution raised to the $3/4$ power to penalize highly frequent words (like "the", "is").

---

## 4. Implementation in Python (PyTorch)

This script demonstrates how to construct a standard **Skip-gram model architecture with Negative Sampling** using PyTorch layers.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramNegSampling(nn.Module):
    """
    Skip-gram architecture optimized with a Negative Sampling optimization objective.
    """
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramNegSampling, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Center target embeddings
        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Context/Negative sample embeddings
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize weights matching Xavier variance bounds
        initrange = 1.0 / self.embedding_dim
        nn.init.uniform_(self.u_embeddings.weight, -initrange, initrange)
        nn.init.uniform_(self.v_embeddings.weight, -0, 0) # Zero initialization for context bias

    def forward(self, target, context, negative_samples):
        # Shapes: target [batch_size], context [batch_size], negative_samples [batch_size, K]
        
        # 1. Lookup embeddings
        emb_target = self.u_embeddings(target)           # Shape: [batch_size, embedding_dim]
        emb_context = self.v_embeddings(context)         # Shape: [batch_size, embedding_dim]
        emb_neg = self.v_embeddings(negative_samples)    # Shape: [batch_size, K, embedding_dim]
        
        # 2. Positive Sample score: Dot product between target and true context
        # bmm = batch matrix multiplication
        pos_score = torch.bmm(emb_context.unsqueeze(1), emb_target.unsqueeze(2)).squeeze()
        pos_loss = F.logsigmoid(pos_score)
        
        # 3. Negative Samples score: Dot product between target and random noise words
        neg_score = torch.bmm(emb_neg, emb_target.unsqueeze(2)).squeeze()
        neg_loss = torch.sum(F.logsigmoid(-neg_score), dim=1)
        
        # Total loss object is the negative average of both targets
        return -torch.mean(pos_loss + neg_loss)

# Verification execution pass
if __name__ == "__main__":
    V = 5000  # Vocabulary Size
    dim = 128 # Continuous space dimensions
    K = 5     # Number of negative samples per positive pair
    
    model = SkipGramNegSampling(vocab_size=V, embedding_dim=dim)
    
    # Simulating a mini-batch of size 4
    mock_target = torch.tensor([12, 45, 802, 144])
    mock_context = torch.tensor([13, 44, 801, 145])
    mock_negatives = torch.randint(0, V, (4, K))
```
    
    loss = model(mock_target, mock_context, mock_negatives)
    print(f"Calculated Negative Sampling Loss: {loss.item():.4f}")
