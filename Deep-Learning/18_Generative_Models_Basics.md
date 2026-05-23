# 18. Generative Models Basics (GANs & VAEs)

While discriminative models learn the conditional probability distribution $P(y \mid \mathbf{x})$ to predict labels from data, generative models learn the underlying joint probability distribution $P(\mathbf{x})$ (or $P(\mathbf{x}, y)$) to generate entirely new data points that look like the training data.

---

## 1. Generative Adversarial Networks (GANs)

Introduced by Ian Goodfellow et al. (2014), GANs frame generative modeling as a minimax game between two competing neural networks:
1.  **The Generator ($G$):** Takes a random noise vector $\mathbf{z} \sim p_{\mathbf{z}}$ and maps it to the data space, attempting to create realistic samples that fool the Discriminator.
2.  **The Discriminator ($D$):** Takes a data sample (either real from the dataset or fake from the Generator) and outputs a single scalar probability $D(\mathbf{x}) \in [0, 1]$ indicating whether the sample is real.

### The Minimax Objective Function
The optimization landscape is defined by the value function $V(D, G)$:

$$\min_{G} \max_{D} V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}}[\log (1 - D(G(\mathbf{z})))]$$

Where:
* $\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})]$: The Discriminator's ability to correctly classify real images as real.
* $\mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}}[\log (1 - D(G(\mathbf{z})))]$: The Discriminator's ability to catch fake images, while the Generator minimizes this term to force $D(G(\mathbf{z})) \to 1$.

---

## 2. Variational Autoencoders (VAEs)

Introduced by Kingma & Welling (2013), VAEs are probabilistic models that force the latent space of a standard autoencoder to follow a known distribution (typically a standard Gaussian $\mathcal{N}(\mathbf{0}, \mathbf{I})$). This allows us to sample points randomly from the latent space to generate new data.

### The Reparameterization Trick
To train a VAE with backpropagation, we cannot sample directly from the latent distribution $\mathbf{z} \sim q_\phi(\mathbf{z} \mid \mathbf{x})$ because sampling is a non-differentiable operation. 

Instead, the network outputs the parameters of the distribution (mean $\boldsymbol{\mu}$ and variance $\boldsymbol{\sigma}^2$) and shifts the stochasticity to an auxiliary noise variable $\boldsymbol{\epsilon}$:

$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \text{where } \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

### The Loss Function: Evidence Lower Bound (ELBO)
The VAE objective minimizes reconstruction error while enforcing structural regularity on the latent distribution via Kullback-Leibler (KL) Divergence:

$$\mathcal{L}_{\text{VAE}}(\theta, \phi; \mathbf{x}) = \text{Reconstruction Loss} + D_{\text{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \parallel p(\mathbf{z}))$$

For a standard Gaussian prior, the analytical KL divergence for a single dimension reduces to:
$$D_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^{d} \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$

---

## 3. Structural Comparison

| Property | GAN (Generative Adversarial Network) | VAE (Variational Autoencoder) |
| :--- | :--- | :--- |
| **Training Style** | Adversarial game (Implicit density) | Probabilistic tracking (Explicit explicit lower bound) |
| **Output Quality** | Sharp, highly detailed, realistic images | Tends to be blurry or overly smooth |
| **Latent Space** | Often unconstrained or unstructured | Smooth, continuous, easily interpolatable |
| **Primary Issues** | Mode collapse, training instability | Blurrier samples due to pixel-wise reconstruction objectives |

---

## 4. Implementation in Python (PyTorch)

This script demonstrates a clean standalone pipeline for a standard **Variational Autoencoder (VAE)** including the full reparameterization loop and analytical ELBO computation.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoencoder(nn.Module):
    """
    A fully connected Variational Autoencoder (VAE) architecture 
    demonstrating the reparameterization trick.
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder Networks
        self.fc_encoder = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder Networks
        self.fc_decoder_hidden = nn.Linear(latent_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc_encoder(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """
        Applies the reparameterization trick: z = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # Sample auxiliary noise epsilon ~ N(0, I)
        return mu + std * eps

    def decode(self, z):
        h = F.relu(self.fc_decoder_hidden(z))
        return torch.sigmoid(self.fc_output(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    """
    Computes the total VAE loss = Reconstruction Loss + KL Divergence
    """
    # 1. Reconstruction Loss (Binary Cross Entropy for normalized pixels)
    recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # 2. Analytical KL Divergence closed form optimization targeting standard normal distribution
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_divergence

# Verification execution pass
if __name__ == "__main__":
    B, D_in = 4, 784  # Batch=4, Flattened 28x28 image dimension vector
    
    model = VariationalAutoencoder(input_dim=D_in, hidden_dim=256, latent_dim=16)
    mock_batch = torch.rand(B, D_in) # Simulating pixel values bounded [0, 1]
    
    reconstructed_batch, mu, logvar = model(mock_batch)
    loss = vae_loss_function(reconstructed_batch, mock_batch, mu, logvar)
    
    print(f"Input Matrix Shape:               {mock_batch.shape}")
    print(f"Reconstructed Output Shape:       {reconstructed_batch.shape}")
    print(f"Latent Structural Parameters Mu:  {mu.shape}")
    print(f"Computed Total VAE Objective Loss: {loss.item():.4f}")
```
