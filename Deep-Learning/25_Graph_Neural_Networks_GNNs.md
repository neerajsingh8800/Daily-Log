# Graph Neural Networks (GNNs)

## 1. Introduction: Non-Euclidean Data and the Limitation of CNNs
Most standard deep learning architectures are designed for **Euclidean data**—data structuralized in flat, regular grids. For example:
* **Images:** 2D grids of pixels with fixed spatial neighbors.
* **Text:** 1D linear sequences of tokens ordered in time.

However, many real-world domains exist as **Non-Euclidean data**, structured as arbitrary graphs with irregular shapes, varying node degrees, and no global orientation. Examples include social networks, molecular structures, citation webs, and 3D point clouds.
Applying standard CNNs to graphs fails because:
1. **No Fixed Locality:** A node might have 2 neighbors or 2,000 neighbors; standard sliding convolutional filters require a fixed matrix size.
2. **Permutation Invariance:** The structural arrangement of a graph remains identical regardless of how nodes are indexed or ordered in an adjacency matrix. CNNs are highly sensitive to row/column ordering.

**Graph Neural Networks (GNNs)** resolve this by performing operations directly on structural topologies, extracting features based on connectivity independent of grid structures.

---

## 2. Graph Foundations & Mathematical Formulations
A graph is mathematically represented as $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where:
* $\mathcal{V} = \{v_1, v_2, \dots, v_N\}$ is the set of $N$ **Nodes** (Vertices).
* $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$ is the set of $M$ **Edges** representing connectivity between nodes.

### Matrix Representations
* **Node Feature Matrix ($\mathbf{H} \in \mathbb{R}^{N \times F}$):** Contains the input features for each node, where $F$ is the feature dimension.
* **Adjacency Matrix ($\mathbf{A} \in \mathbb{R}^{N \times N}$):** A binary or weighted matrix where $\mathbf{A}_{ij} = 1$ if an edge exists between node $i$ and node $j$, and $0$ otherwise.
* **Degree Matrix ($\mathbf{D} \in \mathbb{R}^{N \times N}$):** A diagonal matrix representing the number of connected edges for each node: $\mathbf{D}_{ii} = \sum_{j} \mathbf{A}_{ij}$.

---

## 3. The Message Passing Framework
Modern GNNs operate via a neighborhood iterative pipeline known as **Message Passing**. In each layer $l$, every individual node collects structural vector states from its immediate neighbors to compute its next-layer representation. This process consists of three sequential steps:

1. **Message:** Calculate a message vector from a neighbor node $j$ to target node $i$:
   $$\mathbf{m}_{ij}^{(l)} = \text{Message}^{(l)}\left(\mathbf{h}_i^{(l)}, \mathbf{h}_j^{(l)}, \mathbf{e}_{ij}\right)$$

2. **Aggregation:** Permutation-invariant pooling (e.g., Sum, Mean, Max) summarizing all incoming structural neighbor messages:
   $$\mathbf{a}_i^{(l)} = \text{Aggregate}^{(l)}\left(\left\{\mathbf{m}_{ij}^{(l)} \mid j \in \mathcal{N}(i)\right\}\right)$$

3. **Update:** Combine the aggregated neighborhood context with the target node's current representation to generate the updated state:
   $$\mathbf{h}_i^{(l+1)} = \text{Update}^{(l)}\left(\mathbf{h}_i^{(l)}, \mathbf{a}_i^{(l)}\right)$$

### Graph Convolutional Networks (GCN) Formulation
Introduced by Kipf and Welling (2016), a **Graph Convolutional Network (GCN)** instantiates message passing efficiently using localized first-order approximations. To prevent a node's own features from being ignored during neighborhood aggregation, self-loops are explicitly added to the adjacency matrix: $\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}_N$.

To prevent nodes with very high degrees from completely scaling out and destabilizing feature embeddings, the representation is symmetrically normalized using the modified degree matrix $\mathbf{\hat{D}}$:

$$\mathbf{H}^{(l+1)} = \sigma \left( \mathbf{\hat{D}}^{-\frac{1}{2}} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-\frac{1}{2}} \mathbf{H}^{(l)} \mathbf{W}^{(l)} \right)$$

Where:
* $\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}_N$ (Adjacency matrix with self-loops).
* $\mathbf{\hat{D}}_{ii} = \sum_{j} \mathbf{\hat{A}}_{ij}$ (Diagonal degree matrix of $\mathbf{\hat{A}}$).
* $\mathbf{W}^{(l)}$ (Trainable weight projection matrix for layer $l$).
* $\sigma$ (Non-linear activation function, e.g., ReLU).

---

## 4. PyTorch Implementation from Scratch
This implementation builds a structural GCN Layer using pure PyTorch tensors and matrix math, bypassing external libraries like PyTorch Geometric (PyG). This makes it easy to inspect the underlying linear operations.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    A custom Graph Convolutional Network (GCN) Layer executing 
    symmetrically normalized neighborhood aggregation from scratch.
    """
    def __init__(self, in_features: int, out_features: int):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Trainable Weight Transformation Matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # Trainable Bias Parameter
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Standard Xavier/Glorot initialization initialization for stability
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x:   Node Feature Matrix of shape (N, In_Features)
        adj: Binary Adjacency Matrix of shape (N, N)
        """
        N = x.size(0)
        
        # 1. Add Self-Loops (A_hat = A + I)
        I = torch.eye(N, device=adj.device)
        adj_hat = adj + I
        
        # 2. Compute Degree Matrix (D_hat)
        degree = torch.sum(adj_hat, dim=1)
        
        # 3. Compute Inverse Square Root of Degree Matrix (D_hat^-0.5)
        # Avoid division by zero by clamping minimum degree value
        deg_inv_sqrt = torch.pow(degree, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        # 4. Compute Symmetrically Normalized Adjacency (D^-0.5 * A_hat * D^-0.5)
        norm_adj = torch.mm(torch.mm(D_inv_sqrt, adj_hat), D_inv_sqrt)
        
        # 5. Linear Feature Projection (X * W)
        support = torch.mm(x, self.weight)
        
        # 6. Neighborhood Aggregation via Matrix Multiplication
        output = torch.mm(norm_adj, support)
        
        return output + self.bias


class GCNNetwork(nn.Module):
    """
    A 2-Layer GCN Network tailored for node classification tasks.
    """
    def __init__(self, num_features: int, num_hidden: int, num_classes: int, dropout: float = 0.5):
        super(GCNNetwork, self).__init__()
        self.gcn1 = GraphConvolution(num_features, num_hidden)
        self.gcn2 = GraphConvolution(num_hidden, num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Layer 1 followed by ReLU activation and Dropout
        x = F.relu(self.gcn1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2 (Output layer yielding classification logits)
        x = self.gcn2(x, adj)
        return x


# --- Operational Verification Block ---
if __name__ == "__main__":
    # Simulate a small synthetic graph dataset
    NUM_NODES = 6
    FEATURE_DIM = 4
    HIDDEN_DIM = 8
    NUM_CLASSES = 3

    # 1. Create Mock Node Features (N x F)
    # 6 nodes, each containing a 4-dimensional feature vector
    dummy_features = torch.randn(NUM_NODES, FEATURE_DIM)

    # 2. Create a Mock Symmetric Adjacency Matrix (N x N)
    dummy_adjacency = torch.tensor([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ], dtype=torch.float32)

    # Instantiate the 2-layer model topology
    model = GCNNetwork(num_features=FEATURE_DIM, num_hidden=HIDDEN_DIM, num_classes=NUM_CLASSES)
    model.eval()

    with torch.no_grad():
        output_logits = model(dummy_features, dummy_adjacency)

    print("--- Graph Convolutional Network Compilation Verification ---")
    print(f"Node Features Matrix Shape:      {dummy_features.shape}")
    print(f"Graph Adjacency Matrix Shape:    {dummy_adjacency.shape}")
    print(f"Model Class Logits Output Shape: {output_logits.shape} (Expected: [{NUM_NODES}, {NUM_CLASSES}])")
    print("\nGCN Layer matrix propagation completed successfully.")
```
