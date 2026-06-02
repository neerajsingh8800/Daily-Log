# 28. Distributed Training Paradigms (Data, Pipeline, and Tensor Parallelism)

When training modern state-of-the-art deep learning architectures (such as LLMs with billions of parameters), a single GPU quickly encounters hard limits in compute capabilities and memory capacity. To overcome this, training must be scaled across multi-GPU clusters using distributed paradigms. This requires partitioning the model weights, optimizer states, gradients, or batch dimensions across distinct computing nodes.

---

## 1. Core Distributed Paradigms
### A. Distributed Data Parallelism (DDP)
In DDP, the entire model architecture and its optimizer states are fully replicated onto every participating GPU. 
* **Mechanism:** The global training batch size $B$ is split evenly into local micro-batches across $N$ workers. Each worker performs an independent forward and backward pass.
* **Synchronization:** Before optimizer parameter updates occur, workers synchronize their local gradients across the network using a ring-topology collective communication mechanism called **All-Reduce**, computing the mean gradient:

$$\bar{\mathbf{g}} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{g}_i$$

* **Limitation:** DDP fails if the model weights, gradients, and optimizer states combined exceed the VRAM boundaries of a single GPU node.

### B. Pipeline Parallelism (PP)
Pipeline Parallelism partitions the model vertically by grouping sequences of entire layers together and assigning each cluster block to a distinct GPU device.
* **Mechanism:** Activations (hidden states) flow sequentially forward from GPU $i$ to GPU $i+1$, while gradients propagate backward from GPU $i+1$ to GPU $i$.
* **The "Bubble" Problem:** If a batch is processed naively, higher-indexed GPUs sit completely idle waiting for activations from earlier stages. Advanced configurations (like the *1F1B — One Forward, One Backward* paradigm) split the batch into smaller micro-batches to interleave forward and backward passes, keeping the processors active.

### C. Tensor Parallelism (TP)
Tensor Parallelism splits a single layer's weight matrix *intra-layer* across multiple devices (typically utilizing Megatron-LM styles). This is critical for incredibly wide projection structures in multi-head self-attention mechanisms.

For a linear transformation layer $Y = XW$, we partition $W$ along either columns or rows:
* **Column-Parallel Linear Layer:** Splits $W$ vertically into $W = [W_1 \mid W_2]$. 
  * Forward calculation: $Y_1 = XW_1$ and $Y_2 = XW_2$.
  * Device outputs are concatenated together at the end of the pass.
* **Row-Parallel Linear Layer:** Splits $W$ horizontally into top and bottom halves.
  * Requires an **All-Reduce (Sum)** communication step immediately following matrix multiplication to aggregate partial hidden state evaluations across matching nodes.

---

## 2. Distributed Communication Complexities

Distributed training efficiency depends heavily on minimizing communication overhead across high-bandwidth networks (like NVLink or InfiniBand).

| Communication Primitive | Operational Target | Network Communication Volume Complexity |
| :--- | :--- | :--- |
| **All-Reduce** | Sums/averages arrays across all workers and returns results to all workers. | $2 \times \frac{N-1}{N} \times |W|$ |
| **All-Gather** | Collects data blocks from all workers and concatenates them onto all workers. | $\frac{N-1}{N} \times |W|$ |
| **Reduce-Scatter** | Sums arrays across all workers and splits the resulting values across them. | $\frac{N-1}{N} \times |W|$ |

Where $N$ represents the number of active GPU nodes and $|W|$ defines the parameter buffer data size footprint being synchronized.

---

## 3. Implementation in Python (PyTorch Distributed)

This standalone blueprint demonstrates the fundamental structure for setting up a multi-GPU **Distributed Data Parallel (DDP)** environment using PyTorch's native `torch.distributed` communication engine.

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        return self.net(x)

class RandomDataset(Dataset):
    def __len__(self):
        return 128
        
    def __getitem__(self, idx):
        return torch.randn(32), torch.randint(0, 10, (1,)).squeeze()

def setup_distributed_environment(rank: int, world_size: int):
    """
    Initializes the process group mapping communication links across GPU nodes.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group targeting NCCL (standard backend for NVIDIA hardware)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed_environment():
    dist.destroy_process_group()

def distributed_training_loop(rank: int, world_size: int):
    """
    Core execution logic ran concurrently across each designated worker GPU thread.
    """
    print(f"Initializing Worker Process Node on GPU Rank: {rank}/{world_size}")
    setup_distributed_environment(rank, world_size)
    
    # 1. Instantiate network and port explicitly to the allocated device ID
    model = ToyModel().to(rank)
    
    # Wrap model with DistributedDataParallel to hook automated gradient All-Reduce calls
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 2. Configure dataset and DistributedSampler to ensure non-overlapping batch allocation
    dataset = RandomDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)
    
    # Training pass execution
    model.train()
    for epoch in range(2):
        # Set epoch seed configuration on sampler to maintain clean shuffling synchronization
        sampler.set_epoch(epoch)
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(rank), targets.to(rank)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradients are automatically aggregated via an internal Ring All-Reduce step here
            optimizer.step()
            
            if rank == 0 and batch_idx % 4 == 0:
                print(f"Epoch: {epoch} | Batch Item Index: {batch_idx}/{len(dataloader)} | Worker Zero Training Loss Reference: {loss.item():.4f}")
                
    cleanup_distributed_environment()
    print(f"Process Node on GPU Rank {rank} execution pipeline successfully finished.")

if __name__ == "__main__":
    # Mocking environment simulation check
    # In practice, this script is launched via command line processing utilities:
    # torchrun --nproc_per_node=NUM_GPUS 28_Distributed_Training_Paradigms.md
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"System Check -> Available Hardware Compute Accelerator GPUs Detected: {gpu_count}")
    
    if gpu_count >= 2:
        # Multiprocessing context spawns distributed tasks across available targets
        torch.multiprocessing.spawn(distributed_training_loop, args=(gpu_count,), nprocs=gpu_count, join=True)
    else:
        print("Distributed execution checks require a multi-GPU multi-accelerator physical hardware environment loop configuration.")
```
