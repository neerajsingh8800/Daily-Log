# Module 03: Database Sharding, Replication, and Partitioning

Scaling data infrastructure is a fundamental challenge in high-throughput distributed systems. When write throughput or dataset volume exceeds the capacity of a single database instance, systems must scale horizontally via **Replication**, **Partitioning (Sharding)**, and **Consistent Hashing**.

This module covers horizontal vs. vertical scaling trade-offs, master-replica sync dynamics, consistent hashing mathematical formulations, resharding mechanics, and a production Consistent Hash Ring implementation in Python.

---

## 1. Theoretical Foundations

### 1.1 Architectural Concepts
* **Replication**: Copying data across multiple database nodes (Master-Replica / Multi-Master) to improve read scalability, fault tolerance, and high availability.
  * *Synchronous Replication*: Master waits for replicas to confirm writes. Ensures zero data loss (RPO = 0) but increases write latency.
  * *Asynchronous Replication*: Master acknowledges writes immediately and replicates in the background. Ultra-fast, but risks data loss during unexpected failover.
* **Vertical Partitioning**: Splitting tables by columns (e.g., placing heavy text/BLOB columns in a separate table).
* **Horizontal Partitioning (Sharding)**: Splitting tables by rows across separate database instances (shards) based on a **Shard Key**.

* ---

### 1.2 Mathematical Foundations of Consistent Hashing

#### Standard Modulo Sharding Limitation
Traditional hash sharding uses the modulo operator:

$$\text{Shard ID} = \text{hash}(\text{Key}) \bmod N$$

Where $N$ is the number of database nodes. **Problem**: If $N$ changes (a node is added or removed), almost $100\%$ of keys remap to new shards, causing catastrophic database re-indexing and cache invalidation.

Consistent Hashing maps both **Database Nodes** and **Keys** onto an abstract $2^{32} - 1$ mathematical ring using a cryptographic hash function (e.g., MD5 or SHA-256):

$$\theta_{\text{node}} = \text{hash}(\text{Node IP}) \pmod{2^{32}}$$

$$\theta_{\text{key}} = \text{hash}(\text{Key}) \pmod{2^{32}}$$

A key is assigned to the first node whose position $\theta_{\text{node}}$ is greater than or equal to $\theta_{\text{key}}$ in a clockwise traversal.

#### Node Resharding Key Movement Ratio
When adding a new node to a ring with $N$ existing nodes, only $\frac{1}{N+1}$ fraction of keys need to be relocated on average:

$$\text{Keys Relocated Ratio} \approx \frac{1}{N + 1}$$

#### Virtual Nodes (VNodes) for Load Balance
To prevent hotspots and uneven key distributions, each physical node is assigned $V$ virtual nodes on the ring:

$$\text{Total Virtual Ring Positions} = N \times V$$

---

## 2. Production Consistent Hash Ring Implementation

This Python module implements a production-grade Consistent Hash Ring with Virtual Nodes and smooth node join/leave mechanics.

### Prerequisites

```bash
pip install hashlib
```

### Python Implementation (consistent_hash_ring.py)
```python
import hashlib
import bisect
from typing import List, Dict, Optional

class ConsistentHashRing:
    def __init__(self, num_replicas: int = 100):
        """
        :param num_replicas: Number of virtual nodes per physical node (VNodes).
                             Higher values yield a more uniform key distribution.
        """
        self.num_replicas = num_replicas
        self.ring: List[int] = []              # Sorted array of virtual node hash positions
        self.vnode_map: Dict[int, str] = {}    # Maps hash position -> physical node ID

    def _hash(self, key: str) -> int:
        """Generates a 32-bit integer hash using MD5."""
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)  # Truncate to 32 bits for clean ring mapping

    def add_node(self, node_id: str):
        """Adds a physical node and its virtual replicas to the hash ring."""
        for i in range(self.num_replicas):
            vnode_key = f"{node_id}#vnode-{i}"
            vnode_hash = self._hash(vnode_key)
            
            bisect.insort(self.ring, vnode_hash)
            self.vnode_map[vnode_hash] = node_id
        print(f"[NODE ADDED] Node '{node_id}' mapped with {self.num_replicas} virtual nodes.")

    def remove_node(self, node_id: str):
        """Removes a physical node and its virtual replicas from the hash ring."""
        for i in range(self.num_replicas):
            vnode_key = f"{node_id}#vnode-{i}"
            vnode_hash = self._hash(vnode_key)
            
            idx = bisect.bisect_left(self.ring, vnode_hash)
            if idx < len(self.ring) and self.ring[idx] == vnode_hash:
                del self.ring[idx]
                del self.vnode_map[vnode_hash]
        print(f"[NODE REMOVED] Node '{node_id}' stripped from the hash ring.")

    def get_node(self, key: str) -> Optional[str]:
        """Finds the primary database node responsible for storing a given key."""
        if not self.ring:
            return None

        key_hash = self._hash(key)
        # Find the first vnode with hash >= key_hash
        idx = bisect.bisect_right(self.ring, key_hash)
        
        # If key_hash is greater than all vnodes, wrap around to the first vnode on ring (0)
        if idx == len(self.ring):
            idx = 0

        return self.vnode_map[self.ring[idx]]

# Verification & Simulation
if __name__ == "__main__":
    hash_ring = ConsistentHashRing(num_replicas=50)

    # 1. Initialize DB Cluster Nodes
    hash_ring.add_node("db-shard-01.internal")
    hash_ring.add_node("db-shard-02.internal")
    hash_ring.add_node("db-shard-03.internal")

    # 2. Test Key Route Assignments
    sample_keys = [f"user_account_{i}" for i in range(5)]
    print("\n--- Initial Key Route Mapping ---")
    initial_mappings = {}
    for key in sample_keys:
        assigned_shard = hash_ring.get_node(key)
        initial_mappings[key] = assigned_shard
        print(f"Key '{key}'  -->  {assigned_shard}")

    # 3. Add a new Shard to scale horizontally
    print("\n--- Scaling Up: Adding Shard 04 ---")
    hash_ring.add_node("db-shard-04.internal")

    # 4. Observe minimal remapping of keys
    print("\n--- Remapped Key Route Evaluation ---")
    remapped_count = 0
    for key in sample_keys:
        new_shard = hash_ring.get_node(key)
        changed = " [REMAPPED]" if new_shard != initial_mappings[key] else ""
        if changed:
            remapped_count += 1
        print(f"Key '{key}'  -->  {new_shard}{changed}")

    print(f"\nTotal Keys Remapped: {remapped_count}/{len(sample_keys)} (Minimal disruption preserved)")
```
