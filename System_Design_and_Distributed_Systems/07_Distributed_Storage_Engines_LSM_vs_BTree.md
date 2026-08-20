# Module 07: Distributed Storage Engines: LSM-Trees vs. B-Trees

Storage engines form the low-level foundation of state persistence in databases. Choosing between write-optimized engines (**Log-Structured Merge-Trees / LSM-Trees**) and read-optimized engines (**B-Trees / $B^+$-Trees**) dictates system performance under write-heavy versus read-heavy workloads.

This module covers the underlying disk structures, Write-Ahead Logs (WAL), SSTables, Bloom Filter acceleration, mathematical formulations for Write Amplification Factor (WAF) and Space Amplification Factor (SAF), and a complete in-memory LSM-Tree storage engine implementation in Python.

---

## 1. Theoretical Foundations

### 1.1 Structural Mechanics & Trade-offs

* **B-Trees / $B^+$-Trees (In-Place Update Engine)**:
  * Organizes data into fixed-size pages (typically 4KB–16KB) arranged in a balanced search tree on disk.
  * Modifies data **in-place**, requiring random I/O operations for non-contiguous key writes.
  * Features small, bounded read latency ($O(\log_B N)$) but suffers from random write penalties and fragmentation. Used in relational systems like PostgreSQL (B-Tree) and MySQL InnoDB ($B^+$-Tree).

* **LSM-Trees (Append-Only Out-of-Place Engine)**:
  * Converts random writes into sequential disk I/O by appending incoming mutative operations to an in-memory buffer (**MemTable**) backed by an append-only **Write-Ahead Log (WAL)**.
  * When the MemTable exceeds a threshold, it is flushed to disk as an immutable, sorted file (**SSTable / Sorted String Table**).
  * Background **Compaction** processes periodically merge overlapping SSTables to reclaim space and purge deleted/superseded keys. Used in NoSQL systems like RocksDB, Apache Cassandra, and Google Bigtable.
 
  * ---

### 1.2 Mathematical Foundations: Write Amplification & Amplification Factors

#### 1. Write Amplification Factor (WAF)
Write Amplification Factor measures the ratio of bytes written to underlying physical storage relative to bytes submitted by application writes:

$$\text{WAF} = \frac{\text{Bytes Written to Disk (WAL + Flushes + Compactions)}}{\text{Bytes Written by Application}}$$

* High WAF accelerates SSD wear and consumes disk I/O bandwidth.
* **LSM-Trees** trade background Compaction WAF ($\text{WAF} \approx 10 - 30$) to achieve ultra-fast sequential write speeds.
* **B-Trees** suffer high page-level WAF ($\text{WAF} \approx 20 - 100$) due to modifying entire 4KB/16KB pages even when updating a single 50-byte record.

#### 2. LSM-Tree Level Size Ratio & Read Complexity
For a Leveled Compaction LSM-Tree with growth factor $T$ (typically $T \approx 10$) and $L$ levels, the total size of level $i$ is:

$$\text{Size}(L_i) = \text{Size}(L_0) \cdot T^i$$

Point lookups search through $L_0$ files plus 1 SSTable per subsequent level $L_1 \dots L_k$. The worst-case disk seek count without Bloom Filters is:

$$\text{Max Read Seeks} = \vert{}N_{L_0}\vert{} + (L - 1)$$

---

## 2. Storage Engine Architecture Comparison

| Metric / Dimension | B-Tree / $B^+$-Tree Engine | LSM-Tree Engine |
| :--- | :--- | :--- |
| **Write Pattern** | In-place random updates | Out-of-place sequential appends |
| **Primary Bottleneck** | Random I/O latency & page locks | Background Compaction I/O & WAF |
| **Read Performance** | Excellent ($O(\log N)$ single page fetch) | Requires Bloom Filter / SSTable binary search |
| **Space Amplification (SAF)**| Low (fixed page layout) | Moderate-High (duplicate keys across levels) |
| **Hardware Suitability** | HDDs / Battery-backed RAM caches | Modern NVMe SSDs (High sequential write efficiency) |

---

## 3. Production LSM-Tree Storage Engine Implementation

This Python module implements an **LSM-Tree Storage Engine** complete with an in-memory **MemTable**, disk **WAL (Write-Ahead Log)**, **SSTables (Sorted String Tables)**, and **Bloom Filters**.

### Prerequisites

```bash
pip install hashlib
```
### Python Implementation (lsm_storage_engine.py)
```python
import os
import json
import hashlib
from typing import Dict, Optional, List, Tuple

# -------------------------------------------------------------------
# 1. BLOOM FILTER FOR FAST SSTABLE MISS DETECTION
# -------------------------------------------------------------------
class SimpleBloomFilter:
    def __init__(self, size: int = 1000):
        self.size = size
        self.bit_array = [0] * size

    def _hashes(self, key: str) -> List[int]:
        h1 = int(hashlib.md5(key.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        return [h1 % self.size, h2 % self.size]

    def add(self, key: str):
        for idx in self._hashes(key):
            self.bit_array[idx] = 1

    def contains(self, key: str) -> bool:
        return all(self.bit_array[idx] == 1 for idx in self._hashes(key))


# -------------------------------------------------------------------
# 2. SSTABLE COMPONENT (IMMUTABLE DISK SEGMENT)
# -------------------------------------------------------------------
class SSTable:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.bloom_filter = SimpleBloomFilter()
        self.index: Dict[str, int] = {}  # Sparse Index: Key -> Byte Offset in file
        self._build_index()

    def _build_index(self):
        if not os.path.exists(self.filepath):
            return
        with open(self.filepath, 'r') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                record = json.loads(line)
                key = record["key"]
                self.index[key] = offset
                self.bloom_filter.add(key)

    def get(self, key: str) -> Optional[str]:
        # Step 1: Bloom Filter check
        if not self.bloom_filter.contains(key):
            return None  # Definitely not present in this SSTable

        # Step 2: Index lookup and byte seek
        if key in self.index:
            with open(self.filepath, 'r') as f:
                f.seek(self.index[key])
                record = json.loads(f.readline())
                return record["val"]
        return None


# -------------------------------------------------------------------
# 3. LSM-TREE ENGINE WITH MEMTABLE AND FLUSH
# -------------------------------------------------------------------
class LSMTreeEngine:
    def __init__(self, data_dir: str = "./lsm_data", memtable_threshold: int = 3):
        self.data_dir = data_dir
        self.memtable_threshold = memtable_threshold
        self.memtable: Dict[str, str] = {}
        self.sstables: List[SSTable] = []
        self.wal_path = os.path.join(self.data_dir, "wal.log")
        self.sstable_counter = 0

        os.makedirs(self.data_dir, exist_ok=True)
        self._recover_from_wal()

    def _recover_from_wal(self):
        """Reconstructs MemTable state on restart using WAL log."""
        if os.path.exists(self.wal_path):
            with open(self.wal_path, 'r') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        self.memtable[record["key"]] = record["val"]
            print(f"[RECOVERY] Restored {len(self.memtable)} keys from WAL.")

    def put(self, key: str, val: str):
        """Appends to WAL and inserts into MemTable."""
        # Step 1: Write-Ahead Log
        with open(self.wal_path, 'a') as f:
            f.write(json.dumps({"key": key, "val": val}) + "\n")

        # Step 2: MemTable Insert
        self.memtable[key] = val
        print(f"[WRITE] Key='{key}' inserted into MemTable.")

        # Step 3: Check Flush Threshold
        if len(self.memtable) >= self.memtable_threshold:
            self._flush_memtable()

    def get(self, key: str) -> Optional[str]:
        """Reads key across MemTable -> SSTables (newest to oldest)."""
        # Search Step 1: MemTable check
        if key in self.memtable:
            val = self.memtable[key]
            if val == "__TOMBSTONE__":
                return None  # Key marked deleted
            print(f"[READ HIT] Found in MemTable: {key}")
            return val

        # Search Step 2: SSTable Search (Reverse order for recent SSTables)
        for sstable in reversed(self.sstables):
            val = sstable.get(key)
            if val is not None:
                if val == "__TOMBSTONE__":
                    return None
                print(f"[READ HIT] Found in SSTable '{sstable.filepath}': {key}")
                return val

        print(f"[READ MISS] Key '{key}' not found.")
        return None

    def delete(self, key: str):
        """Executes soft deletion via Tombstone marker insertion."""
        self.put(key, "__TOMBSTONE__")

    def _flush_memtable(self):
        """Flushes MemTable to a new SSTable on disk."""
        self.sstable_counter += 1
        sstable_filename = os.path.join(self.data_dir, f"sstable_{self.sstable_counter}.db")
        
        # Write sorted keys to disk
        sorted_keys = sorted(self.memtable.keys())
        with open(sstable_filename, 'w') as f:
            for k in sorted_keys:
                f.write(json.dumps({"key": k, "val": self.memtable[k]}) + "\n")

        print(f"[FLUSH] Created SSTable on disk: {sstable_filename}")
        
        # Load new SSTable into active list
        self.sstables.append(SSTable(sstable_filename))
        
        # Clear MemTable and purge WAL
        self.memtable.clear()
        open(self.wal_path, 'w').close()


# -------------------------------------------------------------------
# EXECUTION / SIMULATION RUNNER
# -------------------------------------------------------------------
if __name__ == "__main__":
    import shutil
    shutil.rmtree("./lsm_data", ignore_errors=True)

    engine = LSMTreeEngine(memtable_threshold=3)

    # 1. Trigger MemTable Flush
    engine.put("user_1", "Alice")
    engine.put("user_2", "Bob")
    engine.put("user_3", "Charlie")  # Triggers Flush #1 to SSTable_1

    # 2. Add more keys for Flush #2
    engine.put("user_4", "David")
    engine.put("user_1", "Alice Updated")  # Update existing key
    engine.put("user_5", "Eve")      # Triggers Flush #2 to SSTable_2

    # 3. Read Operations
    print("\n--- Read Queries ---")
    print("Result user_1:", engine.get("user_1"))  # Reads updated value from SSTable_2
    print("Result user_2:", engine.get("user_2"))  # Reads value from SSTable_1
    print("Result user_99:", engine.get("user_99"))  # Miss across all components
```

## 4. Architectural Guidelines & Best Practices

* Bloom Filter Precision: Tune Bloom Filter false-positive probability ($p \approx 0.01$) to eliminate unnecessary disk seeks during cache misses.
* Tombstone Garbage Collection: Ensure compactions purge Tombstone records (__TOMBSTONE__) once deleted keys cascade past the oldest active level ($L_{\text{max}}$).
* Direct I/O (O_DIRECT): B-Tree storage engines bypass operating system page caches using direct I/O to prevent double-buffering performance penalties.
