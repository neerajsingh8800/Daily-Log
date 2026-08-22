# Module 09: Distributed ID Generators and Unique Keys

Generating unique, ordered, and collision-free primary identifiers at scale is a fundamental requirement of distributed databases, event streams, and microservices. Monolithic auto-incrementing database sequences create single-point bottlenecks and cross-datacenter contention when horizontal sharding is introduced.

This module covers unique ID generation strategies (UUIDv4, Twitter Snowflake, Ticket Servers, ULID), bit layout math, clock drift handling strategies, probability formulas for UUID collision risk (Birthday Paradox), and a complete multi-threaded Twitter Snowflake ID generator implementation in Python.

---

## 1. Theoretical Foundations

### 1.1 Generation Paradigms & Trade-offs

* **UUIDv4 (128-bit Random Identification)**:
  * Uses 122 bits of pseudo-randomness.
  * Completely decentralized with zero network coordination overhead.
  * **Trade-off**: Non-monotonic random distribution causes severe $B^+$-Tree index fragmentation and cache thrashing in databases (e.g., MySQL InnoDB).

* **Twitter Snowflake (64-bit Time-Ordered Identifiers)**:
  * Generates 64-bit integer IDs structured into Bit-Fields: `Epoch Timestamp` + `Node/Worker ID` + `Sequence Counter`.
  * Guarantees rough time sorting (**k-sortable**) while retaining compact integer storage footprint ($8\text{ bytes}$ vs $36\text{ bytes}$ for UUID strings).
  * Requires clock synchronization protocols (NTP) to prevent ID collision or regression across worker nodes.

* **Ticket Servers (Centralized Auto-Increment Batching)**:
  * Uses centralized database instances (e.g., MySQL `REPLACE INTO` with `auto_increment_increment`) to grant numeric ID ranges to application workers in memory chunks.
  * Ensures simple 64-bit integers but creates network dependency on ticket coordinator instances.
 
---

### 1.2 Mathematical Foundations

#### 1. Birthday Paradox & Collision Probability (UUIDv4 / Random IDs)
The probability $P(n)$ that at least two generated $d$-bit random IDs collide among $n$ generated items is approximated using the exponential expansion:

$$P(n) \approx 1 - e^{-\frac{n^2}{2 \cdot 2^d}} = 1 - e^{-\frac{n^2}{2^{d+1}}}$$

*For 128-bit UUIDv4 ($d = 122$ effective random bits)*: To reach a collision probability $P(n) \approx 50\%$, the required number of generated keys $n$ is:

$$n \approx 2^{d/2} \cdot \sqrt{2 \ln(2)} \approx 2^{61} \approx 2.3 \times 10^{18} \text{ keys}$$

#### 2. Twitter Snowflake Throughput Limits
With a 12-bit sequence field, a single Snowflake worker node generates up to:

$$\text{Max Throughput per Node} = 2^{12} \text{ IDs/millisecond} = 4,096 \text{ IDs/ms} = 4,096,000 \text{ IDs/sec}$$

With 10 bits allocated for Node IDs ($2^{10} = 1,024$ unique nodes), maximum cluster-wide throughput equals:

$$\text{Max Cluster Throughput} = 1,024 \text{ nodes} \times 4,096,000 \text{ IDs/sec} \approx 4.19 \times 10^9 \text{ IDs/sec}$$

---

## 2. Distributed ID Architectures Comparison

| Strategy | Size | Sorting Guarantee | Network Call Required | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **UUIDv4** | 128-bit (16 Bytes) | Unsorted (Random) | No | Distributed tracing, stateless keys |
| **Snowflake** | 64-bit (8 Bytes) | Rough Time-Ordered (k-sortable) | No | Database Primary Keys, Event Streams |
| **ULID** | 128-bit (16 Bytes) | Strictly Monotonic in ms | No | Log indexing, S3 object keys |
| **Ticket Server** | 64-bit (8 Bytes) | Strictly Sequential | Yes | Centralized relational IDs |

---

## 3. Production Twitter Snowflake ID Generator Implementation

This Python module implements a **Thread-Safe Twitter Snowflake ID Generator** with custom epoch configuration, clock backward-drift safeguards, and multi-threaded worker protection.

### Python Implementation (`snowflake_generator.py`)

```python
import time
import threading

class SnowflakeIDGenerator:
    def __init__(self, node_id: int, custom_epoch: int = 1704067200000):
        """
        :param node_id: Unique worker node ID (0 - 1023)
        :param custom_epoch: Starting epoch in milliseconds (Default: Jan 1, 2024 UTC)
        """
        self.node_id_bits = 10
        self.sequence_bits = 12
        
        self.max_node_id = -1 ^ (-1 << self.node_id_bits)  # 1023
        self.max_sequence = -1 ^ (-1 << self.sequence_bits) # 4095

        if node_id < 0 or node_id > self.max_node_id:
            raise ValueError(f"Node ID must be between 0 and {self.max_node_id}")

        self.node_id = node_id
        self.custom_epoch = custom_epoch

        # Bit shift offset calculations
        self.node_id_shift = self.sequence_bits
        self.timestamp_shift = self.sequence_bits + self.node_id_bits

        self.sequence = 0
        self.last_timestamp = -1
        self.lock = threading.Lock()

    def _current_timestamp_ms(self) -> int:
        return int(time.time() * 1000)

    def _wait_for_next_ms(self, last_ts: int) -> int:
        """Blocks thread execution until clock advances to next millisecond."""
        ts = self._current_timestamp_ms()
        while ts <= last_ts:
            ts = self._current_timestamp_ms()
        return ts

    def generate_id(self) -> int:
        """Generates a thread-safe 64-bit unique, time-ordered Snowflake ID."""
        with self.lock:
            current_ts = self._current_timestamp_ms()

            # Clock Backward Drift Safeguard
            if current_ts < self.last_timestamp:
                drift_ms = self.last_timestamp - current_ts
                raise RuntimeError(
                    f"[CLOCK DRIFT DETECTED] System clock moved backwards by {drift_ms} ms. "
                    "Refusing to generate ID to protect uniqueness."
                )

            if current_ts == self.last_timestamp:
                # Same millisecond: increment local sequence counter
                self.sequence = (self.sequence + 1) & self.max_sequence
                if self.sequence == 0:
                    # Sequence capacity exhausted (4096 IDs in 1 ms); wait for next ms
                    current_ts = self._wait_for_next_ms(self.last_timestamp)
            else:
                # New millisecond: reset sequence counter
                self.sequence = 0

            self.last_timestamp = current_ts

            # Bitwise Assembly of 64-bit Integer
            snowflake_id = (
                ((current_ts - self.custom_epoch) << self.timestamp_shift) |
                (self.node_id << self.node_id_shift) |
                self.sequence
            )
            return snowflake_id

    def parse_id(self, snowflake_id: int) -> dict:
        """Deconstructs a Snowflake ID into its component parts."""
        sequence = snowflake_id & self.max_sequence
        node_id = (snowflake_id >> self.node_id_shift) & self.max_node_id
        timestamp_ms = (snowflake_id >> self.timestamp_shift) + self.custom_epoch

        return {
            "snowflake_id": snowflake_id,
            "timestamp_ms": timestamp_ms,
            "node_id": node_id,
            "sequence": sequence
        }


# -------------------------------------------------------------------
# VERIFICATION / SIMULATION RUNNER
# -------------------------------------------------------------------
if __name__ == "__main__":
    generator = SnowflakeIDGenerator(node_id=42)

    print("--- Single Thread ID Generation Test ---")
    generated_ids = []
    for _ in range(5):
        uid = generator.generate_id()
        generated_ids.append(uid)
        parsed = generator.parse_id(uid)
        print(f"ID: {uid} | Parsed: {parsed}")

    print("\n--- Multi-Thread Concurrency Test ---")
    results = set()
    threads = []

    def worker_task():
        for _ in range(1000):
            uid = generator.generate_id()
            results.add(uid)

    for _ in range(10):
        t = threading.Thread(target=worker_task)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Total Unique IDs Generated across 10 threads: {len(results)} (Expected: 10000)")
    assert len(results) == 10000, "Collision detected in concurrent ID generation!"
```

## 4. Operational Best Practices

* Clock Drift Mitigation: Use NTP configured in slewing mode (ntp -x) rather than stepping mode to prevent abrupt clock jumps backward.
* Database Primary Key Efficiency: Prefer 64-bit integer Snowflake IDs over 128-bit UUID strings for SQL primary keys to maintain compact $B^+$-Tree index node sizes and fast sequential insertion rates.
* Worker Node Provisioning: Automate Node ID registration using ZooKeeper, etcd, or Kubernetes Pod ordinal indexes (e.g., StatefulSet pod index) to guarantee unique worker IDs across clusters.
