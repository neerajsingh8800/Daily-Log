# 15: Data Engineering System Design and Architecture

This module covers **Data Platform System Design**, architectural paradigms (**Lambda vs. Kappa Architectures**), back-of-the-envelope storage and throughput estimations, fault-tolerance mechanisms, and an end-to-end Python implementation of a Kappa stream-first ingestion engine.

---

## 1. System Design Paradigms: Lambda vs. Kappa Architecture

When designing large-scale enterprise data platforms, architects must balance high-throughput batch historical processing with low-latency real-time event analytics.

### Comparative Architectural Analysis

| Feature | Lambda Architecture | Kappa Architecture |
| :--- | :--- | :--- |
| **Data Paths** | Two distinct paths (Speed Layer + Batch Layer). | Single unified stream-processing path. |
| **Code Maintenance**| High operational complexity; requires maintaining dual codebases (e.g., Spark Batch + Storm/Flink). | Low operational complexity; single processing engine handles real-time and reprocessing streams. |
| **Reprocessing Model** | Reruns batch jobs over historical raw storage buckets. | Replays historical offset windows directly from immutable stream logs. |
| **Data Consistency**| Risk of eventual consistency mismatches between real-time and batch views. | Strong consistent state guarantees using unified windowing. |

---

## 2. Back-of-the-Envelope Capacity Planning Calculus

System design interviews and production deployments require precise data volume estimation to prevent memory bottlenecks and storage exhaustion.

### 1. Daily Ingestion Storage Estimate ($S_{daily}$)
Let $N_{events}$ be the number of daily incoming events, $Bytes_{event}$ be the average payload size per event in bytes, and $R_{replication}$ be the storage replication factor ($R = 3$ for standard fault-tolerant clusters):

$$S_{daily} = N_{events} \times Bytes_{event} \times R_{replication}$$

$$\text{Example: } 100\text{ Million Events/Day} \times 1\text{ KB/Event} \times 3 = 300\text{ GB/Day Uncompressed}$$

Applying a standard Parquet/Snappy compression factor $C \approx 0.20$ (80% size reduction):

$$S_{compressed} = S_{daily} \times C = 300\text{ GB} \times 0.20 = 60\text{ GB/Day}$$

---

### 2. Required Ingestion Throughput ($MB/s$)
Let $T_{peak}$ be the peak traffic multiplier relative to average daily throughput ($T_{peak} \approx 2.5 \times \text{Average}$):

$$\text{Throughput}_{avg} = \frac{N_{events} \times Bytes_{event}}{86,400 \text{ seconds}}$$

$$\text{Throughput}_{peak} = \text{Throughput}_{avg} \times T_{peak}$$

$$\text{Throughput}_{avg} = \frac{100,000,000 \times 1,000 \text{ Bytes}}{86,400} \approx 1.157 \text{ MB/sec}$$

$$\text{Throughput}_{peak} = 1.157 \text{ MB/sec} \times 2.5 = 2.892 \text{ MB/sec}$$

---

## 3. Core Architectural Trade-offs & High-Availability Patterns

### 1. Storage Choice: Row-Based vs. Columnar vs. Key-Value
*   **OLTP / Key-Value (e.g., PostgreSQL / DynamoDB):** Optimized for low-latency row point lookups and transactional ACID writes.
*   **OLAP Columnar (e.g., Snowflake / ClickHouse):** Optimized for analytical aggregation scans (`SUM`, `COUNT`, `AVG`) across billions of rows.

### 2. High Availability & Distributed Consensus
*   **Leader-Follower Replication:** Active primary node handles all writes; read replicas maintain synchronized copies.
*   **Quorum Consensus:** To safely accept a write across $N$ cluster replicas, at least $W$ nodes must acknowledge write operations, and $R$ nodes must be read:

$$W + R > N$$

---

## 4. Production Implementation: Kappa Architecture Ingestion Simulator

Here is a complete, production-grade Python script simulating a Kappa Architecture pipeline featuring streaming event ingestion, sliding window aggregation, and automated replay capabilities.

```python
import json
import time
from collections import deque
from datetime import datetime, timezone

# -------------------------------------------------------------------
# 1. Immutable Event Log Store (Stream Source)
# -------------------------------------------------------------------
class ImmutableEventStreamLog:
    """Simulates an immutable, append-only distributed event stream log (e.g., Kafka topic)."""

    def __init__(self):
        self._log = []

    def append_event(self, key: str, payload: dict):
        offset = len(self._log)
        event = {
            "offset": offset,
            "key": key,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload
        }
        self._log.append(event)
        return offset

    def read_from_offset(self, start_offset: int = 0):
        """Allows real-time streaming and historical replay from arbitrary offsets."""
        for event in self._log[start_offset:]:
            yield event


# -------------------------------------------------------------------
# 2. Kappa Stream Processing Engine
# -------------------------------------------------------------------
class KappaStreamProcessor:
    """Unified processor handling both real-time stream processing and historical replay."""

    def __init__(self, stream_log: ImmutableEventStreamLog):
        self.stream_log = stream_log
        self.serving_layer_view = {}

    def process_event(self, event: dict):
        """Applies real-time aggregations over incoming events."""
        key = event["key"]
        amount = event["payload"].get("amount", 0.0)

        if key not in self.serving_layer_view:
            self.serving_layer_view[key] = {"total_revenue": 0.0, "event_count": 0}

        self.serving_layer_view[key]["total_revenue"] += amount
        self.serving_layer_view[key]["event_count"] += 1

    def run_realtime_stream(self, start_offset: int = 0):
        """Processes events continuously from the stream."""
        print(f"\n--- Processing Stream starting from Offset {start_offset} ---")
        for event in self.stream_log.read_from_offset(start_offset):
            self.process_event(event)

    def replay_historical_data(self):
        """Replays all historical events from Offset 0 (Kappa Reprocessing Model)."""
        print("\n--- Executing Full Historical Replay (Resetting Serving Views) ---")
        self.serving_layer_view.clear()
        self.run_realtime_stream(start_offset=0)


# -------------------------------------------------------------------
# 3. System Design Simulation Execution
# -------------------------------------------------------------------
def main():
    print("--- 1. Initializing Kappa Stream Architecture ---")
    event_log = ImmutableEventStreamLog()
    processor = KappaStreamProcessor(event_log)

    # Simulate live streaming ingested events
    print("Simulating real-time event ingestion...")
    event_log.append_event("US_EAST", {"amount": 150.00, "user_id": 101})
    event_log.append_event("US_WEST", {"amount": 300.50, "user_id": 102})
    event_log.append_event("US_EAST", {"amount": 89.90, "user_id": 103})

    # Run real-time stream execution
    processor.run_realtime_stream(start_offset=0)
    print("Serving View State After Real-Time Batch:")
    print(json.dumps(processor.serving_layer_view, indent=2))

    # Simulate new incoming stream events
    print("\nIngesting additional stream events...")
    event_log.append_event("US_EAST", {"amount": 500.00, "user_id": 104})
    
    # Process incrementally from offset 3
    processor.run_realtime_stream(start_offset=3)
    print("Serving View State After Incremental Stream:")
    print(json.dumps(processor.serving_layer_view, indent=2))

    # Demonstrate Kappa Historical Replay capability
    processor.replay_historical_data()
    print("Serving View State After Full Replay Validation:")
    print(json.dumps(processor.serving_layer_view, indent=2))

if __name__ == "__main__":
    main()
```
