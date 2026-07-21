# 11: Real-Time Streaming with Apache Kafka

This module explores **Event-Driven Architectures**, real-time data streaming mechanics, the architectural core of **Apache Kafka**, partition consumer calculus, write-ahead logs, and production-grade implementations in Python using `confluent-kafka`.

---

## 1. Batch Processing vs. Real-Time Event Streaming

Traditional data pipelines rely on **Batch Processing**, where data is collected over an interval (e.g., hourly or daily) and processed as static bulk chunks. **Real-Time Streaming** shifts this paradigm to continuous event processing as data is generated.

| Feature | Batch Processing (e.g., Spark Batch / dbt) | Real-Time Event Streaming (e.g., Apache Kafka) |
| :--- | :--- | :--- |
| **Data Boundary** | Bounded datasets (finite start and end). | Unbounded datasets (infinite, continuous event streams). |
| **Latency Boundary** | Minutes to Hours (High Latency). | Sub-second / Milliseconds (Low Latency). |
| **Processing Paradigm** | Schedule-driven (Cron, Airflow). | Event-driven (Triggered instantly by state changes). |
| **Storage Mechanism** | Static files / tables on disk. | Immutable append-only write-ahead commit logs. |

---

## 2. Core Architecture of Apache Kafka

Apache Kafka is a distributed event-streaming platform built on an append-only distributed commit log model.

### Core Architecture Components:
*   **Producer:** Client applications that publish (write) events to Kafka topics.
*   **Broker:** Individual server nodes composing the Kafka cluster, responsible for receiving, storing, and serving event logs.
*   **Topic:** A logical channel or category where events are organized and stored.
*   **Partition:** Physical log splits of a topic across brokers that enable horizontal parallelism and high throughput.
*   **Offset:** A monotonically increasing 64-bit integer assigned to each message within a partition, acting as a unique logical address.
*   **Consumer Group:** A collection of consumers working together to read data from a topic in parallel. Each partition is assigned to exactly one consumer worker within a group.

---

## 3. Mathematical Modeling: Consumer Partition Calculus & Lag

### 1. The Parallelism Bound Equation
To ensure active parallelism without idle thread overhead, the number of active consumer instances ($C$) within a single consumer group must not exceed the total partition count ($P$) of the target topic:

$$C_{active} \le P_{total}$$

$$\text{If } C_{active} > P_{total} \implies (C_{active} - P_{total}) \ \text{Consumers Remain Idle}$$

### 2. Consumer Lag Metric
**Consumer Lag** represents the delay between the latest offset written to a partition by producers ($O_{latest}$) and the current offset processed and committed by a consumer group ($O_{committed}$).

$$\text{Lag}_i = O_{latest, \ i} - O_{committed, \ i}$$

$$\text{Total Consumer Group Lag} = \sum_{i=0}^{P_{total}-1} \left( O_{latest, \ i} - O_{committed, \ i} \right)$$

*   **Engineering Rule:** A growing total lag indicates that downstream analytics consumers are lagging behind incoming event velocities, requiring horizontal partition re-scaling or consumer performance tuning.

---

## 4. Message Delivery Semantics & Guarantees

When designing streaming applications, data engineers configure producers and consumers to enforce specific delivery semantics:

1.  **At-Most-Once:** Messages may be lost, but are never duplicated. Offsets are committed *before* processing completes.
2.  **At-Least-Once (Default Standard):** No messages are lost, but duplicates may occur if a worker crashes before committing offsets. Requires downstream processing systems to handle **idempotency**.
3.  **Exactly-Once Processing (EOS):** Messages are delivered and processed exactly once end-to-end using Kafka's transactional producer APIs and atomic offset commits.

---

## 5. Production Python Implementation using `confluent-kafka`

Here is a complete, production-grade event streaming application featuring both a **Kafka Event Producer** (with delivery callbacks and key-based partitioning) and a **Kafka Event Consumer** (with offset tracking and graceful shutdown hooks).

```python
import json
import time
from confluent_kafka import Consumer, KafkaError, Producer

# -------------------------------------------------------------------
# 1. Producer Implementation: Stream JSON Event Logs
# -------------------------------------------------------------------
def delivery_report(err, msg):
    """Callback triggered on successful event delivery or failure."""
    if err is not None:
        print(f"❌ Event delivery failed: {err}")
    else:
        print(f"✅ Event published to {msg.topic()} [Partition {msg.partition()}] @ Offset {msg.offset()}")

def run_producer():
    producer_config = {
        'bootstrap.servers': 'localhost:9092',
        'acks': 'all',  # Guarantee strong durability across replicas
        'retries': 3,
        'compression.type': 'snappy'  # Reduce network payload bandwidth
    }

    producer = Producer(producer_config)
    topic_name = "ecom_user_purchases"

    print(f"\n--- Starting Kafka Event Producer [Topic: {topic_name}] ---")

    # Generate sample e-commerce purchasing events
    events = [
        {"order_id": "ORD-1001", "user_id": "USER-501", "amount": 120.50, "currency": "INR"},
        {"order_id": "ORD-1002", "user_id": "USER-502", "amount": 450.00, "currency": "USD"},
        {"order_id": "ORD-1003", "user_id": "USER-501", "amount": 89.90, "currency": "INR"},
    ]

    for event in events:
        # Using user_id as partition key ensures ordering per user
        key_bytes = event["user_id"].encode('utf-8')
        value_bytes = json.dumps(event).encode('utf-8')

        producer.produce(
            topic=topic_name,
            key=key_bytes,
            value=value_bytes,
            callback=delivery_report
        )
        # Flush internal buffer to push events out
        producer.poll(0)
        time.sleep(0.5)

    producer.flush()
    print("Producer successfully sent all events.\n")

# -------------------------------------------------------------------
# 2. Consumer Implementation: Stream Event Listener
# -------------------------------------------------------------------
def run_consumer():
    consumer_config = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'analytics_ingestion_group',
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False  # Manual offset commit for At-Least-Once processing
    }

    consumer = Consumer(consumer_config)
    topic_name = "ecom_user_purchases"

    consumer.subscribe([topic_name])
    print(f"--- Listening for events on group 'analytics_ingestion_group' [Topic: {topic_name}] ---")

    events_processed = 0
    max_events_to_read = 3

    try:
        while events_processed < max_events_to_read:
            msg = consumer.poll(timeout=2.0)

            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    print(f"Reached end of partition {msg.partition()}")
                else:
                    print(f"Consumer Error: {msg.error()}")
                break

            # Deserialize string data
            event_key = msg.key().decode('utf-8') if msg.key() else None
            event_data = json.loads(msg.value().decode('utf-8'))

            print(f"Received Event: Key={event_key} | Data={event_data} | Partition={msg.partition()} | Offset={msg.offset()}")

            # Perform operational processing logic here...
            
            # Manually commit offset after processing completion
            consumer.commit(msg, asynchronous=False)
            events_processed += 1

    except KeyboardInterrupt:
        print("\nConsumer interrupted by user.")
    finally:
        consumer.close()
        print("Consumer connection closed safely.")

if __name__ == "__main__":
    # Simulate producer execution followed by stream consumer ingestion
    run_producer()
    run_consumer()
```
