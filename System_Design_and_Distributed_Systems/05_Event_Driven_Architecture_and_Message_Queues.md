# Module 05: Event-Driven Architecture and Message Queues

Event-Driven Architecture (EDA) decouples software components by producing, detecting, and consuming events asynchronously. Instead of synchronous, tightly-coupled RPC/REST calls, services communicate by publishing immutable state changes (events) to distributed log streams or message queues.

This module covers message broker paradigms, delivery semantics, consumer group partition rebalancing mechanics, Little's Law for queue sizing, transactional outbox patterns, and a complete Apache Kafka producer/consumer implementation using Python.

---

## 1. Theoretical Foundations

### 1.1 Broker Paradigms: Message Queues vs. Distributed Commit Logs
* **Message Queues (e.g., RabbitMQ, AMQP)**:
  * Uses a **Smart Broker / Dumb Consumer** pattern.
  * Messages are pushed to consumers and deleted upon acknowledgment ($ACK$).
  * Best for transient task queuing, complex routing, and worker-pool distribution.
* **Distributed Commit Logs (e.g., Apache Kafka, Pulsar)**:
  * Uses a **Dumb Broker / Smart Consumer** pattern.
  * Append-only log files partitioned across disks.
  * Messages persist based on retention policy (e.g., 7 days) and are pulled by consumers using managed offset tracking.
  * Enables replayability, event-sourcing, and horizontal scaling across consumer groups.

---

### 1.2 Mathematical Foundations of Queueing Systems & Partitioning

#### Little's Law for Queue Sizing
In any stable queueing system, the average number of messages $L$ in a message broker queue equals the average arrival rate $\lambda$ multiplied by the average message processing time $W$:

$$L = \lambda \cdot W$$

*Example*: If a system receives $\lambda = 5,000 \text{ msgs/sec}$ and worker services take $W = 0.05 \text{ seconds}$ ($50\text{ms}$) to process each message:

$$L = 5000 \cdot 0.05 = 250 \text{ messages in queue}$$

To maintain stability without backpressure buildup, the total consumer processing capacity $C_{\text{total}}$ across $K$ workers must satisfy:

$$C_{\text{total}} = \frac{K}{W} > \lambda \implies K > \lambda \cdot W$$

#### Partition Routing Hash Math
To guarantee strict message ordering for a specific entity (e.g., order or user updates), messages with identical key $K$ are routed to the same partition index $P$ out of $N$ total topic partitions:

$$P = \Big( \text{MurmurHash2}(K) \ \& \ \text{0x7FFFFFFF} \Big) \pmod N$$

---

## 2. Delivery Semantics & Reliability Frameworks

| Semantic Model | Description | Failure Handling / Risk | Best Use Case |
| :--- | :--- | :--- | :--- |
| **At-Most-Once** | Offsets committed *before* processing message. | Zero duplicate messages; risk of data loss on worker crash. | High-frequency telemetry / metrics collection |
| **At-Least-Once** | Offsets committed *after* successful processing. | Zero data loss; risk of duplicate messages during network retries. | Financial ledgers, notification systems |
| **Exactly-Once (EOS)** | Atomic transactional reads, processing, and offset writes. | Higher execution latency due to two-phase commit tracking. | Payment processing, audit-critical event streams |

---

## 3. Transactional Outbox Pattern

To avoid dual-write failures (e.g., database commit succeeds but message broker publish fails), systems insert events into a local database **Outbox Table** within the same ACID transaction:

1. **Service DB Transaction**: Update application table AND insert record into `outbox` table atomically.
2. **Outbox Processor (CDC / Debezium)**: Reads non-published events from `outbox` table and publishes them to Kafka/RabbitMQ.
3. **Mark Processed**: Set `processed = TRUE` upon receiving broker confirmation.

---

## 4. Production Kafka Producer/Consumer Implementation

This Python module implements an **At-Least-Once Kafka Producer and Consumer** with explicit manual offset management, error handling, and dead-letter queue routing using `confluent-kafka`.

### Prerequisites

```bash
pip install confluent-kafka
```

### Python Implementation (event_stream.py)
```python
import json
import time
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC_NAME = "order_events"
DLQ_TOPIC = "order_events_dlq"

# -------------------------------------------------------------------
# 1. AT-LEAST-ONCE PRODUCER WITH ACKNOWLEDGMENT TRACKING
# -------------------------------------------------------------------
class ResilientProducer:
    def __init__(self, bootstrap_servers: str):
        conf = {
            'bootstrap.servers': bootstrap_servers,
            'acks': 'all',                  # Wait for leader and all in-sync replicas
            'retries': 5,                   # Retry transient connection errors
            'max.in.flight.requests.per.connection': 1  # Preserve partition message ordering
        }
        self.producer = Producer(conf)

    def delivery_report(self, err, msg):
        """Callback triggered on successful publish or terminal failure."""
        if err is not None:
            print(f"[PRODUCER ERROR] Message delivery failed: {err}")
        else:
            print(f"[PRODUCER SUCCESS] Event sent to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")

    def publish_event(self, topic: str, key: str, payload: dict):
        serialized_data = json.dumps(payload).encode('utf-8')
        self.producer.produce(
            topic=topic,
            key=key.encode('utf-8'),
            value=serialized_data,
            on_delivery=self.delivery_report
        )
        self.producer.poll(0)  # Trigger callbacks asynchronously

    def flush(self):
        self.producer.flush()


# -------------------------------------------------------------------
# 2. CONSUMER GROUP WITH MANUAL OFFSET COMMIT & DLQ
# -------------------------------------------------------------------
class ReliableConsumer:
    def __init__(self, bootstrap_servers: str, group_id: str):
        conf = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False   # Manual offset control for At-Least-Once semantics
        }
        self.consumer = Consumer(conf)
        self.dlq_producer = ResilientProducer(bootstrap_servers)

    def start_listening(self, topics: list):
        self.consumer.subscribe(topics)
        print(f"[CONSUMER STARTED] Listening on topics: {topics}")

        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        raise KafkaException(msg.error())

                # Process event
                key = msg.key().decode('utf-8') if msg.key() else "N/A"
                value = json.loads(msg.value().decode('utf-8'))
                
                success = self._process_business_logic(key, value)

                if success:
                    # Explicit manual synchronous offset commit
                    self.consumer.commit(msg, asynchronous=False)
                else:
                    # Move bad payload to Dead Letter Queue (DLQ)
                    print(f"[DLQ ROUTE] Event key '{key}' failed processing. Forwarding to DLQ.")
                    self.dlq_producer.publish_event(DLQ_TOPIC, key, value)
                    self.dlq_producer.flush()
                    self.consumer.commit(msg, asynchronous=False)

        except KeyboardInterrupt:
            print("\n[CONSUMER STOPPED] Closing consumer connection...")
        finally:
            self.consumer.close()

    def _process_business_logic(self, key: str, value: dict) -> bool:
        """Simulates business operation; returns False if processing fails."""
        try:
            print(f"[PROCESSING] Key: {key} | Order Status: {value.get('status')}")
            if value.get("amount", 0) < 0:
                raise ValueError("Invalid negative order amount")
            return True
        except Exception as e:
            print(f"[LOGIC ERROR] Failed to process message: {e}")
            return False


# -------------------------------------------------------------------
# VERIFICATION / SIMULATION RUNNER
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("Kafka Producer & Consumer Module Loaded.")
    print("To test live, run local Kafka: `docker-compose up -d kafka`")
```
## 5. Architectural Best Practices

* Idempotent Consumers: Always design downstream consumer handlers to be idempotent (e.g., using unique event IDs) to safely withstand duplicate deliveries caused by retries.

* Partition Scaling: Align the number of topic partitions with maximum anticipated consumer instances; idle consumers occur when consumer count exceeds total partitions.

* Dead Letter Queue (DLQ) Alerts: Monitor DLQ growth closely—messages routed to DLQ indicate schema drift or persistent application bugs requiring manual inspection.

