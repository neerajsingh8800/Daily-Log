# NoSQL Architectures and Use Cases

This module explores the core principles of **NoSQL Database Architectures**, the mathematical trade-offs of the **CAP Theorem**, and structural deep-dives into the four major NoSQL categories along with their targeted production use cases.

---

## 1. Relational vs. NoSQL: The Paradigm Shift

As data pipelines scale horizontally, traditional RDBMS barriers (strict schemas, costly compute joins, single-node vertical scaling) require the introduction of non-relational distributed data stores.

| Feature | Relational (RDBMS) | NoSQL (Distributed Non-Relational) |
| :--- | :--- | :--- |
| **Data Model** | Structured tables with rigid rows and columns. | Flexible schemas: Key-Value, Document, Wide-Column, Graph. |
| **Scaling** | Vertical (Scale-Up: upgrade CPU/RAM on one node). | Horizontal (Scale-Out: distribute data across multiple commodity nodes). |
| **Transactions** | Strict ACID compliance (Atomicity, Consistency, Isolation, Durability). | BASE model (Basically Available, Soft state, Eventual consistency). |
| **Join Operations** | Native, highly optimized relational joins. | Avoided structurally; data is typically denormalized or split. |

---

## 2. Distributed Systems: The CAP Theorem

When designing a distributed data architecture across a network of nodes, Eric Brewer's **CAP Theorem** states that a system can guarantee at most **two out of three** core properties simultaneously:

*   **Consistency (C):** Every read operation across the cluster receives the most recent write or an error.
*   **Availability (A):** Every non-failing node returns a non-error response for every request (without guaranteeing it contains the most recent write).
*   **Partition Tolerance (P):** The system continues to operate despite an arbitrary number of messages being dropped or delayed by the network between nodes.

### The Trade-Off Reality
Because physical networks are inherently prone to communication drops, **Partition Tolerance (P) is mandatory** in production data engineering. Therefore, modern architectures must choose between:

1.  **CP Systems (Consistency + Partition Tolerance):** The system shuts down or rejects conflicting requests on isolated partitions to ensure absolute data uniformity (e.g., HBase, MongoDB).
2.  **AP Systems (Availability + Partition Tolerance):** The system accepts local writes and reads on all accessible partitions, allowing nodes to remain functional while syncing data asymptotically over time (e.g., Cassandra, DynamoDB).

---

## 3. The Four Core NoSQL Architectures

### 🔑 1. Key-Value Stores
Data is stored as an unindexed schema-agnostic value pair mapped directly to a unique key lookup identifier.

*   **Core Characteristics:** Ultra-low latency, memory-mapped caching engine, minimal structural overhead.
*   **Target Production Use Cases:** Session caching, real-time leaderboard states, shopping cart storage.
*   **Primary Tool:** **Redis**, Amazon DynamoDB.

#### Practical Redis Python Implementation:
```python
import redis

# Initialize distributed key-value memory cache connection
cache = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Write operational session state data (O(1) time complexity)
cache.set("user_session:neeraj5821", "authenticated_token_xyz123", ex=3600)

# Fetch data near-instantaneously
session_token = cache.get("user_session:neeraj5821")
print(f"Retrieved Session Token: {session_token}")
```
## 📄 2. Document Stores

Data is encapsulated in hierarchical, self-describing structures (JSON/BSON strings) where each document can maintain entirely unique attribute properties.

Core Characteristics: Dynamic nested schemas, flexible structure, native secondary indexes.

Target Production Use Cases: E-Commerce catalogs, Content Management Systems (CMS), user profiles.

Primary Tool: MongoDB, Couchbase.

Practical MongoDB Aggregation Pipeline Script:

```javascript
// Complex document query aggregation within a modern catalog
db.product_catalog.aggregate([
    { $match: { is_active: true, category: "Electronics" } },
    { $group: { 
        _id: "$brand", 
        average_price: { $avg: "$price" },
        total_inventory: { $sum: "$stock_count" }
    }},
    { $sort: { average_price: -1 } }
]);
```
## 🏛️ 3. Wide-Column Stores

Data is stored in column families rather than sequential rows. Rows are dynamic, meaning different rows within the same column family can have completely different columns.

Core Characteristics: High-volume sequential write optimization, massive distributed storage scaling, predictable row keys.

Target Production Use Cases: IoT time-series telemetry pipelines, clickstream tracking, financial transaction history logs.

Primary Tool: Apache Cassandra, ScyllaDB, HBase.

Cassandra Query Language (CQL) Schema Design:

```sql
-- Designing a optimized IoT sensor tracking model
CREATE KEYSPACE sensor_analytics 
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 3};

CREATE TABLE sensor_analytics.telemetry_logs (
    sensor_id UUID,
    log_date date,
    timestamp timestamp,
    metric_value double,
    PRIMARY KEY ((sensor_id, log_date), timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);
```
## 🕸️ 4. Graph Databases

Data is modeled explicitly as Nodes (entities), Edges (relationships), and Properties (key-value metadata attached to either).

Core Characteristics: Index-free adjacency (traversing deep relational links without costly SQL JOIN engines).

Target Production Use Cases: Identity resolution, social networking connection engines, fraud ring detection graphs.

Primary Tool: Neo4j, Amazon Neptune.

Neo4j Cypher Traversal Query:

```cypher
// Trace a potential credit card fraud loop through a shared account graph
MATCH (c1:Customer {id: 'CUST-8800'})-[r1:USED_IP]->(ip:IPAddress)<-[r2:USED_IP]-(c2:Customer)
WHERE c1 <> c2
RETURN c1.name, ip.address, c2.name, c2.fraud_risk_score;
```
