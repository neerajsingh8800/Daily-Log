# 09: Cloud Data Warehouses Architecture

This module deep-dives into the architecture of modern, cloud-native data warehouses like **Snowflake** and **Google BigQuery**. It covers the paradigm shift of decoupling storage from compute, the mathematics of micro-partitioning, and practical query optimization using clustering keys.

---

## 1. Architectural Evolution: Shared-Nothing vs. Decoupled Storage & Compute

Traditional on-premise analytical databases utilized a **Shared-Nothing Architecture** where each node maintained its own localized CPU, memory, and dedicated SSD storage. While fast for localized operations, scaling required purchasing uniform hardware slots, leading to inefficient resource utilization.

Modern cloud data warehouses introduced a **Decoupled Storage and Compute Architecture**.

### Core Architecture Comparison

| Metric | Shared-Nothing (Legacy OLAP) | Decoupled Architecture (Snowflake / BigQuery) |
| :--- | :--- | :--- |
| **Scaling Boundaries** | Scaling storage requires scaling compute simultaneously. | Scale compute up/down instantly or pause entirely; storage scales independently. |
| **Storage Medium** | Localized High-IOPS NVMe SSD drives. | Low-cost centralized cloud object stores (S3, GCS). |
| **State Persistence** | State is tied to active physical server nodes. | State is completely persistent in the storage layer; compute is ephemeral. |
| **Concurrency Management**| High concurrency causes resource contention and query queuing. | Spin up isolated warehouse clusters over the same shared data layer without contention. |

---

## 2. Storage Subsystem Engineering: Micro-Partitioning & Columnar Layouts

Data warehouses do not write data in traditional table rows. Instead, incoming rows are automatically ingested, compressed, and written out into immutable, structured files called **Micro-Partitions** (typically ranging from 50MB to 500MB in size).

### Row-Oriented vs. Columnar Storage Layout
Consider an employee metadata table. A columnar engine groups data values vertically by attribute rather than horizontally by individual record:

Row-Oriented Layout (OLTP):     [Row1: ID, Name, Dept][Row2: ID, Name, Dept]
Columnar Layout (OLAP Warehouses): [All IDs][All Names][All Departments]


### The Power of Pruning
Each micro-partition maintains structured header metadata tracking the **Minimum and Maximum values** for every single column contained within that specific file block.

When an analytical query executes a `WHERE` filter clause, the cloud service optimizer reviews these min/max bounds first, completely skipping (**pruning**) millions of unmatching files without reading a single byte from object storage.

---

## 3. Mathematical Modeling of Partition Pruning Efficiency

To measure how effectively an analytical data warehouse prunes unnecessary storage blocks during a query run, engineers calculate the **Pruning Efficiency Index ($PEI$)**:

Let $P_{total}$ be the total number of micro-partitions composing an analytical table layout, and $P_{scanned}$ be the exact number of file blocks physically opened and parsed by the execution cluster engine.

$$PEI = 1 - \frac{P_{scanned}}{P_{total}}$$

*   **Optimal Pruning Layout:** A $PEI \to 1.0$ indicates that the database engine successfully pruned almost all files, fetching data instantly using minimal compute resource configurations.
*   **Poor Layout (Scan Heavy):** A $PEI \to 0.0$ means the engine executed a full table scan across every partition block, indicating significant structural **data clustering degradation**.

---

## 4. Query Optimization via Clustering Keys

When data is naturally loaded chronologically, it implicitly structures itself efficiently by date bounds. However, if business queries frequently filter across a different non-chronological attribute (like `tenant_id` or `region`), data can become scattered across all partitions over time, killing pruning efficiency.

### What is Clustering?
Clustering is the process of physically co-locating rows with matching column keys into the same micro-partition blocks. 

*   **Clustering Depth:** A mathematical measurement of how overlapping and scattered column values are across a table's micro-partitions. Lower depth equals highly optimized data layout configurations.

#### Practical Snowflake Optimization Script:
```sql
-- Step 1: Instantiate a high-performance transactional orders target table
CREATE OR REPLACE TABLE analytics_prod.marts.fct_global_transactions (
    transaction_id VARCHAR(64),
    tenant_id VARCHAR(32),
    transaction_date DATE,
    gross_amount NUMBER(12,2),
    tax_amount NUMBER(12,2)
)
-- Establish a composite clustering layout structure to speed up lookups
CLUSTER BY (tenant_id, transaction_date);

-- Step 2: Validate the current structural clustering health state metrics
SELECT SYSTEM$CLUSTERING_INFORMATION('analytics_prod.marts.fct_global_transactions');

-- Step 3: Explicitly alter the operational warehouse cluster state to maintain key bounds
ALTER TABLE analytics_prod.marts.fct_global_transactions RESUME RECLUSTER;
5. Production Google BigQuery Optimization Implementations
Google BigQuery uses two main design methods to minimize compute query byte scanning overheads: Partitioning (physically splitting data by date/integer ranges) and Clustering (sorting data strings inside those partitions).

Here is a complete production DDL deployment script demonstrating optimal architectural layouts in BigQuery SQL:

SQL
-- Creating an optimized enterprise logging mart leveraging both Partitioning and Clustering
CREATE OR REPLACE TABLE `bi_analytics_dw.security_logs.mart_user_activities`
(
    event_timestamp TIMESTAMP OPTIONS(description="Precise chronological logging time"),
    user_id STRING OPTIONS(description="Unique system identity string identifier"),
    ip_address STRING,
    action_type STRING,
    payload_size_kb INT64
)
-- 1. Apply Partitioning: Physically isolates log data into daily slots based on time bounds
PARTITION BY DATE(event_timestamp)
-- 2. Apply Clustering: Sorts rows within each daily slot by identity strings and actions
CLUSTER BY user_id, action_type
OPTIONS(
    require_partition_filter = true, -- Hard guardrails: Prevents users from executing costly full-table scans
    partition_expiration_days = 365  -- Automated data lifecycle governance enforcement
);

-- Production Analytical Execution Query Example:
-- The engine will ONLY charge compute fees for scanning the single partition matching the filtered date
SELECT 
    user_id,
    COUNT(action_type) as login_failure_count
FROM 
    `bi_analytics_dw.security_logs.mart_user_activities`
WHERE 
    DATE(event_timestamp) = DATE("2026-07-19") -- Hits the partition index directly
    AND user_id = 'USER-8800'                 -- Uses the cluster sort key to prune blocks within that day
GROUP BY 
    1;
```
