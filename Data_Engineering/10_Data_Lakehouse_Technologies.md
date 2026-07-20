# 10: Data Lakehouse Technologies (Delta Lake & Apache Iceberg)

This module explores the architectural convergence of **Data Lakes** and **Data Warehouses** into the **Data Lakehouse** pattern. It covers open-table format mechanics, ACID transactions on cloud object stores, schema enforcement/evolution, time travel metadata capabilities, and hands-on implementations using **Delta Lake** and **Apache Iceberg**.

---

## 1. Architectural Evolution: Data Lakehouse Paradigm

Historically, enterprise data architectures required a two-tier system: raw semi-structured data was ingested into a **Data Lake** (low-cost storage, but no transactional guarantees or consistency) and then moved via complex ETL into a **Data Warehouse** (high query performance and ACID compliance, but expensive and locked into proprietary formats).

The **Data Lakehouse** unifies both worlds by bringing data warehouse reliability and ACID transactions directly on top of open object storage formats (Parquet/ORC).

### Architectural Comparison

| Feature | Data Lake (e.g., S3 + Parquet) | Data Warehouse (e.g., Snowflake) | Data Lakehouse (Delta / Iceberg) |
| :--- | :--- | :--- | :--- |
| **Storage Cost** | Very Low (Open cloud storage) | Proprietary / High | Very Low (Open cloud storage) |
| **Format Openness** | Fully Open (Parquet, ORC, CSV) | Proprietary internal formats | Fully Open (Standard Parquet files) |
| **ACID Compliance** | ❌ None (Risk of partial writes) | ✅ Strict ACID | ✅ Strict ACID |
| **Schema Governance** | ❌ Schema-on-Read (Prone to corruption) | ✅ Strict Schema Enforcement | ✅ Schema Enforcement & Safe Evolution |
| **Time Travel / Versioning** | ❌ Requires manual backups | ✅ Native Time Travel | ✅ Native Snapshot Time Travel |

---

## 2. Core Mechanics: Open Table Formats & Transaction Logs

The core engine enabling Lakehouse functionality is the **Open Table Format**. Instead of treating a directory of Parquet files as a table based on file paths, a Lakehouse defines a table by tracking state explicitly through a metadata management layer.

### How Transaction Logs Work
Every operation (`INSERT`, `UPDATE`, `DELETE`, `MERGE`) creates an atomic **commit** in an underlying transaction log directory.

*   **Optimistic Concurrency Control (OCC):** Writers attempt to write data files and log commits concurrently without locking the table. If two writers attempt a commit on the same version, Spark or the execution engine checks if the underlying files overlap. If no conflict exists, the commit succeeds; otherwise, it retries automatically.
*   **ACID Guarantees:** Readers only inspect file versions logged in successful commit manifests, ensuring complete isolation from concurrent uncommitted writes.

---

## 3. Mathematical Modeling: Compaction Efficiency Ratio ($CER$)

Frequent micro-batch streaming or small updates produce thousands of tiny Parquet files (the "Small File Problem"), degrading metadata scanning and read performance. Lakehouse engines run background file compaction (`OPTIMIZE` / `REWRITE`) to pack small files into target sizes (typically 128MB–1GB).

To measure compaction efficiency across a Lakehouse storage layer, engineers calculate the **Compaction Efficiency Ratio ($CER$)**:

Let $F_{initial}$ be the total number of small file blocks before compaction, $F_{compacted}$ be the resulting consolidated file count, and $S_{avg\_compacted}$ be the target average file size in megabytes.

$$CER = \frac{F_{initial} - F_{compacted}}{F_{initial}} \times 100\%$$

$$\text{Ideal Compaction Target:} \quad CER \ge 85\% \quad \text{and} \quad S_{avg\_compacted} \approx 128\text{MB} - 512\text{MB}$$

*   **High $CER$ (e.g., 90%):** Significantly reduces object store listing API calls (`LIST`/`GET`) and speeds up reader file pruning paths.

---

## 4. Key Capabilities: Time Travel & Schema Governance

### 1. Time Travel & Snapshot Isolation
Because transaction logs track every atomic change and retain old data files until garbage-collected (`VACUUM` / `EXPIRE SNAPSHOTS`), users can query historical states of a table using specific version numbers or timestamps.

*   **Use Cases:** Auditing regulatory updates, rolling back accidental deletions, and reproducing machine learning model training datasets.

### 2. Schema Enforcement vs. Schema Evolution
*   **Schema Enforcement:** Rejects any incoming write payload containing columns or data types that do not strictly match the table's registered metadata schema.
*   **Schema Evolution:** Safely allows users to add new nullable columns or merge shifting attributes without requiring full table rewrites or breaking downstream read pipelines.

---

## 5. Production Implementations: PySpark with Delta Lake & Apache Iceberg

Here is a production-grade Python script using PySpark demonstrating ACID transactions, schema evolution, UPSERT (`MERGE INTO`) operations, and Time Travel queries on a Lakehouse architecture.

```python
import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit

def init_lakehouse_spark() -> SparkSession:
    """Initializes a local PySpark session configured with Delta Lake packages."""
    return SparkSession.builder \
        .appName("Data_Lakehouse_Technologies") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

def main():
    spark = init_lakehouse_spark()
    lakehouse_path = os.path.abspath("./tmp/lakehouse_customer_orders")

    # Clean up previous directory runs if present
    if os.path.exists(lakehouse_path):
        shutil.rmtree(lakehouse_path)

    try:
        print("\n--- 1. Initial Batch Ingestion (Version 0) ---")
        data_v0 = [
            (101, "Alice", "NEW", 150.00),
            (102, "Bob", "PROCESSING", 200.50),
            (103, "Charlie", "DELIVERED", 89.90)
        ]
        columns = ["order_id", "customer_name", "order_status", "amount"]
        df_v0 = spark.createDataFrame(data_v0, columns)

        # Write out initial batch using Delta Lake format
        df_v0.write.format("delta").mode("overwrite").save(lakehouse_path)
        print(f"Data successfully saved to Lakehouse path: {lakehouse_path}")

        print("\n--- 2. Executing Schema Evolution (Version 1) ---")
        # Ingest new batch containing an additional column ('discount')
        data_v1 = [
            (104, "David", "NEW", 310.00, 15.00)
        ]
        columns_v1 = ["order_id", "customer_name", "order_status", "amount", "discount"]
        df_v1 = spark.createDataFrame(data_v1, columns_v1)

        # MergeSchema option enables safe schema evolution on write
        df_v1.write.format("delta") \
            .option("mergeSchema", "true") \
            .mode("append") \
            .save(lakehouse_path)

        print("\n--- Current Lakehouse Table State (Version 1) ---")
        current_df = spark.read.format("delta").load(lakehouse_path)
        current_df.orderBy("order_id").show()

        print("\n--- 3. Performing UPSERT (MERGE INTO) Operation ---")
        from delta.tables import DeltaTable

        delta_table = DeltaTable.forPath(spark, lakehouse_path)

        # Incoming updates dataframe: Update order_id 102 status & insert order_id 105
        updates_data = [
            (102, "Bob", "COMPLETED", 200.50, 0.00),
            (105, "Eva", "NEW", 450.00, 25.00)
        ]
        updates_df = spark.createDataFrame(updates_data, columns_v1)

        # Atomic Merge / Upsert Statement
        delta_table.alias("target").merge(
            updates_df.alias("source"),
            "target.order_id = source.order_id"
        ).whenMatchedUpdate(set={
            "order_status": col("source.order_status"),
            "amount": col("source.amount")
        }).whenNotMatchedInsertAll().execute()

        print("\n--- Table State After Merge ---")
        delta_table.toDF().orderBy("order_id").show()

        print("\n--- 4. Executing Time Travel Query (Version 0) ---")
        # Reading Version 0 before updates and schema changes
        df_version_0 = spark.read.format("delta") \
            .option("versionAsOf", 0) \
            .load(lakehouse_path)
        
        print("Historical Snapshot (Version 0 Data):")
        df_version_0.orderBy("order_id").show()

        print("\n--- 5. Reviewing Transaction Log History Metadata ---")
        history_df = delta_table.history()
        history_df.select("version", "timestamp", "operation", "operationParameters").show(truncate=False)

    finally:
        # Clean up temporary local directory
        if os.path.exists(lakehouse_path):
            shutil.rmtree(lakehouse_path)
        spark.stop()

if __name__ == "__main__":
    main()
```
