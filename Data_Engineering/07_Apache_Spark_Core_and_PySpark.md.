# 07: Apache Spark Core and PySpark

This module explores the core mechanics of distributed computing, transitioning from disk-bound batch architectures (Hadoop MapReduce) to in-memory processing with **Apache Spark**, architectural calculations, lazy evaluation, and hands-on big data manipulation using **PySpark**.

---

## 1. The Distributed Evolution: MapReduce vs. Apache Spark

To process datasets that exceed the memory and storage capacity of a single physical machine, data workloads must be distributed across a cluster of nodes.

| Evaluation Metric | Hadoop MapReduce Architecture | Apache Spark Compute Framework |
| :--- | :--- | :--- |
| **Execution Speed** | Slower. Heavy reliance on disk I/O operations between map and reduce states. | Up to 100x faster. Processes data primarily in-memory (RAM). |
| **Storage Dependency**| Strictly relies on HDFS (Hadoop Distributed File System) for state storage. | Agnostic. Can read from/write to HDFS, AWS S3, Google Cloud Storage, or Azure ADLS. |
| **Data Processing State**| Batch-only execution boundaries. | Unified engine for Batch, Interactive SQL, Real-Time Streaming, and Machine Learning. |
| **Intermediate State** | Written explicitly to physical disk (Spilling/Shuffling). | Maintained in memory unless forced to cache/persist to disk. |

---

## 2. Cluster Architecture & Execution Hierarchy

Apache Spark utilizes a master-worker clustering model to split up execution tracks across separate computational nodes.

*   **Driver Program:** The orchestrator node. It runs the main application code, instantiates the `SparkSession`, translates programmatic code into execution plans, and coordinates tasks across workers.
*   **Cluster Manager:** An external infrastructure allocator (e.g., Spark Standalone, Apache YARN, Kubernetes) that provisions resources for the Spark application.
*   **Worker Node:** Physical or virtual machine instances in the cluster dedicated to execution tasks.
*   **Executor:** A dedicated JVM process launched on a worker node to execute specialized task chunks and retain data blocks in memory.

---

## 3. Foundational Data Abstractions: RDDs vs. DataFrames

### Resilient Distributed Datasets (RDDs)
The fundamental backbone abstraction of Spark. An RDD represents an immutable, fault-tolerant distributed collection of objects partitioned across the cluster.
*   **Resilient:** Fault-tolerant via a lineage graph. If a partition fails, Spark reconstructs it automatically.
*   **Distributed:** Split across multiple nodes.
*   **Dataset:** Contains records (e.g., Python objects, tuples, or primitives).

### Spark DataFrames
Built on top of the RDD execution engine, DataFrames introduce a logical structure—conceptualizing data as tables with named columns (schema-aware). DataFrames leverage Spark's **Catalyst Optimizer** to structurally restructure queries for optimal execution paths before execution.

---

## 4. Execution Mechanics: Lazy Evaluation & Lineage

Spark optimization depends heavily on separating data manipulations into two distinct categories:

### 1. Transformations
Operations that modify an existing dataset to generate a new dataset wrapper. Transformations are **lazy**—they do not execute immediately; they are simply appended to a structural roadmap called the **Lineage Graph**.
*   **Narrow Transformations:** Each input partition contributes to at most one output partition. No data movement across network zones is required (e.g., `map()`, `filter()`, `flatMap()`).
*   **Wide Transformations:** Multiple input partitions are required to compile a single output partition. Requires a **Shuffle** operation across worker nodes (e.g., `groupByKey()`, `reduceByKey()`, `join()`).

### 2. Actions
Operations that trigger actual computations by compiling the Lineage Graph and executing tasks across worker nodes to return a result to the driver or write it to storage (e.g., `count()`, `collect()`, `saveAsTextFile()`).

### Architectural Performance: Memory Fraction Equation
Executors divide allocated JVM heap space between operational computations and storage caching mechanisms. The usable storage capacity for caching distributed RDD blocks is bounded by:

$$\text{Available Storage RAM} = \text{Executor JVM Memory} \times \text{spark.memory.fraction} \times \text{spark.memory.storageFraction}$$

---

## 5. Production PySpark Implementation: Clickstream Log Processing

Here is a comprehensive production-grade PySpark script demonstrating how to initialize a structured session, read raw data, apply transformations using both PySpark SQL functions and native RDD mechanics, and save structural aggregations back to partition storage layers.

```python
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc, sum as spark_sum, upper, when

def init_spark_session(app_name: str) -> SparkSession:
    """Initializes a high-performance local SparkSession."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()
    return spark

def main():
    # 1. Initialize local compute environment context
    spark = init_spark_session("Clickstream_Analytics_Pipeline")
    print(f"Spark Running. Engine Version: {spark.version}")

    # 2. Generate a structural mock data log file path
    mock_csv_path = "raw_clickstream_logs.csv"
    
    with open(mock_csv_path, "w") as f:
        f.write("user_id,page_id,action,revenue,country\n")
        f.write("101,home,view,0.00,India\n")
        f.write("102,cart,add,45.50,USA\n")
        f.write("101,checkout,purchase,120.00,India\n")
        f.write("103,home,view,0.00,Germany\n")
        f.write("102,checkout,purchase,45.50,USA\n")

    try:
        # 3. Read raw unstructured metrics logs into a DataFrame
        df = spark.read.csv(mock_csv_path, header=True, inferSchema=True)
        print("\n--- Raw DataFrame Schema ---")
        df.printSchema()

        # 4. Apply Narrow Transformations (Filtering & Standardizing)
        processed_df = df.filter(col("country").isNotNull()) \
            .withColumn("action_upper", upper(col("action"))) \
            .withColumn("high_value_flag", when(col("revenue") >= 100.0, 1).otherwise(0))

        # 5. Apply Wide Transformations (Grouping & Aggregating)
        # Note: This operation causes a shuffle barrier across partitions
        country_aggregation = processed_df.groupBy("country") \
            .agg(
                spark_sum("revenue").alias("total_country_revenue"),
                spark_sum("high_value_flag").alias("high_value_transaction_count")
            ) \
            .orderBy(desc("total_country_revenue"))

        # 6. Execute Action to trigger computation and print to driver terminal
        print("\n--- Analytical Aggregation Results ---")
        country_aggregation.show()

        # 7. Demonstrate fallback to low-level RDD transformation processing API
        print("Transforming DataFrame columns using underlying RDD tuples...")
        rdd_lineage = df.rdd.map(lambda row: (row["user_id"], row["revenue"]))
        rdd_reduced = rdd_lineage.reduceByKey(lambda acc, val: acc + val)
        
        print("\n--- Computed RDD Tuples Output ---")
        print(rdd_reduced.collect())

    finally:
        # Clean up local system file system artifacts
        if os.path.exists(mock_csv_path):
            os.remove(mock_csv_path)
        # Terminate core SparkContext environment safely
        spark.stop()

if __name__ == "__main__":
    main()
```
