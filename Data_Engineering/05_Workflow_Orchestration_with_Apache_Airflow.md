# 05: Workflow Orchestration with Apache Airflow

This module covers the core fundamentals of programmatic workflow orchestration, transitioning from legacy cron scheduling to modern dynamic data pipelines, **DAG design mechanics**, data movement patterns via **XComs**, and resilient failure handling patterns.

---

## 1. The Need for Orchestration: Cron vs. Airflow

In enterprise data engineering, relying on basic system schedulers like Unix Cron introduces significant structural risks. 

| Evaluation Metric | Legacy Cron Scheduling | Apache Airflow Orchestration |
| :--- | :--- | :--- |
| **Dependency Awareness** | Time-based only. Cannot natively pause Task B if Task A fails. | Functional DAG dependency graphing. Task paths branch on upstream status. |
| **Failure Handling & Retries** | Requires custom error wrapper scripts. | Automatic retry configurations, exponential backoff, and alerting. |
| **Centralized Logging** | Scattered across system syslog or flat files. | Centralized UI showing logs per task instance run. |
| **Dynamic Execution** | Rigid static bash scripts. | Dynamic programmatic generation via Python scripting. |

---

## 2. Mathematical Modeling of Workflows: Directed Acyclic Graphs (DAGs)

Airflow organizes data pipelines as a **Directed Acyclic Graph (DAG)**. 

Mathematically, a DAG is defined as a graph structure containing a set of vertices (Tasks) $V$ and a set of directed edges (Dependencies) $E$:

$$G = (V, E)$$

### Core Invariants of a DAG:
1.  **Directed:** Edges have an explicit direction indicating execution flow:
    $$(T_1, T_2) \in E \implies T_1 \text{ must execute before } T_2$$
2.  **Acyclic:** There must be absolutely no path starting at any task $T_i$ that loops back to $T_i$. If a cycle is present:
    $$\text{Path } (T_i \rightarrow \dots \rightarrow T_i) \implies \text{Infinite Loop Error}$$

    ---

## 3. Core Airflow Pillars & Scheduling Mechanics

### 🏗️ Architecture Components
*   **Webserver:** The UI portal used to inspect logs, trigger DAGs, and review execution statuses.
*   **Scheduler:** The foundational heartbeat process that parses DAG folders, evaluates execution time boundaries, and passes ready tasks to the executor.
*   **Executor:** The computational engine abstraction that handles running tasks (e.g., `SequentialExecutor`, `LocalExecutor`, or horizontally scalable `CeleryExecutor`/`KubernetesExecutor`).

### ⏰ The Execution Date Equation
A common point of confusion is how Airflow triggers runs. A DAG run is executed only **at the end of its scheduled interval**, not at the start.

$$\text{Data Interval End} = \text{Logical Date} + \text{Schedule Interval}$$

> **Example:** If a daily DAG has a `start_date` of `2026-07-15` and a schedule of `@daily`, the first run will execute at **00:00:00 on 2026-07-16**. The `logical_date` (historically called execution date) inside the context logs will read `2026-07-15`, representing the analytical window of data being extracted.

---

## 4. Operational State Transmission: Tasks vs. XComs

Airflow tasks run in isolated environments or separate worker nodes. They **do not share memory or state variables**. 

### What is XCom?
XCom (**Cross-Communication**) allows tasks to exchange small pieces of state metadata (e.g., a partition folder path or a database primary key range limit). 

*   **Storage Warning:** XCom values are serialized into the Airflow metadata backend database. Passing large payloads (like entire dataframes or millions of rows) will fill up disk space and slow down your orchestration engine.
*   **Best Practice:** Write massive data to an intermediate object store (like AWS S3 or Google Cloud Storage) and pass only the storage URI via XCom.

---

## 5. Production Code Implementation: E-Commerce Analytics DAG

Here is a comprehensive production-grade implementation of a dynamic Airflow DAG utilizing TaskFlow API syntax, proper error handling, retry limits, and operational XCom data transitions.

```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator

# 1. Define foundational default execution arguments
default_args = {
    'owner': 'neeraj_rathore',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['neerajrathore5821@gmail.com'],
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=1)
}

# 2. Instantiate the core DAG context window
with DAG(
    dag_id='ecommerce_sales_orchestration_v1',
    default_args=default_args,
    description='Production ETL pipeline for processing daily store revenue transactions',
    start_date=datetime(2026, 7, 1),
    schedule_interval='@daily',
    catchup=False,
    tags=['core_analytics', 'sales']
) as dag:

    # Task A: Run system sanity checks via traditional BashOperator
    environment_check = BashOperator(
        task_id='verify_environment_dependencies',
        bash_command='echo "Executing pipeline run for logical date: {{ ds }}" && python3 --version'
    )

    # Task B: Extract step utilizing TaskFlow API (Returns data location to XCom automatically)
    @task(task_id='extract_raw_sales')
    def extract():
        """Simulates extracting daily transactional metrics from an external API."""
        print("Extracting relational data from upstream sales endpoints...")
        raw_storage_uri = "s3://lakehouse-raw-zone/sales/year=2026/month=07/data.json"
        return raw_storage_uri  # Pushed automatically to XCom storage layer

    # Task C: Transform step taking the output location from the extraction task
    @task(task_id='transform_sales_metrics')
    def transform(raw_uri: str):
        """Simulates reading data, applying aggregation rules, and outputting to clean staging partitions."""
        print(f"Reading raw source logs from: {raw_uri}")
        print("Executing business normalization, currency translation, and null checks...")
        staging_uri = "s3://lakehouse-staging-zone/sales/processed_metrics.parquet"
        return staging_uri

    # Task D: Load step writing out data targets
    @task(task_id='load_to_warehouse')
    def load(staging_uri: str):
        """Simulates finalizing transaction states into a decoupled warehouse environment."""
        print(f"Ingesting clean data partition from: {staging_uri}")
        print("Executing micro-partition clustering keys update on analytical tables...")
        return "SUCCESS"

    # 3. Establish clear execution path layout dependencies
    raw_path = extract()
    transformed_path = transform(raw_path)
    load_status = load(transformed_path)

    # Wire the initial infrastructure check step to run before the extraction task
    environment_check >> raw_path
```
