# 03: Feature Stores and Training-Serving Skew

This module explores **Feature Engineering Operations, Feature Stores, and Training-Serving Skew Prevention**. It covers dual-storage architectures (Online vs. Offline stores), time-travel point-in-time joins to eliminate target leakage, Feast configuration manifests, and an end-to-end Python feature ingestion pipeline.

---

## 1. Enterprise Feature Store Architecture

In production ML pipelines, training-serving skew occurs when features used during model training differ from features computed during real-time inference. A centralized **Feature Store** acts as the single source of truth, enforcing unified feature definitions across batch training and real-time serving pipelines.

### Core Architecture Components

* **Offline Feature Store (Parquet / BigQuery):** Optimized for high-throughput, columnar batch queries used during historical model training and backtesting.
* **Online Feature Store (Redis / DynamoDB):** Optimized for low-latency ($<10\text{ms}$) point lookups by entity ID during live online model inference.
* **Point-in-Time Correctness (Time-Travel Joins):** Ensures that feature values joined to observation entity keys correspond strictly to timestamps *before* the observation event occurred, preventing **target leakage**.

---

## 2. Mathematical Modeling: Point-In-Time Joins & Skew Metrics

### 1. Point-in-Time Matrix Join Calculus
Let $E = \{(e_i, t_i)\}_{i=1}^N$ be an entity observation dataframe, where $e_i$ is the entity ID (e.g., `user_id`) and $t_i$ is the target observation timestamp. Let $F = \{(e_j, v_j, \tau_j)\}_{j=1}^M$ be a feature event record with value $v_j$ generated at timestamp $\tau_j$.

The Point-in-Time feature value $f^*(e_i, t_i)$ selected for training is computed as:

$$f^*(e_i, t_i) = \operatorname{arg\,max}_{\tau_j \le t_i} \{ \tau_j \mid e_j = e_i \}$$

$$\text{Constraint:} \quad \tau_j \le t_i \quad (\text{Prevents Future Data Leakage})$$

---

### 2. Training-Serving Skew Quantification (Population Stability Index - PSI)
To detect distribution drift between features computed at training ($P$) and features retrieved in production serving ($Q$), we calculate the Population Stability Index across $K$ probability buckets:

$$\text{PSI} = \sum_{k=1}^{K} \left( P_k - Q_k \right) \times \ln\left( \frac{P_k}{Q_k} \right)$$

* **Interpretation:** $\text{PSI} < 0.1 \implies \text{No Skew}$; $0.1 \le \text{PSI} \le 0.2 \implies \text{Moderate Skew}$; $\text{PSI} > 0.2 \implies \text{Critical Training-Serving Skew}$.

---

## 3. Feast Feature Store Definition (`feature_store.yaml` & `features.py`)

### 1. `feature_store.yaml`
```yaml
project: customer_churn_feature_store
registry: data/registry.pb
provider: local
offline_store:
  type: file
online_store:
  type: redis
  connection_string: "localhost:6379"
```

### 2. `features.py` (Feature Definitions)
```python
from datetime import timedelta
from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
    ValueType,
)
from feast.types import Float32, Int64

# Define Entity Key
user_entity = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    description="Unique identifier for customer user"
)

# Define Offline File Source (Parquet format)
user_stats_source = FileSource(
    name="user_transaction_stats_source",
    path="data/user_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define Feature View (Mapping features to Entity and Source)
user_stats_fv = FeatureView(
    name="user_transaction_stats",
    entities=[user_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="avg_transaction_amount_30d", dtype=Float32),
        Field(name="failed_transactions_7d", dtype=Int64),
        Field(name="login_frequency_30d", dtype=Int64),
    ],
    online=True,
    source=user_stats_source,
    tags={"team": "risk_and_fraud"},
)
```

## 4. Production Implementation: End-to-End Feature Store Operations Pipeline

This complete Python script implements synthetic feature generation, saves data to Parquet (Offline Store), executes Point-in-Time Joins for training data generation, materializes features into Redis (Online Store), and retrieves real-time vectors for low-latency inference.
```python
import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Configure structured enterprise logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FeatureStoreEngine")


# -------------------------------------------------------------------
# 1. Synthetic Feature Data Generator
# -------------------------------------------------------------------
def generate_synthetic_feature_data(num_users: int = 1000) -> pd.DataFrame:
    """Generates synthetic historical user feature data with timestamps."""
    logger.info(f"📊 Generating historical feature data for {num_users} users...")
    
    np.random.seed(42)
    now = datetime.utcnow()
    
    data = []
    for user_id in range(1001, 1001 + num_users):
        # Generate multiple historical observations per user
        for days_back in range(30, 0, -5):
            event_time = now - timedelta(days=days_back)
            data.append({
                "user_id": user_id,
                "event_timestamp": event_time,
                "created_timestamp": event_time,
                "avg_transaction_amount_30d": float(np.random.uniform(10.0, 500.0)),
                "failed_transactions_7d": int(np.random.randint(0, 5)),
                "login_frequency_30d": int(np.random.randint(1, 50))
            })

    df = pd.DataFrame(data)
    os.makedirs("data", exist_ok=True)
    parquet_path = "data/user_stats.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info(f"💾 Saved offline Parquet feature data to {parquet_path} ({len(df)} rows).")
    return df


# -------------------------------------------------------------------
# 2. Point-in-Time Join Logic (Simulating Offline Retrieval)
# -------------------------------------------------------------------
def simulate_point_in_time_join(
    entity_df: pd.DataFrame, feature_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Executes Point-in-Time Time-Travel Join to combine target observation
    timestamps with feature records, preventing target leakage.
    """
    logger.info("⏳ Performing Point-in-Time correctness join...")
    
    merged = pd.merge(entity_df, feature_df, on="user_id", how="inner")
    
    # Filter out any features generated AFTER the target event timestamp
    valid_features = merged[merged["event_timestamp"] <= merged["observation_timestamp"]]
    
    # Select the most recent feature record prior to observation
    idx = valid_features.groupby(["user_id", "observation_timestamp"])["event_timestamp"].idxmax()
    pit_joined_df = valid_features.loc[idx].reset_index(drop=True)
    
    logger.info(f"✅ Point-in-Time join completed: {len(pit_joined_df)} training rows returned.")
    return pit_joined_df


# -------------------------------------------------------------------
# 3. Online Store In-Memory Mock (Simulating Low-Latency Redis Fetch)
# -------------------------------------------------------------------
class MockOnlineFeatureStore:
    """Simulates an online feature store (e.g., Redis) for real-time model serving."""
    
    def __init__(self):
        self._store: Dict[int, Dict[str, Any]] = {}

    def materialize(self, df: pd.DataFrame):
        """Materializes latest feature values per entity into low-latency store."""
        logger.info("🚀 Materializing latest feature values to Online Store...")
        
        # Sort by timestamp to grab latest snapshot per entity
        latest_df = df.sort_values("event_timestamp").groupby("user_id").last().reset_index()
        
        for _, row in latest_df.iterrows():
            self._store[int(row["user_id"])] = {
                "avg_transaction_amount_30d": row["avg_transaction_amount_30d"],
                "failed_transactions_7d": row["failed_transactions_7d"],
                "login_frequency_30d": row["login_frequency_30d"]
            }
        logger.info(f"⚡ Online Store materialized with {len(self._store)} entities.")

    def get_online_features(self, entity_keys: List[int]) -> List[Dict[str, Any]]:
        """Retrieves feature vectors in real time for online model inference."""
        start_time = time.time()
        results = []
        for key in entity_keys:
            features = self._store.get(key, {})
            results.append({"user_id": key, **features})
        
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"⏱️ Online feature fetch latency: {latency_ms:.3f} ms for {len(entity_keys)} entities.")
        return results


# -------------------------------------------------------------------
# 4. Pipeline Orchestration Execution
# -------------------------------------------------------------------
def main():
    # Step 1: Generate Raw Offline Data
    feature_df = generate_synthetic_feature_data(num_users=100)

    # Step 2: Define Target Observation Entity DataFrame (Training Labels)
    now = datetime.utcnow()
    entity_df = pd.DataFrame([
        {"user_id": 1001, "observation_timestamp": now - timedelta(days=2), "churned": 1},
        {"user_id": 1002, "observation_timestamp": now - timedelta(days=10), "churned": 0},
        {"user_id": 1003, "observation_timestamp": now - timedelta(days=1), "churned": 0},
    ])

    # Step 3: Execute Point-In-Time Join for Training
    training_dataset = simulate_point_in_time_join(entity_df, feature_df)
    print("\n================ TRAINING DATASET (PIT JOIN) ================")
    print(training_dataset[["user_id", "observation_timestamp", "event_timestamp", "avg_transaction_amount_30d", "churned"]])

    # Step 4: Materialize Features to Online Store & Serve
    online_store = MockOnlineFeatureStore()
    online_store.materialize(feature_df)

    # Step 5: Real-Time Online Inference Feature Retrieval
    live_request_users = [1001, 1003]
    online_vector = online_store.get_online_features(live_request_users)
    print("\n================ ONLINE FEATURE VECTOR (INFERENCE) ================")
    print(online_vector)


if __name__ == "__main__":
    main()
```
