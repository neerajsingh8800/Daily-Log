# 02: Experiment Tracking, DVC, and MLflow Model Registry

This module explores **Reproducible ML Engineering, Artifact Tracking, and Lifecycle Management**. It covers Data Version Control (DVC) pipelines, MLflow experiment tracking, model registry transitions, training-serving lineage hashing, and an automated Python training pipeline.

---

## 1. Enterprise Experimentation & Artifact Lineage Architecture

Traditional software version control (Git) is designed for code, not multi-gigabyte datasets, feature matrices, or binary model weights. Decoupling code versioning from data/model state tracking is critical for deterministic model reproducibility and enterprise auditability.

### Core Architecture Components

* **Data Version Control (DVC):** Tracks dataset versions, intermediate feature stores, and model artifacts using content-addressable hash pointers (`.dvc` files) stored in Git, offloading binary files to remote object storage (S3/GCS/Azure).
* **MLflow Tracking Server:** Centralized backend database (PostgreSQL) and artifact repository (S3) for logging hyperparameter configurations, evaluation metrics, confusion matrices, and model run artifacts.
* **MLflow Model Registry:** A centralized model store with explicit semantic versioning, stage transitions (`Staging`, `Production`, `Archived`), and automated approval gates.

---

## 2. Mathematical Modeling: Hash-Based Lineage & Metric Optimization

### 1. Deterministic Lineage Verification
To verify that model weights $M_t$ originate strictly from dataset $D_k$ and code state $C_i$, we define the immutable lineage hash $H_{lineage}$:

$$H_{lineage} = \text{SHA256}\left( \text{Hash}(C_i) \;\vert{}\vert{}\; \text{Hash}(D_k) \;\vert{}\vert{}\; \text{Hash}(P) \right)$$

where $P$ is the hyperparameter map dictionary and $\vert{}\vert{}$ represents string concatenation. If $H_{lineage}$ mismatches the production deployment record, execution halts to prevent training-serving skew.

---

## 3. DVC Pipeline Automation Manifest (`dvc.yaml`)

DVC orchestrates multi-stage ML pipelines, caching execution outputs based on dependency hashes (`deps`) and re-running only dirty or modified stages.

```yaml
stages:
  prepare:
    cmd: python src/prepare_data.py --input data/raw/dataset.csv --output data/processed/
    deps:
      - src/prepare_data.py
      - data/raw/dataset.csv
    outs:
      - data/processed/train.parquet
      - data/processed/test.parquet

  train:
    cmd: python src/train.py --config params.yaml
    deps:
      - src/train.py
      - data/processed/train.parquet
      - params.yaml
    params:
      - train.learning_rate
      - train.n_estimators
      - train.max_depth
    outs:
      - models/classifier.onnx
    metrics:
      - reports/metrics.json:
          cache: false
4. Production Implementation: Automated Training, Tracking, and Registration
This complete, production-grade Python script runs model training, logs parameters and metrics to an MLflow tracking server, saves model artifacts, and conditionally registers the model to the MLflow Model Registry if validation performance exceeds defined thresholds.

Python
import os
import json
import logging
import argparse
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Configure structured enterprise logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MLflowTrackingEngine")

# Environmental Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "Enterprise_Customer_Churn_Prediction"
REGISTERED_MODEL_NAME = "CustomerChurnClassifier"
ACCURACY_REGISTRATION_THRESHOLD = 0.85


# -------------------------------------------------------------------
# 1. Dataset Generation & Preprocessing Engine
# -------------------------------------------------------------------
def prepare_synthetic_data(n_samples: int = 5000, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Generates synthetic classification dataset simulating customer churn features."""
    logger.info(f"📊 Generating synthetic dataset with {n_samples} samples...")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=random_state
    )
    
    feature_names = [f"feature_{i:02d}" for i in range(20)]
    df_x = pd.DataFrame(X, columns=feature_names)
    series_y = pd.Series(y, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        df_x, series_y, test_size=0.2, random_state=random_state, stratify=series_y
    )
    return X_train, X_test, y_train, y_test


# -------------------------------------------------------------------
# 2. MLflow Experiment Execution Engine
# -------------------------------------------------------------------
def train_and_track_model(params: Dict[str, Any]):
    """Executes model training, metrics evaluation, MLflow logging, and automated model registration."""
    
    # Initialize MLflow Client Configuration
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    X_train, X_test, y_train, y_test = prepare_synthetic_data()

    logger.info("🚀 Starting MLflow Run tracking session...")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"🆔 Active MLflow Run ID: {run_id}")

        # 1. Log Hyperparameters
        mlflow.log_params(params)
        mlflow.set_tag("developer", "neerajsingh")
        mlflow.set_tag("pipeline_stage", "automated_training")

        # 2. Train Model Architecture
        logger.info("🏋️ Training RandomForestClassifier model...")
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=params["random_state"]
        )
        model.fit(X_train, y_train)

        # 3. Evaluate Model Predictions
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        metrics = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4)
        }

        # 4. Log Metrics to MLflow Tracking Server
        logger.info(f"📈 Evaluation Results: {json.dumps(metrics)}")
        mlflow.log_metrics(metrics)

        # 5. Log Model Artifact with Signature
        input_example = X_train.iloc[:5]
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        # 6. Automated Model Registry Promotion Gate
        if acc >= ACCURACY_REGISTRATION_THRESHOLD:
            logger.info(f"✅ Accuracy ({acc:.4f}) exceeded threshold ({ACCURACY_REGISTRATION_THRESHOLD}). Promoting to Registry...")
            
            # Register Model
            model_uri = f"runs:/{run_id}/model"
            mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
            logger.info(f"🏷️ Registered Model '{REGISTERED_MODEL_NAME}' Version {mv.version}")

            # Transition to Staging via MLflow Client API
            client = MlflowClient()
            client.transition_model_version_stage(
                name=REGISTERED_MODEL_NAME,
                version=mv.version,
                stage="Staging",
                archive_existing_versions=True
            )
            logger.info(f"🚀 Model Version {mv.version} successfully promoted to 'Staging'.")
        else:
            logger.warning(f"⚠️ Accuracy ({acc:.4f}) below threshold ({ACCURACY_REGISTRATION_THRESHOLD}). Skipping registration.")


# -------------------------------------------------------------------
# 3. Entrypoint Command Line Interface
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated MLflow Experiment Engine")
    parser.add_argument("--n_estimators", type=int, default=150, help="Number of trees in forest")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum depth of trees")
    parser.add_argument("--min_samples_split", type=int, default=4, help="Minimum samples required to split node")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    hyperparameters = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "random_state": args.random_state
    }

    train_and_track_model(hyperparameters)
```
