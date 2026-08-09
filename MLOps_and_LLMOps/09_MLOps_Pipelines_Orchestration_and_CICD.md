# Module 09: MLOps Pipelines, Orchestration, and CI/CD

In production machine learning systems, model development is only a fraction of the overall lifecycle. Automating the workflow from code commit to data preprocessing, distributed model training, evaluation, registration, and deployment requires robust pipeline orchestration and Continuous Integration/Continuous Deployment (CI/CD) practices designed specifically for ML (often referred to as **CT - Continuous Training**).

This module covers the operational theory, mathematical formulation of pipeline triggers, hands-on pipeline orchestration using **Apache Airflow**, and automated ML workflows using **GitHub Actions & CML (Continuous Machine Learning)**.

---

## 1. Theoretical Foundations

### 1.1 Orchestration vs. Traditional CI/CD
* **Traditional CI/CD**: Focuses on code validation, automated unit/integration testing, building artifacts (e.g., Docker images), and deploying software endpoints.
* **MLOps Pipelines & CT**: Expands traditional CI/CD to include **Data, Code, and Model Artifacts**. It continuously monitors data triggers, orchestrates multi-step DAGs (Directed Acyclic Graphs), evaluates candidate models against production baselines, and automates canary/blue-green model deployments.

### 1.2 Mathematical Formulation of Automated Retraining Triggers

In continuous training, pipelines are triggered periodically or dynamically based on data/performance metrics.

#### Performance Degradation Trigger Rule
Let $M_0$ be the baseline production model performance metric (e.g., $F_1$-score, ROC-AUC) on a validation distribution $D_{val}$, and let $\hat{M}_t$ be the calculated model metric evaluated on recent production inference logs $D_t$ at time $t$:

$$\text{Trigger Retrain} = \begin{cases} 1 & \text{if } M_0 - \hat{M}_t > \gamma \\ 0 & \text{otherwise} \end{cases}$$

Where $\gamma \in (0, 1)$ is the predefined threshold tolerance for metric decay.

#### Data Drift Retraining Threshold (PSI Trigger)
Using the Population Stability Index (PSI) over key feature distributions:

$$\text{PSI} = \sum_{i=1}^{k} \left( P_i - Q_i \right) \times \ln\left(\frac{P_i}{Q_i}\right)$$

* $\text{PSI} < 0.1$: No significant distribution change.
* $0.1 \le \text{PSI} < 0.25$: Slight shift; trigger validation warnings.
* $\text{PSI} \ge 0.25$: Significant shift; automatically trigger the training pipeline.

---

## 2. Production Pipeline Orchestration with Apache Airflow

The following DAG orchestrates an end-to-end ML workflow: data extraction, model training, evaluation check against a baseline threshold, and registration in MLflow.

### Prerequisites

```bash
pip install apache-airflow mlflow scikit-learn pandas
Airflow DAG (dags/ml_training_pipeline.py)
Python
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Default arguments for DAG
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "breast_cancer_classifier"
ACCURACY_THRESHOLD = 0.90

def extract_and_preprocess(**kwargs):
    """Extract dataset and split into train/test sets."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save temporary processing artifacts
    X_train.to_csv('/tmp/X_train.csv', index=False)
    X_test.to_csv('/tmp/X_test.csv', index=False)
    y_train.to_csv('/tmp/y_train.csv', index=False)
    y_test.to_csv('/tmp/y_test.csv', index=False)
    print("Data successfully extracted and preprocessed.")

def train_and_evaluate(**kwargs):
    """Train ML model and log artifacts to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("automated_training_pipeline")

    X_train = pd.read_csv('/tmp/X_train.csv')
    X_test = pd.read_csv('/tmp/X_test.csv')
    y_train = pd.read_csv('/tmp/y_train.csv').values.ravel()
    y_test = pd.read_csv('/tmp/y_test.csv').values.ravel()

    with mlflow.start_run() as run:
        n_estimators = 100
        max_depth = 5
        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Log parameters and metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        print(f"Model Training Completed. Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

        # Push metadata to XCom for downstream checks
        kwargs['ti'].xcom_push(key='model_accuracy', value=acc)
        kwargs['ti'].xcom_push(key='run_id', value=run.info.run_id)

def register_model_if_passed(**kwargs):
    """Register model to MLflow registry if accuracy threshold is met."""
    ti = kwargs['ti']
    acc = ti.xcom_pull(key='model_accuracy', task_ids='train_and_evaluate_task')
    run_id = ti.xcom_pull(key='run_id', task_ids='train_and_evaluate_task')

    if acc >= ACCURACY_THRESHOLD:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, MODEL_NAME)
        print(f"Model registered successfully! Version: {result.version}")
    else:
        raise ValueError(f"Model accuracy {acc:.4f} below threshold {ACCURACY_THRESHOLD}. Registration aborted.")

# Instantiate DAG
with DAG(
    'mlops_continuous_training_pipeline',
    default_args=default_args,
    description='Automated orchestration pipeline for ML retraining',
    schedule_interval='@weekly',
    catchup=False,
) as dag:

    extract_data_task = PythonOperator(
        task_id='extract_and_preprocess_task',
        python_callable=extract_and_preprocess,
    )

    train_eval_task = PythonOperator(
        task_id='train_and_evaluate_task',
        python_callable=train_and_evaluate,
    )

    register_task = PythonOperator(
        task_id='register_model_task',
        python_callable=register_model_if_passed,
    )

    notify_task = BashOperator(
        task_id='notify_success_task',
        bash_command='echo "Pipeline executed successfully and model promoted to registry."',
    )
```
 ## Define DAG Dependencies
  extract_data_task >> train_eval_task >> register_task >> notify_task
  
## 3. CI/CD & CML Integration with GitHub Actions

Automate model testing and metric reports directly inside GitHub pull requests using CML (Continuous Machine Learning).

### .github/workflows/cml_train_and_report.yml
```YAML
name: Continuous Machine Learning (CML) Pipeline

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  train-and-report:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Setup CML
        uses: iterative/setup-cml@v1

      - name: Install Dependencies
        run: |
          pip install --upgrade pip
          pip install pandas scikit-learn matplotlib seaborn cml

      - name: Train Model & Generate Report
        env:
          REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          # Run training script and generate report metrics
          python -c "
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
report = classification_report(y_test, preds)

with open('metrics.txt', 'w') as f:
    f.write('## Model Performance Metrics\n\n')
    f.write('```\n' + report + '\n```\n')

# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
"
          # Create Markdown Comment via CML
          echo "##  Automated ML Evaluation Report" > report.md
          cat metrics.txt >> report.md
          echo "### Confusion Matrix" >> report.md
          cml image send --url confusion_matrix.png >> report.md
          
          # Post PR comment
          cml comment create report.md
```
