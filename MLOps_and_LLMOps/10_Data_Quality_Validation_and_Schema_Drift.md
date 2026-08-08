# Module 10: Data Quality, Validation, and Schema Drift

In production MLOps and LLMOps, model performance is heavily bounded by the quality of input data—a concept often summarized as **"Garbage In, Garbage Out"**. As data pipelines scale, data models evolve, and real-world distributions shift, static data quality checks are insufficient.

This module covers the core principles, mathematical frameworks, and hands-on implementations for detecting data quality issues, validating schemas, and identifying schema drift before bad data corrupts feature stores or downstream model inferences.

---

## 1. Theoretical Foundations

### 1.1 Data Quality vs. Schema Drift
* **Data Quality**: Refers to the cleanliness and sanity of data values at a specific point in time (e.g., non-null values, valid ranges, correct formatting, uniqueness constraints).
* **Schema Drift**: Occurs when the structure or contract of the incoming data changes over time without explicit downstream migration (e.g., added/deleted columns, altered data types, renamed fields, or structural JSON nesting changes).

### 1.2 Mathematical Foundations of Data Quality & Drift

To quantify data quality degradation and distribution anomalies automatically, several statistical tools are used:

#### Missingness Rate
Measures the proportion of missing or null values in a feature $X$:

$$Missingness(X) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(X_i \text{ is Null})$$

Where $\mathbb{I}(\cdot)$ is the indicator function.

#### Schema Incompatibility Score
Quantifies structural divergence between a expected baseline schema $S_{base}$ and an incoming batch schema $S_{target}$:

$$Drift_{schema} = 1 - \frac{\vert{}S_{base} \cap S_{target}\vert{}}{\vert{}S_{base} \cup S_{target}\vert{}}$$

Where $\vert{}S\vert{}$ represents the set of key-type pairs (e.g., `("age", "int64")`). A score of `0` indicates perfect alignment, while `1` signifies complete divergence.

---

## 2. Key Strategies for Handling Schema Drift

1. **Strict Validation (Fail Fast)**: Reject any batch or payload that fails schema alignment. Ideal for strict financial or medical ML systems.
2. **Schema Evolution (Permissive)**: Automatically adapt schema definitions for additive changes (e.g., new nullable columns) while alerting developers on breaking changes (e.g., type casting, deleted features).
3. **Quarantine & Dead Letter Queue (DLQ)**: Route malformed or invalid records to a secondary storage destination for manual inspection or async re-processing without breaking downstream batch processing jobs.

---

## 3. Production Implementation with Great Expectations

The following python script uses `Great Expectations` to establish baseline data assertions, build a validation pipeline, and validate new incoming batches against predefined rules.

### Prerequisites

```bash
pip install great_expectations pandas
```
## 4. Python Validation Pipeline

### validate_data.py

```python
import pandas as pd
import numpy as np
import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest

def generate_sample_datasets():
    """Generates baseline clean data and a corrupted drifted dataset."""
    
    # Baseline Clean Data
    baseline_df = pd.DataFrame({
        "user_id": np.arange(1000, 1010),
        "age": np.random.randint(18, 65, size=10),
        "account_balance": np.random.uniform(10.0, 5000.0, size=10),
        "signup_source": np.random.choice(["web", "ios", "android"], size=10)
    })
    
    # Drifted & Corrupted Incoming Data
    corrupted_df = pd.DataFrame({
        "user_id": [1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020],
        "age": [25, 30, -5, 42, 120, 31, 29, np.nan, 33, 40], # Invalid negative & extreme values, missing value
        "account_balance": [100.0, -50.0, 250.0, 0.0, 310.0, 420.0, 1200.0, 850.0, 90.0, 500.0], # Negative balance issue
        "signup_source": ["web", "ios", "smart_tv", "android", "web", "ios", "android", "web", "ios", "android"], # Unexpected categories
        "new_feature_v2": [1, 0, 1, 1, 0, 0, 1, 0, 1, 1] # Schema drift: unexpected column
    })
    
    return baseline_df, corrupted_df

def run_quality_pipeline():
    # 1. Initialize Great Expectations Context
    context = gx.get_context()

    baseline_df, corrupted_df = generate_sample_datasets()

    # 2. Create Datasource and Data Asset
    datasource = context.sources.add_pandas(name="user_data_datasource")
    data_asset = datasource.add_dataframe_asset(name="user_data_asset")

    # 3. Build Expectations Suite
    suite_name = "user_features_expectation_suite"
    suite = context.add_or_update_expectation_suite(expectation_suite_name=suite_name)

    # 4. Define Expectations (Schema & Quality Rules)
    batch_request = data_asset.build_batch_request(dataframe=baseline_df)
    validator = context.get_validator(batch_request=batch_request, expectation_suite_name=suite_name)

    # Table/Schema-level Expectations
    validator.expect_table_columns_to_match_ordered_list(
        column_list=["user_id", "age", "account_balance", "signup_source"]
    )
    
    # Column-level Expectations
    validator.expect_column_values_to_not_be_null(column="user_id")
    validator.expect_column_values_to_be_unique(column="user_id")
    
    validator.expect_column_values_to_be_between(column="age", min_value=18, max_value=100)
    validator.expect_column_values_to_not_be_null(column="age")
    
    validator.expect_column_values_to_be_between(column="account_balance", min_value=0.0, max_value=100000.0)
    
    validator.expect_column_distinct_values_to_be_in_set(
        column="signup_source", 
        value_set=["web", "ios", "android"]
    )

    validator.save_expectation_suite(discard_failed_expectations=False)
    print(" Expectation Suite successfully built and saved.")

    # 5. Run Validation against Corrupted Data
    print("\n--- Validating Corrupted Data Batch ---")
    checkpoint = context.add_or_update_checkpoint(
        name="user_data_checkpoint",
        validator=context.get_validator(
            batch_request=data_asset.build_batch_request(dataframe=corrupted_df),
            expectation_suite_name=suite_name
        )
    )
    
    validation_result = checkpoint.run()
    
    # Parse Results
    success = validation_result.list_validation_results()[0]["success"]
    results = validation_result.list_validation_results()[0]["results"]

    print(f"\nOverall Batch Validation Passed: {success}\n")
    print("Validation Failure Details:")
    for res in results:
        if not res["success"]:
            expectation_type = res["expectation_config"]["expectation_type"]
            kwargs = res["expectation_config"]["kwargs"]
            print(f" Failed Check: {expectation_type} on target {kwargs}")

if __name__ == "__main__":
    run_quality_pipeline()
```
