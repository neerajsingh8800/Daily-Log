# 12: Data Quality and Testing Frameworks

This module explores **Data Quality Engineering**, the core pillars of data observability, statistical anomaly detection, mathematical profiling bounds, and programmatic test implementations using **Great Expectations** and **Soda Core**.

---

## 1. The Core Pillars of Data Quality

In production data engineering, garbage data ingested upstream leads to invalid dashboards and corrupted ML models downstream ("Garbage In, Garbage Out"). Data Quality Engineering enforces automated assertions at ingestion boundaries.

| Pillar | Definition | Failure Scenario |
| :--- | :--- | :--- |
| **Accuracy** | Data correctly reflects real-world events or ground truth entities. | Negative values in `transaction_amount` or invalid email strings. |
| **Completeness** | No missing, unexpectedly null, or truncated records across expected rows. | Drop in daily row counts from 100k to 2k due to an upstream API outage. |
| **Timeliness** | Data arrives within defined latency boundaries (SLA). | Daily batch pipeline finishes at 10 AM instead of 6 AM before business hours. |
| **Validity** | Attributes conform strictly to domain formats, types, and value sets. | `order_status` receiving `'CANCELLED_BY_USER'` when schema expects `'cancelled'`. |
| **Uniqueness** | Records contain no unexpected duplicates across primary key attributes. | Primary key `user_id` appears multiple times in a dimension table. |

---

## 2. Mathematical Modeling: Anomaly Detection via Z-Score Bounds

To automatically detect volumetric drops or unexpected metric spikes without hardcoding static thresholds, engineers apply statistical process control using the **Z-Score Metric**.

Let $x_t$ be the incoming daily row volume or metric value at time $t$. We compute the rolling mean ($\mu$) and standard deviation ($\sigma$) over a historical window $W$ (e.g., $W = 30$ days):

$$\mu = \frac{1}{W} \sum_{i=1}^{W} x_{t-i}$$

$$\sigma = \sqrt{\frac{1}{W} \sum_{i=1}^{W} (x_{t-i} - \mu)^2}$$

The Z-Score ($Z$) measures how many standard deviations the current data point $x_t$ deviates from the historical norm:

$$Z = \frac{x_t - \mu}{\sigma}$$

$$\text{Data Quality Rule:} \quad \text{If } \vert{}Z\vert{} > 3.0 \implies \text{Flag Volumetric Anomaly \& Halt Pipeline}$$

*   **Z-Score $> 3.0$:** Indicates an unexpected spike or severe volume anomaly ($>99.73\%$ confidence bound under a normal distribution).

---

## 3. Data Profiling Calculus: Null Rate & Uniqueness Ratio

Before defining expectations, automated profilers compute structural dataset health metrics:

### 1. Null Value Rate ($NR$)
$$NR(A) = \frac{\text{Count of NULL records in Column } A}{\text{Total Row Count } N}$$

### 2. Uniqueness Ratio ($UR$)
$$UR(A) = \frac{\text{Count of Distinct Values in Column } A}{\text{Total Row Count } N}$$

$$\text{For Primary Key Attributes:} \quad UR(A) = 1.0 \quad \text{and} \quad NR(A) = 0.0$$

---

## 4. Production Testing Blueprint: Great Expectations & Soda Core

Here is a complete, production-grade Python script demonstrating how to run automated data quality assertions against a PySpark/Pandas DataFrame using **Great Expectations** and validate a schema configuration using **Soda Core** principles.

```python
import json
import pandas as pd
import numpy as np
import great_expectations as ge

def run_data_quality_suite():
    print("--- 1. Generating Mock Production Batch ---")
    # Generating sample transactional batch with intentional anomalies for testing
    raw_data = {
        "transaction_id": ["TXN-1001", "TXN-1002", "TXN-1003", "TXN-1004", "TXN-1004"], # Duplicate ID
        "user_id": [501, 502, 503, 504, 505],
        "amount": [150.00, 200.50, -50.00, 89.90, 1200.00],  # Negative amount anomaly
        "payment_method": ["credit_card", "upi", "net_banking", "upi", "invalid_vendor"], # Unaccepted value
        "created_at": ["2026-07-22", "2026-07-22", "2026-07-22", "2026-07-22", "2026-07-22"]
    }
    
    df = pd.DataFrame(raw_data)
    
    # Wrap Pandas DataFrame in Great Expectations dataset wrapper
    ge_df = ge.from_pandas(df)

    print("\n--- 2. Defining Data Quality Assertions ---")
    
    # Assertion 1: Primary key uniqueness
    ge_df.expect_column_values_to_be_unique(column="transaction_id")

    # Assertion 2: Mandatory non-null values
    ge_df.expect_column_values_to_not_be_null(column="user_id")

    # Assertion 3: Validity range checks (amount must be positive)
    ge_df.expect_column_values_to_be_between(
        column="amount", 
        min_value=0.00, 
        max_value=10000.00
    )

    # Assertion 4: Allowed categorical value sets
    ge_df.expect_column_values_to_be_in_set(
        column="payment_method",
        value_set=["credit_card", "debit_card", "upi", "net_banking"]
    )

    print("\n--- 3. Executing Validation Suite ---")
    validation_results = ge_df.validate()

    # Parse and print overall status
    success = validation_results["success"]
    print(f"Overall Data Quality Validation Suite Status: {'SUCCESS' if success else 'FAILED'}")

    print("\n--- 4. Detailed Failed Expectations Summary ---")
    for result in validation_results["results"]:
        if not result["success"]:
            expectation_type = result["expectation_config"]["expectation_type"]
            column = result["expectation_config"]["kwargs"].get("column")
            unexpected_count = result["result"].get("unexpected_count")
            unexpected_list = result["result"].get("unexpected_list")
            
            print(f"❌ Failed Rule: {expectation_type} on column '{column}'")
            print(f"   Unexpected Record Count: {unexpected_count}")
            print(f"   Violating Samples: {unexpected_list}\n")

def generate_soda_cl_yaml():
    """Generates a SodaCL declarative YAML quality checks blueprint."""
    soda_yaml = """
# SodaCL (Soda Check Language) Declarative Configuration
table_name: fct_daily_transactions
metrics:
  - row_count
  - missing_count

checks:
  # Volume anomaly checks
  - row_count > 0
  - row_count change avg last 7 days < 20%

  # Column-level validations
  - missing_count(transaction_id) = 0
  - duplicate_count(transaction_id) = 0
  - min(amount) >= 0.00
  - invalid_count(payment_method) = 0:
      valid values: ['credit_card', 'debit_card', 'upi', 'net_banking']
"""
    print("--- 5. Declarative SodaCL Quality Blueprint ---")
    print(soda_yaml)

if __name__ == "__main__":
    run_data_quality_suite()
    generate_soda_cl_yaml()
```
