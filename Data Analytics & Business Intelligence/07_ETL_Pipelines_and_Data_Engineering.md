# Module 07: ETL Pipelines and Data Engineering

Before data can be analyzed in a Power BI dashboard or fed into a predictive machine learning model, it must be acquired, cleaned, and stored. This automated movement of data is known as **Data Engineering**, and its core process is the **ETL Pipeline**.

This module covers the theory of data pipelines, data quality mathematics, and a production-grade Python implementation for moving raw data into a SQL database.

---

## 1. The ETL Framework

ETL stands for **Extract, Transform, Load**. It is the industry-standard architecture for integrating data from multiple disparate sources into a central Data Warehouse.

### A. Extract
* **Theory:** Pulling raw data from source systems. This could be querying a third-party API, scraping a website, or reading daily CSV dumps from a legacy system.
* **Key Challenge:** Doing this efficiently without crashing the source system (e.g., using incremental extraction instead of full loads).

### B. Transform
* **Theory:** The heaviest lifting in the pipeline. Raw data is messy. Transformation involves standardizing date formats, handling null values, dropping duplicates, and joining tables.
* **Key Challenge:** Ensuring data passes quality checks before it enters the warehouse.

### C. Load
* **Theory:** Writing the transformed data into the target destination, typically a structured SQL database formatted as a Star Schema (as covered in Module 02).
* **Key Challenge:** Upserting (Update + Insert) records without creating duplicates.

---

## 2. Data Quality & Pipeline Metrics

Strict engineering teams monitor their pipelines mathematically to ensure the data warehouse doesn't become a "data swamp." 

### Completeness Rate
Ensures no critical fields (like `TotalRevenue` or `CustomerID`) are dropping null values during the extraction phase.

$$
\text{Completeness} = \left( \frac{\text{Total Records} - \text{Records with NULLs}}{\text{Total Records}} \right) \times 100
$$

### Data Freshness (Latency)
Measures the time delay between when an event occurs in the real world and when it is available in the dashboard.

$$
\text{Latency} = T_{\text{available in warehouse}} - T_{\text{event creation}}
$$

---

## 3. Implementation Example: Python to SQL Pipeline

The most common stack for modern Data Engineering is using **Python (Pandas)** for the Extract and Transform phases, and **SQLAlchemy** to Load the data into a relational database.

Here is a robust script demonstrating how to extract raw daily e-commerce logs, transform them, and load them into our SQL `Fact_Sales` table.

```python
import pandas as pd
from sqlalchemy import create_engine
import datetime

# ==========================================
# 1. EXTRACT
# ==========================================
def extract_data(file_path):
    """Simulates extracting a daily batch of raw sales data."""
    print(f"Extracting data from {file_path}...")
    # In a real scenario, this could be requests.get('[https://api.vendor.com/sales](https://api.vendor.com/sales)')
    raw_data = pd.read_csv(file_path)
    return raw_data

# ==========================================
# 2. TRANSFORM
# ==========================================
def transform_data(df):
    """Cleans and formats data to match the SQL Star Schema."""
    print("Transforming data...")
    
    # Drop rows where critical revenue data is missing (Completeness Check)
    df = df.dropna(subset=['TotalRevenue', 'CustomerID'])
    
    # Standardize data types
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    
    # Create the DateKey for the Star Schema (e.g., 2024-07-09 becomes 20240709)
    df['DateKey'] = df['OrderDate'].dt.strftime('%Y%m%d').astype(int)
    
    # Clean up strings
    df['OrderNumber'] = df['OrderNumber'].str.strip().str.upper()
    
    # Keep only the columns that exist in the SQL Fact_Sales table
    columns_to_keep = ['DateKey', 'ProductKey', 'CustomerKey', 'OrderNumber', 'Quantity', 'TotalRevenue']
    clean_df = df[columns_to_keep]
    
    return clean_df

# ==========================================
# 3. LOAD
# ==========================================
def load_data(df, table_name, db_connection_string):
    """Pushes the transformed data into the SQL Data Warehouse."""
    print(f"Loading data into {table_name}...")
    
    # Create the database engine
    engine = create_engine(db_connection_string)
    
    # Load data into SQL. 'append' adds to existing data.
    # In enterprise pipelines, we would use 'upsert' logic here.
    try:
        df.to_sql(name=table_name, con=engine, if_exists='append', index=False)
        print("Pipeline execution successful!")
    except Exception as e:
        print(f"Load failed: {e}")

# ==========================================
# PIPELINE EXECUTION
# ==========================================
if __name__ == "__main__":
    # Define parameters
    SOURCE_FILE = 'raw_daily_sales.csv'
    TARGET_TABLE = 'Fact_Sales'
    DB_STRING = 'postgresql://username:password@localhost:5432/analytics_db'
    
    # Run the ETL workflow
    raw_df = extract_data(SOURCE_FILE)
    clean_df = transform_data(raw_df)
    load_data(clean_df, TARGET_TABLE, DB_STRING)
```
