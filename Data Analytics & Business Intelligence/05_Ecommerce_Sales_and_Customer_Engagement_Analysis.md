# Module 05: E-Commerce Sales and Customer Engagement Analysis

Data engineering and schema design are only valuable if they lead to actionable business insights. In the retail and e-commerce sector, companies survive not just by tracking *what* was sold, but by understanding *who* is buying and *how often* they return.

This module covers the theoretical foundations of Customer Engagement Analytics, key mathematical formulas for retail KPIs, and a practical implementation using SQL and Power BI DAX.

---

## 1. Core Retail Metrics and Formulas

To build an effective executive dashboard, you must move beyond basic revenue tracking and calculate metrics that indicate business health and customer loyalty.

### A. Average Order Value (AOV)
* **Theory:** Measures the average amount spent each time a customer places an order.
* **Use Case:** Used to evaluate pricing strategies, cross-selling effectiveness, and free-shipping thresholds.
* **Formula:**

$$
AOV = \frac{\text{Total Revenue}}{\text{Total Number of Orders}}
$$

### B. Purchase Frequency (PF)
* **Theory:** Measures how often an average customer makes a purchase over a specific time period (usually a year).
* **Use Case:** Identifies if a product is a one-time purchase or drives habitual buying.
* **Formula:**

$$
PF = \frac{\text{Total Number of Orders}}{\text{Total Unique Customers}}
$$

### C. Customer Lifetime Value (CLV)
* **Theory:** The total revenue a business can reasonably expect from a single customer account throughout the business relationship.
* **Use Case:** CLV is the ultimate metric for marketing teams. If your CLV is $150, you know you can safely spend up to $40 to acquire a new customer (Customer Acquisition Cost) and still remain highly profitable.
* **Formula:** (Simplified Historical CLV)

$$
CLV = AOV \times PF \times \text{Average Customer Lifespan}
$$

---

## 2. Cohort Analysis (Understanding Retention)

**Theory:** A "vanity metric" might show that total sales are going up every month. However, if you are losing 80% of your old customers and only growing because of expensive marketing to *new* customers, the business is failing. 

**Cohort Analysis** solves this by grouping customers based on their first purchase date (e.g., "The January 2024 Cohort") and tracking their specific repeat purchase behavior over subsequent months.

* **High Retention:** Indicates strong product-market fit and good customer service.
* **Sharp Drop-off (Churn):** Indicates poor product quality, bad shipping experiences, or lack of engagement marketing.

---

## 3. Implementation Example: Amazon Sales Dataset

Here is how you implement these advanced metrics using a combination of SQL for data prep and DAX for dynamic dashboard calculations.

### Step 1: SQL Data Prep (Flagging First-Time vs. Repeat Customers)
Before loading the data into Power BI, it is highly efficient to use SQL Window Functions to determine a customer's first purchase date.

```sql
WITH CustomerFirstPurchase AS (
    SELECT 
        CustomerKey,
        MIN(DateKey) AS FirstPurchaseDateKey
    FROM Fact_Sales
    GROUP BY CustomerKey
)
SELECT 
    fs.SalesKey,
    fs.CustomerKey,
    fs.DateKey,
    fs.TotalRevenue,
    CASE 
        WHEN fs.DateKey = cfp.FirstPurchaseDateKey THEN 'New Customer'
        ELSE 'Repeat Customer'
    END AS CustomerType
FROM Fact_Sales fs
JOIN CustomerFirstPurchase cfp ON fs.CustomerKey = cfp.CustomerKey;
```
### 3. Step 2: Dynamic DAX Measures in Power BI

Once the Star Schema is loaded into Power BI, we create explicit measures so the executive team can slice these KPIs by Product Category or Region.
```dax
-- 1. Average Order Value (AOV)
AOV = 
    DIVIDE(
        SUM(Fact_Sales[TotalRevenue]), 
        DISTINCTCOUNT(Fact_Sales[OrderNumber]), 
        0
    )

-- 2. Total Unique Customers
Unique Customers = DISTINCTCOUNT(Fact_Sales[CustomerKey])

-- 3. Repeat Customer Rate
-- Calculates the percentage of total customers who have made more than one purchase
Repeat Customer Rate = 
    VAR CustomersWithMultipleOrders = 
        CALCULATE(
            [Unique Customers],
            FILTER(
                Dim_Customer,
                CALCULATE(DISTINCTCOUNT(Fact_Sales[OrderNumber])) > 1
            )
        )
    RETURN 
        DIVIDE(CustomersWithMultipleOrders, [Unique Customers], 0)
```
## 4. Dashboard Design & Visualization Strategy

When building the final Power BI report for E-commerce data, follow these enterprise design principles:

Top-Level KPIs (The "Glance" View): Place standard KPI cards at the very top (Total Revenue, AOV, YoY Growth %). Executives should understand business health within 5 seconds.

Trend Lines over Time: Use Line Charts with a continuous Date axis to show Revenue and Active Customers. Always include a trend line or a moving average to smooth out daily volatility.

Pareto Analysis (80/20 Rule): Use a Bar Chart to visualize revenue by Product Category or City. Usually, 80% of sales come from 20% of the catalog. Identifying these top performers is critical for inventory management.

Tooltips: Keep the main canvas clean. Embed secondary metrics (like total units sold or average discount applied) inside custom report page tooltips that appear only when the user hovers over a specific product.


