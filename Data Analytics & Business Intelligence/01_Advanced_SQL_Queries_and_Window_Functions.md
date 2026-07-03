# Module 01: Advanced SQL Queries and Window Functions

While basic SQL (`SELECT`, `WHERE`, `GROUP BY`) is sufficient for simple data extraction, enterprise-level Data Analytics requires advanced techniques to manipulate, rank, and analyze time-series data without exporting it to Python or Excel first. 

This module covers the core advanced SQL concepts required for deep data wrangling and Business Intelligence (BI) modeling.

---

## 1. Common Table Expressions (CTEs)

### Theory
A Common Table Expression (CTE) is a temporary, named result set that you can reference within a `SELECT`, `INSERT`, `UPDATE`, or `DELETE` statement. Think of it as a temporary table that exists only for the duration of that specific query.

Unlike nested subqueries (which get messy and hard to read), CTEs are defined at the very top of your script using the `WITH` clause, making your code modular, readable, and easier for strict engineering teams to review.

### Use Cases
* Breaking down highly complex queries into logical, step-by-step building blocks.
* Filtering data *before* joining it to large fact tables to optimize memory and processing speed.
* Performing multi-level aggregations (e.g., finding the average of a sum).

### Implementation Example: E-Commerce Sales
Let's analyze a retail dataset to find product categories that are performing above the overall company average. 

```sql
WITH CategorySales AS (
    -- Step 1: Calculate total revenue per category
    SELECT 
        category_name,
        SUM(order_amount) AS total_revenue
    FROM amazon_sales_data
    GROUP BY category_name
),
AverageSales AS (
    -- Step 2: Calculate the overall average revenue across all categories
    SELECT 
        AVG(total_revenue) AS avg_revenue
    FROM CategorySales
)
-- Step 3: Use both CTEs to find top-performing categories
SELECT 
    c.category_name,
    c.total_revenue
FROM CategorySales c
JOIN AverageSales a ON c.total_revenue > a.avg_revenue
ORDER BY c.total_revenue DESC;
```

## 2. Window Functions

### Theory
The most powerful tool in SQL analytics. A Window Function performs a calculation across a set of table rows that are somehow related to the current row. 

**How it differs from `GROUP BY`:** When you use `GROUP BY`, SQL collapses the rows into a single output row per group. When you use a Window Function, SQL performs the aggregation but **keeps the original rows intact**. 

It uses the `OVER()` clause to define the "window" of data. Inside `OVER()`, you typically use:
* `PARTITION BY`: Divides the data into partitions (like `GROUP BY`, but without collapsing).
* `ORDER BY`: Defines the logical order of rows within that partition.

### Mathematical Formulas for Window Concepts
Window functions often handle time-series mathematics natively. For instance, calculating a Simple Moving Average (SMA) over $k$ periods for a specific row $n$:

$$SMA_k = \frac{1}{k} \sum_{i=0}^{k-1} x_{n-i}$$

Calculating a Cumulative Sum (Running Total):

$$S_n = \sum_{i=1}^{n} x_i$$

### Use Cases
* **`ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`:** Creating top-N lists (e.g., Top 5 players per team).
* **`LEAD()` and `LAG()`:** Comparing current rows to previous/next rows (e.g., calculating week-over-week growth).
* **Aggregations (`SUM`, `AVG` over a window):** Calculating running totals or rolling 30-day averages.

### Implementation Example: Sports Analytics (Cricket)
Let's analyze a sports database to calculate a running total of runs scored by a player across matches in a season, and rank them within their team without losing the match-by-match breakdown.

```sql
SELECT 
    match_date,
    team_name,
    player_name,
    runs_scored,
    -- 1. Running Total: Summing runs cumulatively over time for each player
    SUM(runs_scored) OVER (
        PARTITION BY player_name 
        ORDER BY match_date
    ) AS cumulative_runs,
    
    -- 2. Ranking: Rank players within their team based on runs in this specific match
    DENSE_RANK() OVER (
        PARTITION BY team_name, match_date 
        ORDER BY runs_scored DESC
    ) AS team_rank_for_match,

    -- 3. Lag: Find out how many runs this player scored in their previous match
    LAG(runs_scored, 1, 0) OVER (
        PARTITION BY player_name 
        ORDER BY match_date
    ) AS previous_match_runs
    
FROM ipl_match_stats
WHERE season_year = 2024
ORDER BY team_name, match_date, team_rank_for_match;
```

## 3. Performance Considerations & Query Folding

When utilizing advanced SQL for Business Intelligence tools (like Power BI or Tableau), where you process this SQL matters.

Server-Side Processing: Always aim to push complex CTEs and Window Functions back to the SQL database engine (e.g., PostgreSQL, Snowflake) rather than loading raw tables and doing the math inside the BI tool's memory.

Query Folding: Writing native SQL views ensures tools like Power BI can "fold" the query, executing the heavy lifting on the database server, which prevents local Out-Of-Memory (OOM) errors on large datasets.


