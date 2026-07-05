# Module 03: Power BI Data Modeling and DAX Fundamentals

With a robust Star Schema established (Module 02), the next step in the Business Intelligence pipeline is to load the data into Power BI, define relationships, and write the analytical logic using **DAX (Data Analysis Expressions)**. 

This module covers the underlying engine of Power BI, focusing on execution contexts and the critical distinction between storing data in memory versus calculating it on the fly.

---

## 1. The Core Engine: Row Context vs. Filter Context

Understanding "Context" is the single most important concept in DAX. It determines *how* and *when* a formula evaluates data.

### Row Context
* **Theory:** This is the concept of "current row." When Power BI iterates through a table row by row, it uses Row Context. 
* **Where it exists natively:** Inside Calculated Columns and iterative functions (like `SUMX`, `AVERAGEX`).
* **Limitation:** Row context *does not* automatically filter data across relationships. 

### Filter Context
* **Theory:** This is the set of filters applied to the data model before a calculation is performed. 
* **Where it comes from:** Slicers, report filters, row/column headers in a matrix visual, or cross-filtering from clicking a chart.
* **The Magic of `CALCULATE()`:** The `CALCULATE` function is the only way in DAX to modify, override, or create new Filter Context programmatically.

---

## 2. Calculated Columns vs. Measures

This is a guaranteed interview question for any Data Analyst or BI Engineering role. You must know when to use which.

### Calculated Columns
* **How it works:** Computed row-by-row during the **Data Refresh** phase.
* **Storage:** The result is materialized and stored in the VertiPaq engine (RAM), increasing the overall file size of your `.pbix` model.
* **When to use:** Only when you need to slice, filter, or categorize data (e.g., creating an "Age Group" column like "18-24", "25-34").
* **When NOT to use:** Never use a calculated column for aggregations or math that depends on user selections.

### Measures
* **How it works:** Computed on the fly during **Query Time** (when a user clicks a slicer or loads a page).
* **Storage:** Does not consume RAM to store values; it consumes CPU power to calculate the result based on the current Filter Context.
* **When to use:** For almost all numerical aggregations (Sum, Average, Year-over-Year growth, Profit Margins).

---

## 3. Core DAX Implementation Example (E-Commerce)

Building upon the Star Schema from Module 02, here are the essential DAX patterns for an enterprise E-commerce dashboard.

## A. Base Measures (Always hide the raw fact table columns)
Instead of dragging `Fact_Sales[TotalRevenue]` onto a visual, always create an explicit measure.

```dax
-- Total Revenue Measure
Total Revenue = SUM(Fact_Sales[TotalRevenue])

-- Total Quantity Sold
Total Units Sold = SUM(Fact_Sales[Quantity])
```
## B. Iterative Functions (The 'X' Functions)

If TotalRevenue wasn't provided in the fact table, we would have to calculate it row-by-row using SUMX to multiply Quantity by UnitPrice before summing it up.

```dax
-- Evaluates row-by-row (Row Context), then sums the result (Filter Context)
Calculated Revenue = 
    SUMX(
        Fact_Sales, 
        Fact_Sales[Quantity] * Fact_Sales[UnitPrice]
    )
```
## C. Context Modification (CALCULATE)

We want to find the total revenue specifically for the "Electronics" category, ignoring user slicers.

```dax
-- Overriding the filter context to strictly calculate Electronics sales
Electronics Revenue = 
    CALCULATE(
        [Total Revenue],
        Dim_Product[Category] = "Electronics"
    )
```
## D. Time Intelligence DAX

Time Intelligence functions require a properly marked Date Dimension table (like the Dim_Date table from Module 02).

```dax
-- 1. Year-to-Date (YTD) Revenue
Revenue YTD = 
    TOTALYTD(
        [Total Revenue], 
        Dim_Date[FullDate]
    )

-- 2. Previous Year Revenue (For comparison)
Revenue Last Year = 
    CALCULATE(
        [Total Revenue],
        SAMEPERIODLASTYEAR(Dim_Date[FullDate])
    )

-- 3. Year-Over-Year (YoY) Growth Percentage
YoY Growth % = 
    DIVIDE(
        [Total Revenue] - [Revenue Last Year],
        [Revenue Last Year],
        0 -- Alternate result if divide by zero error occurs
    )
```




