# Module 02: Database Schema Design and Normalization

Before a single dashboard is built or a machine learning model is trained, data must be structured logically. Poor database design leads to slow queries, data anomalies (like updating a customer's address in one place but not another), and bloated storage costs.

This module covers the core principles of structuring relational databases: **Normalization** for transactional systems (OLTP) and **Dimensional Modeling** for analytical systems (OLAP).

---

## 1. Database Normalization (The OLTP Approach)

Normalization is the process of organizing data to minimize redundancy and eliminate data anomalies (Insert, Update, and Delete anomalies). It is primarily used for **Online Transaction Processing (OLTP)** databases—the backend systems that run live applications.

The rules of normalization are divided into "Normal Forms."

### First Normal Form (1NF)
* **Rule:** Every column must contain atomic (indivisible) values. No repeating groups or arrays.
* **Bad:** A `skills` column containing `"Python, SQL, PowerBI"`.
* **Good:** Separate rows or a junction table for each skill.

### Second Normal Form (2NF)
* **Rule:** Must be in 1NF, and all non-key attributes must be fully functionally dependent on the primary key. (No partial dependencies).
* **Example:** If your primary key is a combination of `(OrderID, ProductID)`, a column like `ProductName` shouldn't be in this table because it only depends on `ProductID`, not the whole key. `ProductName` should move to a separate `Products` table.

### Third Normal Form (3NF)
* **Rule:** Must be in 2NF, and there must be no transitive dependencies. (A non-key attribute cannot depend on another non-key attribute).
* **Example:** A `Customers` table has `ZipCode` and `City`. `City` actually depends on `ZipCode`, not the Customer ID. To achieve 3NF, move `ZipCode` and `City` into a separate `Locations` table.

---

## 2. Dimensional Modeling (The OLAP Approach)

While 3NF is great for application backends (because writes/updates are fast and safe), it is terrible for Data Analytics. Joining 15 highly normalized tables together to calculate total daily revenue will crash your BI tool. 

For **Online Analytical Processing (OLAP)** and BI tools like Power BI, we use Dimensional Modeling. 

### The Star Schema
The industry standard for data warehousing. It consists of two types of tables:
1. **Fact Tables:** The center of the star. They record business events (e.g., a sale, a click, a match played). They contain **quantitative metrics** (price, quantity, runs scored) and foreign keys.
2. **Dimension Tables:** The points of the star. They contain **descriptive attributes** (customer name, product category, date, venue).

**Why it wins for Analytics:** It minimizes `JOIN` operations. You only ever need one join to connect a Fact to a Dimension, making reads incredibly fast.

### The Snowflake Schema
A variation of the Star Schema where the Dimension tables are themselves normalized. 
* **Trade-off:** It saves storage space but requires more complex joins.
* **Storage Math:** The storage size $S$ of a schema can be modeled as:
  $$S = (N_{\text{facts}} \cdot C_{\text{fact}}) + \sum_{i=1}^{k} (N_{\text{dim}_i} \cdot C_{\text{dim}_i})$$
  *(Where $N$ is row count and $C$ is average row byte size. Snowflaking reduces the $C_{\text{dim}}$ factor by stripping redundant strings, but modern data warehouses prefer the compute speed of a Star schema over the storage savings of a Snowflake).*

---

## 3. Implementation Example: E-Commerce Star Schema

To prepare an Amazon-style sales dataset for a high-performance Power BI dashboard, we must transform the raw, flat data into a Star Schema. 

Here is the Data Definition Language (DDL) to create a robust Star Schema architecture.

```sql
-- ==========================================
-- DIMENSION TABLES (Descriptive context)
-- ==========================================

-- 1. Date Dimension (Critical for Time Intelligence DAX in Power BI)
CREATE TABLE Dim_Date (
    DateKey INT PRIMARY KEY,           -- e.g., 20240704
    FullDate DATE NOT NULL,
    Year INT NOT NULL,
    Quarter INT NOT NULL,
    Month INT NOT NULL,
    MonthName VARCHAR(20) NOT NULL,
    DayOfWeek VARCHAR(20) NOT NULL,
    IsWeekend BOOLEAN NOT NULL
);

-- 2. Product Dimension
CREATE TABLE Dim_Product (
    ProductKey INT PRIMARY KEY IDENTITY(1,1), -- Surrogate Key
    ProductSKU VARCHAR(50) UNIQUE NOT NULL,   -- Business Key
    ProductName VARCHAR(255) NOT NULL,
    Category VARCHAR(100) NOT NULL,
    SubCategory VARCHAR(100),
    UnitCost DECIMAL(10, 2)
);

-- 3. Customer Dimension
CREATE TABLE Dim_Customer (
    CustomerKey INT PRIMARY KEY IDENTITY(1,1),
    CustomerID VARCHAR(50) UNIQUE NOT NULL,
    CustomerName VARCHAR(150) NOT NULL,
    Email VARCHAR(150),
    City VARCHAR(100),
    Country VARCHAR(100)
);

-- ==========================================
-- FACT TABLE (The measurable events)
-- ==========================================

CREATE TABLE Fact_Sales (
    SalesKey INT PRIMARY KEY IDENTITY(1,1),
    
    -- Foreign Keys linking to Dimensions
    DateKey INT NOT NULL,
    ProductKey INT NOT NULL,
    CustomerKey INT NOT NULL,
    
    -- Degenerate Dimension (Used for grouping, but no separate table needed)
    OrderNumber VARCHAR(50) NOT NULL,
    
    -- Measures (The quantitative data)
    Quantity INT NOT NULL,
    UnitPrice DECIMAL(10, 2) NOT NULL,
    DiscountAmount DECIMAL(10, 2) DEFAULT 0.00,
    TotalRevenue DECIMAL(12, 2) NOT NULL,
    
    -- Establishing Relationships
    FOREIGN KEY (DateKey) REFERENCES Dim_Date(DateKey),
    FOREIGN KEY (ProductKey) REFERENCES Dim_Product(ProductKey),
    FOREIGN KEY (CustomerKey) REFERENCES Dim_Customer(CustomerKey)
);

-- ==========================================
-- INDEXING FOR ANALYTICS
-- ==========================================
-- Create a non-clustered index on the foreign keys in the fact table 
-- to drastically speed up JOIN operations during dashboard rendering.
CREATE NONCLUSTERED INDEX IX_FactSales_DateKey ON Fact_Sales(DateKey);
CREATE NONCLUSTERED INDEX IX_FactSales_ProductKey ON Fact_Sales(ProductKey);
CREATE NONCLUSTERED INDEX IX_FactSales_CustomerKey ON Fact_Sales(CustomerKey);
```
