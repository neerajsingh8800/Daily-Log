# Database Design, Normalization, and Dimensional Modeling

This module covers the core principles of relational database design (OLTP), the transition to analytical data warehousing structures (OLAP), and the mechanisms for handling historical data changes.

---

## Part 1: Database Design and Normalization

### 1. OLTP vs. OLAP Paradigms

| Feature | OLTP (Online Transaction Processing) | OLAP (Online Analytical Processing) |
| :--- | :--- | :--- |
| **Primary Focus** | Operational efficiency, day-to-day transactions. | Business intelligence, analytical queries, reporting. |
| **Data Architecture**| Highly normalized (usually 3NF) to avoid redundancy. | Denormalized (Star/Snowflake schemas) for fast reads. |
| **Query Types** | Simple, fast inserts, updates, and deletes (`INSERT`, `UPDATE`). | Complex aggregation queries (`SUM`, `AVG`, `GROUP BY`). |
| **Transaction Volume**| Millions of small transactions per day. | Fewer, but highly resource-intensive query batches. |
| **Read/Write Balance**| Heavy Write + Heavy Read. | Read-Intensive (Append-only writes). |
| **Storage Engine** | Row-oriented storage (e.g., PostgreSQL, MySQL). | Column-oriented storage (e.g., Snowflake, BigQuery). |

---

### 2. Functional Dependencies & Mathematical Prerequisites

#### What is a Functional Dependency (FD)?
A Functional Dependency exists when the value of one attribute (or a set of attributes) uniquely determines the value of another attribute. 

We write this mathematically as:
$$X \rightarrow Y$$

*   **X** is called the **Determinant**.
*   **Y** is called the **Dependent**.

> **Example:** In a student database, if `StudentID` uniquely identifies the student's `Email`, we write:
> $$\text{StudentID} \rightarrow \text{Email}$$

#### Armstrong's Axioms
Let $X$, $Y$, $Z$, and $W$ be sets of attributes in a relation $R$:
1.  **Axiom of Reflexivity:** If $Y \subseteq X$, then $X \rightarrow Y$.
2.  **Axiom of Augmentation:** If $X \rightarrow Y$, then $XZ \rightarrow YZ$ for any $Z$.
3.  **Axiom of Transitivity:** If $X \rightarrow Y$ and $Y \rightarrow Z$, then $X \rightarrow Z$.

#### Attribute Closure ($X^+$)
The attribute closure of a set of attributes $X$ under a set of dependencies $F$, denoted as $X^+$, is the set of all attributes that can be functionally determined by $X$.
1. Initialize $X^+ = X$.
2. Loop through each functional dependency $A \rightarrow B$ in $F$. If $A \subseteq X^+$, then add $B$ to $X^+$ ($X^+ = X^+ \cup B$).
3. Repeat until $X^+$ stops expanding.

---

### 3. Database Normalization (1NF to 3NF)

#### ❌ The Unnormalized Form (UNF)
| Employee_ID | Employee_Name | Department_ID | Department_Name | Projects_Assigned |
| :--- | :--- | :--- | :--- | :--- |
| 101 | Alice Smith | D1 | Engineering | Analytics, Dashboard |
| 102 | Bob Jones | D2 | Marketing | Campaign_A |

#### 1️⃣ First Normal Form (1NF)
**Rule:** All attributes must contain **atomic (indivisible) values**. Every row must be unique.
*   **Composite Primary Key:** `(Employee_ID, Project_Assigned)`

| Employee_ID | Employee_Name | Department_ID | Department_Name | Project_Assigned |
| :--- | :--- | :--- | :--- | :--- |
| **101** | Alice Smith | D1 | Engineering | **Analytics** |
| **101** | Alice Smith | D1 | Engineering | **Dashboard** |
| **102** | Bob Jones | D2 | Marketing | **Campaign_A** |

#### 2️⃣ Second Normal Form (2NF)
**Rule:** Must be in **1NF** and contain **No Partial Dependencies** (Every non-prime attribute must depend on the *entire* composite primary key).

##### Table A: `Employees` (PK: `Employee_ID`)
| Employee_ID (PK) | Employee_Name | Department_ID | Department_Name |
| :--- | :--- | :--- | :--- |
| 101 | Alice Smith | D1 | Engineering |
| 102 | Bob Jones | D2 | Marketing |

##### Table B: `Employee_Projects` (Composite PK: `(Employee_ID, Project_Assigned)`)
| Employee_ID (FK) | Project_Assigned |
| :--- | :--- |
| 101 | Analytics |
| 101 | Dashboard |
| 102 | Campaign_A |

#### 3️⃣ Third Normal Form (3NF)
**Rule:** Must be in **2NF** and contain **No Transitive Dependencies** (Non-prime attributes must not depend on other non-prime attributes).

##### Table 1: `Employees` (PK: `Employee_ID`, FK: `Department_ID`)
| Employee_ID (PK) | Employee_Name | Department_ID (FK) |
| :--- | :--- | :--- |
| 101 | Alice Smith | D1 |
| 102 | Bob Jones | D2 |

##### Table 2: `Departments` (PK: `Department_ID`)
| Department_ID (PK) | Department_Name |
| :--- | :--- |
| D1 | Engineering |
| D2 | Marketing |

##### Table 3: `Employee_Projects` (Composite PK: `(Employee_ID, Project_Assigned)`)
| Employee_ID (FK) | Project_Assigned |
| :--- | :--- |
| 101 | Analytics |
| 101 | Dashboard |
| 102 | Campaign_A |

---

### 4. Denormalization Trade-offs
Denormalization is the deliberate process of adding redundant data back into a normalized schema to optimize read performance and accelerate analytical execution times in OLAP environments.

*   **Write Performance vs. Read Performance:** Normalization accelerates writes because data is modified in one place. Denormalization accelerates reads because data is pre-joined.
*   **Storage Overhead:** Denormalization increases storage utilization because duplicate values are written out repeatedly across rows. Given modern, low-cost cloud object storage, this trade-off heavily favors optimizing for compute performance over storage space.

---
---

## Part 2: Dimensional Modeling (Star vs. Snowflake)

### 1. Core Concepts of Dimensional Modeling

#### Fact Tables
The quantitative center of the schema containing the measurable, numerical metrics of a business process.
*   **Keys:** Composed of foreign keys pointing to surrounding dimensions (Composite PK).
*   **Example Metrics:** `quantity_sold`, `gross_revenue`.

#### Dimension Tables
The qualitative context surrounding the facts. They answer the "who, what, where, when, and why" of the business events.
*   **Keys:** Contain a Primary Key (often a Surrogate Key) referenced by the fact table.
*   **Example Attributes:** `customer_name`, `store_city`.

---

### 2. Star Schema vs. Snowflake Schema

#### 🏢 The Star Schema
The central fact table is connected directly to denormalized dimension tables.
*   **Normalization Level:** Denormalized (Dimensions are kept flat in 2NF/1NF).
*   **Join Complexity:** Low. Requires only a single-level join from the fact table to any dimension table.
*   **Query Performance:** Excellent for reads, as it minimizes compute overhead from multi-table joins.

#### ❄️ The Snowflake Schema
Dimension tables are broken down further and normalized into separate tables.
*   **Normalization Level:** Normalized (Dimensions split further into 3NF).
*   **Join Complexity:** High. Requires multi-layered nested joins.
*   **Query Performance:** Slower analytical reads because database engines must execute complex relational paths across billions of rows.

---

### 3. Practical E-Commerce Design Architecture

#### The Central Fact Table: `fact_sales`
```sql
CREATE TABLE fact_sales (
    sales_key INT PRIMARY KEY,       -- Surrogate Primary Key
    date_key INT NOT NULL,           -- FK to dim_date
    customer_key INT NOT NULL,       -- FK to dim_customer
    product_key INT NOT NULL,        -- FK to dim_product
    store_location_key INT NOT NULL, -- FK to dim_location
    quantity_sold INT,
    gross_revenue DECIMAL(12, 2),
    discount_applied DECIMAL(5, 2)
);
```

## Star Schema Design (Denormalized Dimensions)
```sql
CREATE TABLE dim_location_star (
    location_key INT PRIMARY KEY,
    store_id VARCHAR(50),
    city_name VARCHAR(100),       
    state_name VARCHAR(100),      
    country_name VARCHAR(100)     
);
```

## Snowflake Schema Design (Normalized Dimensions)
```sql
CREATE TABLE dim_location_snowflake (
    location_key INT PRIMARY KEY,
    store_id VARCHAR(50),
    city_key INT                  -- FK pointing to dim_city
);

CREATE TABLE dim_city (
    city_key INT PRIMARY KEY,
    city_name VARCHAR(100),
    state_key INT                 -- FK pointing to dim_state
);

CREATE TABLE dim_state (
    state_key INT PRIMARY KEY,
    state_name VARCHAR(100),
    country_name VARCHAR(100)
);
```

## 4. Architectural Evaluation Metrics

### 1. Schema Space Efficiency Ratio ($SSER$)

$$SSER = \frac{\text{Storage Footprint of Star Schema Layout}}{\text{Storage Footprint of Snowflake Schema Layout}}$$

### 2. Analytical Execution Complexity Index ($AECI$)

$$AECI = \text{Average Number of Joins Per Business Query}$$


