# Database Design and Normalization

This module covers the core principles of Relational Database Management Systems (RDBMS), focusing on **OLTP vs. OLAP paradigms**, **Functional Dependencies**, and the step-by-step process of **Database Normalization (1NF to 3NF)** along with **Denormalization trade-offs**.

---

## 1. OLTP vs. OLAP Paradigms

Data engineering workflows require choosing the right architecture based on the workload type. Relational databases are optimized differently depending on whether they serve operational application needs or analytical reporting needs.

| Feature | OLTP (Online Transaction Processing) | OLAP (Online Analytical Processing) |
| :--- | :--- | :--- |
| **Primary Focus** | Operational efficiency, day-to-day transactions. | Business intelligence, analytical queries, reporting. |
| **Data Architecture**| Highly normalized (usually 3NF) to avoid redundancy. | Denormalized (Star/Snowflake schemas) for fast reads. |
| **Query Types** | Simple, fast inserts, updates, and deletes (`INSERT`, `UPDATE`). | Complex aggregation queries (`SUM`, `AVG`, `GROUP BY`). |
| **Transaction Volume**| Millions of small transactions per day. | Fewer, but highly resource-intensive query batches. |
| **Read/Write Balance**| Heavy Write + Heavy Read. | Read-Intensive (Append-only writes). |
| **Storage Engine** | Row-oriented storage (e.g., PostgreSQL, MySQL). | Column-oriented storage (e.g., Snowflake, BigQuery). |

---

## 2. Functional Dependencies & Mathematical Prerequisites

Before normalizing tables, we must define how attributes within a table relate to one another. 

### What is a Functional Dependency (FD)?
A Functional Dependency exists when the value of one attribute (or a set of attributes) uniquely determines the value of another attribute. 

We write this mathematically as:
$$X \rightarrow Y$$

*   This reads as: **"X functionally determines Y"** or **"Y is functionally dependent on X"**.
*   **X** is called the **Determinant**.
*   **Y** is called the **Dependent**.

> **Example:** In a student database, if `StudentID` uniquely identifies the student's `Email`, we write:
> $$\text{StudentID} \rightarrow \text{Email}$$

### Closure of a Set of Functional Dependencies ($F^+$)
The closure of a set of functional dependencies, denoted as $F^+$, is the complete set of all functional dependencies that can be logically inferred from the given set $F$ using **Armstrong's Axioms**.

#### Armstrong's Axioms
Let $X$, $Y$, $Z$, and $W$ be sets of attributes in a relation $R$:

1.  **Axiom of Reflexivity:** If $Y \subseteq X$, then $X \rightarrow Y$.
2.  **Axiom of Augmentation:** If $X \rightarrow Y$, then $XZ \rightarrow YZ$ for any $Z$.
3.  **Axiom of Transitivity:** If $X \rightarrow Y$ and $Y \rightarrow Z$, then $X \rightarrow Z$.

#### Secondary Rules (Derived from Axioms):
*   **Union:** If $X \rightarrow Y$ and $X \rightarrow Z$, then $X \rightarrow YZ$.
*   **Decomposition:** If $X \rightarrow YZ$, then $X \rightarrow Y$ and $X \rightarrow Z$.
*   **Pseudo-transitivity:** If $X \rightarrow Y$ and $WY \rightarrow Z$, then $WX \rightarrow Z$.

### Attribute Closure ($X^+$)
The attribute closure of a set of attributes $X$ under a set of dependencies $F$, denoted as $X^+$, is the set of all attributes that can be functionally determined by $X$.

#### Algorithm to find $X^+$:
1. Initialize $X^+ = X$.
2. Loop through each functional dependency $A \rightarrow B$ in $F$. If $A \subseteq X^+$, then add $B$ to $X^+$ ($X^+ = X^+ \cup B$).
3. Repeat step 2 until $X^+$ stops expanding.

---

## 3. Database Normalization (1NF to 3NF)

Normalization is the systematic process of organizing a database to reduce data redundancy and eliminate **data anomalies** (Insertion, Update, and Deletion anomalies).

Let’s trace the normalization process using a single, unnormalized engineering project table.

### ❌ The Unnormalized Form (UNF)
Consider a raw table tracking employees, their departments, and projects:

| Employee_ID | Employee_Name | Department_ID | Department_Name | Projects_Assigned |
| :--- | :--- | :--- | :--- | :--- |
| 101 | Alice Smith | D1 | Engineering | Analytics, Dashboard |
| 102 | Bob Jones | D2 | Marketing | Campaign_A |

*   **Problem:** The `Projects_Assigned` column contains comma-separated, non-atomic values. This violates relational database standards.

---

### 1️⃣ First Normal Form (1NF)
**Rule:** 
*   All attributes must contain **atomic (indivisible) values**.
*   Each attribute must contain only a single value from its defined domain.
*   Every row must be unique (requires a defined primary key).

#### 1NF Implementation
To convert the UNF table to 1NF, we split multi-valued entries into distinct rows:

| Employee_ID | Employee_Name | Department_ID | Department_Name | Project_Assigned |
| :--- | :--- | :--- | :--- | :--- |
| **101** | Alice Smith | D1 | Engineering | **Analytics** |
| **101** | Alice Smith | D1 | Engineering | **Dashboard** |
| **102** | Bob Jones | D2 | Marketing | **Campaign_A** |

*   **Composite Primary Key:** `(Employee_ID, Project_Assigned)`
*   **Remaining Anomalies in 1NF:**
    *   *Update Anomaly:* If the department name for `D1` changes, we must update multiple rows.
    *   *Insertion Anomaly:* We cannot add a new department if it doesn't have an employee assigned to a project yet.

---

### 2️⃣ Second Normal Form (2NF)
**Rule:**
*   The table must already be in **1NF**.
*   **No Partial Dependencies:** Every non-prime attribute must be *fully* functionally dependent on the entire primary key, not just a subset of it.

$$\text{If Composite Key is } (A, B), \text{ then } A \rightarrow C \text{ is a Partial Dependency and is prohibited.}$$

#### 2NF Implementation
In our 1NF table, the primary key is `(Employee_ID, Project_Assigned)`.
*   `Employee_Name`, `Department_ID`, and `Department_Name` depend *only* on `Employee_ID`. This is a partial dependency.

To fix this, we decompose the table into two separate relations:

#### Table A: `Employees`
*   **Primary Key:** `Employee_ID`

| Employee_ID (PK) | Employee_Name | Department_ID | Department_Name |
| :--- | :--- | :--- | :--- |
| 101 | Alice Smith | D1 | Engineering |
| 102 | Bob Jones | D2 | Marketing |

#### Table B: `Employee_Projects`
*   **Composite Primary Key:** `(Employee_ID, Project_Assigned)`

| Employee_ID (FK) | Project_Assigned |
| :--- | :--- |
| 101 | Analytics |
| 101 | Dashboard |
| 102 | Campaign_A |

---

### 3️⃣ Third Normal Form (3NF)
**Rule:**
*   The table must already be in **2NF**.
*   **No Transitive Dependencies:** Non-prime attributes must not depend on other non-prime attributes. Every non-prime attribute must depend directly on the primary key.

$$\text{If } A \rightarrow B \text{ and } B \rightarrow C, \text{ then } A \rightarrow C \text{ is a Transitive Dependency. B must be removed.}$$

#### 3NF Implementation
Look closely at Table A (`Employees`) from our 2NF step:
*   `Employee_ID` $\rightarrow$ `Department_ID`
*   `Department_ID` $\rightarrow$ `Department_Name`
*   Therefore, `Employee_ID` $\rightarrow$ `Department_Name` is a **transitive dependency**.

We extract the department metadata into its own reference table:

#### Table 1: `Employees`
*   **Primary Key:** `Employee_ID`
*   **Foreign Key:** `Department_ID` referencing `Departments(Department_ID)`

| Employee_ID (PK) | Employee_Name | Department_ID (FK) |
| :--- | :--- | :--- |
| 101 | Alice Smith | D1 |
| 102 | Bob Jones | D2 |

#### Table 2: `Departments`
*   **Primary Key:** `Department_ID`

| Department_ID (PK) | Department_Name |
| :--- | :--- |
| D1 | Engineering |
| D2 | Marketing |

#### Table 3: `Employee_Projects`
*   **Composite Primary Key:** `(Employee_ID, Project_Assigned)`

| Employee_ID (FK) | Project_Assigned |
| :--- | :--- |
| 101 | Analytics |
| 101 | Dashboard |
| 102 | Campaign_A |

**Result:** The schema is now fully structured in 3NF, minimizing operational write overhead and completely eliminating redundancy anomalies.

---

## 4. Denormalization & Trade-offs

While 3NF is critical for transaction-heavy production systems (OLTP), it can degrade reading performance when running large scale analytics pipelines (OLAP).

### What is Denormalization?
Denormalization is the deliberate process of adding redundant data back into a normalized schema to optimize read performance and accelerate analytical execution times.

### The Engineering Trade-offs

*   **Write Performance vs. Read Performance:** Normalization accelerates writes (`INSERT`/`UPDATE`) because data is modified in only one place. Denormalization accelerates reads because data is pre-joined.
*   **Computing Compute Costs:** In cloud data warehouses like BigQuery or Snowflake, computing complex multi-table `JOIN` operations across billions of rows is expensive. Denormalizing data structures into flat tables drastically minimizes cluster computation time.
*   **Storage Overhead:** Denormalization increases storage utilization because duplicate values are written out repeatedly across rows. Given modern, low-cost cloud object storage, this trade-off heavily favors optimizing for compute performance over storage space.
