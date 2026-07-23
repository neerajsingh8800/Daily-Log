# 13: DataOps, CI/CD, and Infrastructure as Code (IaC)

This module explores **DataOps methodologies**, pipeline containerization with **Docker**, automating deployment workflows via **GitHub Actions**, and declaratively provisioning cloud data infrastructure using **Terraform (IaC)**.

---

## 1. The DataOps Philosophy: Software Engineering Applied to Data

**DataOps** is an automated, process-oriented methodology designed to reduce the cycle time of data analytics while ensuring strict data quality and governance. It bridges the gap between Data Engineers, Analytics Engineers, and DevOps practices.

### Core Comparison: DevOps vs. DataOps

| Evaluation Metric | Traditional Software DevOps | Modern DataOps |
| :--- | :--- | :--- |
| **Primary Asset** | Executable application binary code. | Code logic + Unbounded dynamic raw data feeds. |
| **Testing Scope** | Unit and Integration code logic checks. | Code testing + Data quality/schema validation tests. |
| **State Persistence**| Stateless microservices / APIs. | Stateful data warehouses, object stores, and schemas. |
| **Failure Modes** | Code bugs, API timeouts, memory leaks. | Schema drift, data quality corruption, volumetric spikes. |

---

## 2. Infrastructure as Code (IaC) Mechanics: Declarative State Tracking

Instead of manually clicking through cloud web consoles to create storage buckets or database instances, **Infrastructure as Code (IaC)** allows engineers to define entire data platforms using declarative configuration files.

### Declarative vs. Imperative IaC
*   **Imperative (e.g., AWS CLI / Bash):** Defines *how* to achieve a state step-by-step (e.g., "Create a bucket, then change permissions"). Prone to drift and manual execution errors.
*   **Declarative (e.g., Terraform):** Defines *what* the final target state should be. The engine computes the difference between current state (`terraform.tfstate`) and desired state, applying only necessary mutations.

### The Immutable Infrastructure Paradigm
In DataOps, cloud resources are treated as immutable. Rather than SSHing into a live worker node to update PySpark versions or dependencies, engineers re-provision identical infrastructure containers from updated configuration files.

---

## 3. Mathematical Modeling: Continuous Delivery Cycle Time & Deployment Risk

To evaluate the health of a DataOps pipeline, teams track the **Deployment Risk Index ($DRI$)** based on pipeline change velocity and test coverage.

Let $N_{changed}$ be the number of altered pipeline lines/files in a release commit, $T_{coverage}$ be the automated test suite coverage ratio ($0.0 \le T_{coverage} \le 1.0$), and $D_{schema}$ be a binary indicator of breaking schema migrations ($D_{schema} \in \{1, 2\}$):

$$DRI = \frac{N_{changed} \times D_{schema}}{\max(T_{coverage}, \ 0.1)}$$

*   **Low $DRI$ (Safe Release):** Small, frequent modular commits ($N_{changed} \to \text{low}$) backed by high test coverage ($T_{coverage} \ge 0.85$).
*   **High $DRI$ (High Failure Risk):** Monolithic PRs touching multiple schemas without automated quality checks.

---

## 4. Production IaC Implementation: Terraform AWS Data Lake Stack

Here is a complete, production-ready Terraform configuration block (`main.tf`) that declaratively provisions an S3 Data Lake, configures lifecycle rules, and sets up a staging environment IAM policy.

```hcl
# 1. Provider Configuration
terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# 2. Variables Definition
variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "environment" {
  type    = string
  default = "production"
}

# 3. Provision S3 Data Lake Raw Storage Bucket
resource "aws_s3_bucket" "data_lake_raw" {
  bucket        = "lakehouse-raw-zone-${var.environment}-01"
  force_destroy = false

  tags = {
    Environment = var.environment
    ManagedBy   = "Terraform"
    Layer       = "Raw"
  }
}

# 4. Enable Bucket Versioning for Data Governance
resource "aws_s3_bucket_versioning" "raw_versioning" {
  bucket = aws_s3_bucket.data_lake_raw.id
  versioning_configuration {
    status = "Enabled"
  }
}

# 5. Lifecycle Management Rule (Transition to Glacier after 90 days)
resource "aws_s3_bucket_lifecycle_configuration" "raw_lifecycle" {
  bucket = aws_s3_bucket.data_lake_raw.id

  rule {
    id     = "archive_old_raw_partitions"
    status = "Enabled"

    transition {
      days          = 90
      storage_class = "GLACIER"
    }
  }
}

# 6. Outputs
output "raw_bucket_arn" {
  description = "ARN of the provisioned raw data lake bucket"
  value       = aws_s3_bucket.data_lake_raw.arn
}
```
