# 02: n8n Self-Hosting and Environment Setup

This module explores **Enterprise Self-Hosting Architectures** for **n8n**, backing database persistence with **PostgreSQL**, task queuing via **Redis**, environment secret management, and production-grade deployment using **Docker Compose**.

---

## 1. Enterprise Self-Hosting Architecture

While cloud-hosted automation services incur execution costs per operation, self-hosting n8n in a Virtual Private Cloud (VPC) provides unlimited workflow executions, total data privacy compliance (GDPR/HIPAA), and direct access to internal network microservices.

### Core Infrastructure Components

*   **Main Instance:** Handles the web UI, webhook triggers, execution scheduling, and API server.
*   **PostgreSQL Backing Store:** Persists workflow definitions, user credentials (encrypted at rest), and detailed execution logs. Replacing default SQLite with PostgreSQL is mandatory for concurrent production workloads.
*   **Redis Queue Broker:** Decouples incoming webhook bursts from execution workers using Message Queuing.
*   **Worker Instances:** Scalable stateless worker containers that pull workflow jobs from Redis queues and execute code nodes asynchronously.

---

## 2. Scaling Calculus: Queue Workers & Worker Thread Allocation

To prevent incoming webhook drops during high-volume spikes, the required count of active worker containers ($W$) is modeled based on average incoming request throughput and worker job processing latency.

Let $R_{peak}$ be peak incoming requests per second, $L_{avg}$ be average workflow execution latency in seconds, and $C_{worker}$ be the concurrent thread capacity per worker container:

$$W = \left\lceil \frac{R_{peak} \times L_{avg}}{C_{worker}} \right\rceil$$

$$\text{Example: } R_{peak} = 500 \text{ req/sec}, \ L_{avg} = 0.2 \text{ sec}, \ C_{worker} = 10 \text{ threads/container}$$

$$W = \left\lceil \frac{500 \times 0.2}{10} \right\rceil = \left\lceil \frac{100}{10} \right\rceil = 10 \text{ Worker Containers}$$

---

## 3. Production Environment & Secret Management Architecture

For security, sensitive keys (database passwords, API tokens, encryption keys) must never be hardcoded into workflow JSON files.

### Critical n8n Configuration Environment Variables

| Variable Name | Required Value / Purpose | Security Level |
| :--- | :--- | :--- |
| `N8N_ENCRYPTION_KEY` | High-entropy 32+ char random string used to encrypt credentials at rest in Postgres. | 🔴 **CRITICAL** |
| `EXECUTIONS_MODE` | Set to `queue` for multi-worker production scale (defaults to `regular`). | 🟡 High |
| `N8N_HOST` | FQDN domain name (e.g., `n8n.yourcompany.com`). | 🟢 Standard |
| `WEBHOOK_URL` | Explicit public URL endpoint for incoming webhooks. | 🟢 Standard |
| `DB_TYPE` | Set to `postgresdb`. | 🟢 Standard |

---

## 4. Production Implementation: Docker Compose Infrastructure Blueprint

Here is a complete, production-grade `docker-compose.yml` deployment manifest bundling **n8n (Main Engine)**, **PostgreSQL 16**, **Redis 7**, and an **n8n Worker Node**.

```yaml
version: '3.8'

services:
  # -------------------------------------------------------------------
  # 1. Backing Database: PostgreSQL
  # -------------------------------------------------------------------
  postgres:
    image: postgres:16-alpine
    container_name: n8n_postgres
    restart: always
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-n8n_db_user}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-SecureDbPassword2026!}
      POSTGRES_DB: ${POSTGRES_DB:-n8n_production}
    volumes:
      - postgres_storage:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-n8n_db_user} -d${POSTGRES_DB:-n8n_production}"]
      interval: 10s
      timeout: 5s
      retries: 5

  # -------------------------------------------------------------------
  # 2. Queue Broker: Redis
  # -------------------------------------------------------------------
  redis:
    image: redis:7-alpine
    container_name: n8n_redis
    restart: always
    volumes:
      - redis_storage:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # -------------------------------------------------------------------
  # 3. Main Engine: n8n Web UI & Primary Process
  # -------------------------------------------------------------------
  n8n_main:
    image: docker.n8n.io/n8nio/n8n:latest
    container_name: n8n_main
    restart: always
    ports:
      - "5678:5678"
    environment:
      - N8N_HOST=${N8N_HOST:-localhost}
      - N8N_PORT=5678
      - N8N_PROTOCOL=https
      - NODE_ENV=production
      - WEBHOOK_URL=https://${N8N_HOST:-localhost}/
      
      # Persistence Configuration
      - DB_TYPE=postgresdb
      - DB_POSTGRESDB_HOST=postgres
      - DB_POSTGRESDB_PORT=5432
      - DB_POSTGRESDB_DATABASE=${POSTGRES_DB:-n8n_production}
      - DB_POSTGRESDB_USER=${POSTGRES_USER:-n8n_db_user}
      - DB_POSTGRESDB_PASSWORD=${POSTGRES_PASSWORD:-SecureDbPassword2026!}
      
      # Encryption Key for Stored Credentials
      - N8N_ENCRYPTION_KEY=${N8N_ENCRYPTION_KEY:-SuperSecretEncryptionKey_32CharsMin!}
      
      # Scale Queue Configuration
      - EXECUTIONS_MODE=queue
      - QUEUE_BULL_REDIS_HOST=redis
      - QUEUE_BULL_REDIS_PORT=6379
    volumes:
      - n8n_data:/home/node/.n8n
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

  # -------------------------------------------------------------------
  # 4. Asynchronous Queue Worker Container
  # -------------------------------------------------------------------
  n8n_worker:
    image: docker.n8n.io/n8nio/n8n:latest
    container_name: n8n_worker_1
    restart: always
    command: worker
    environment:
      - DB_TYPE=postgresdb
      - DB_POSTGRESDB_HOST=postgres
      - DB_POSTGRESDB_PORT=5432
      - DB_POSTGRESDB_DATABASE=${POSTGRES_DB:-n8n_production}
      - DB_POSTGRESDB_USER=${POSTGRES_USER:-n8n_db_user}
      - DB_POSTGRESDB_PASSWORD=${POSTGRES_PASSWORD:-SecureDbPassword2026!}
      - N8N_ENCRYPTION_KEY=${N8N_ENCRYPTION_KEY:-SuperSecretEncryptionKey_32CharsMin!}
      - EXECUTIONS_MODE=queue
      - QUEUE_BULL_REDIS_HOST=redis
      - QUEUE_BULL_REDIS_PORT=6379
    depends_on:
      - n8n_main

volumes:
  postgres_storage:
  redis_storage:
  n8n_data:
```
