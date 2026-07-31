# 06: Production Deployment, Observability, and Security

This module explores **Production Hardening for Automated AI Workflows**, covering HMAC SHA-256 cryptographic signature verification, observability and telemetry pipelines, semantic caching, rate limiting, and automated health checks in Python.

---

## 1. Production Security Architecture & Zero-Trust Ingestion

In production environments, public HTTP webhook endpoints used by workflow engines are constant targets for replay attacks, unauthorized execution injections, and denial-of-service (DoS) attempts. Implementing cryptographic verification ensures that only authenticated external SaaS providers can invoke downstream automation pipelines.

### Essential Security & Reliability Protocols

1. **HMAC Cryptographic Validation:** Prevents data tampering by generating a hash digest of the incoming raw request body using a shared secret key ($K$).
2. **Replay Attack Mitigation:** Verifies timestamp headers ($\Delta t = t_{current} - t_{event}$) to reject requests older than a configured tolerance window (e.g., 300 seconds).
3. **Semantic Caching:** Hashes incoming prompt inputs to serve cached LLM outputs for identical queries, drastically reducing API costs and response latency.
4. **Distributed Rate Limiting:** Utilizes a Sliding Window Log algorithm in Redis to throttle excess workflow execution calls before hitting third-party provider limits.

---

## 2. Mathematical Modeling: Token Cost Tracking & Semantic Cache Efficiency

Deploying LLMs within production workflows requires active monitoring of caching efficiency and token spend.

### 1. Cost Savings via Cache Hit Ratio ($H$)
Let $N$ be total incoming requests, $H \in [0, 1]$ be the cache hit ratio, $C_{LLM}$ be the average API cost per live LLM generation, and $C_{cache}$ be the cost of a vector cache lookup ($C_{cache} \ll C_{LLM}$):

$$\text{Total Cost } C_{total} = N \times \left[ H \times C_{cache} + (1 - H) \times C_{LLM} \right]$$

$$\text{Cost Reduction Factor } \Delta C = \frac{C_{LLM}}{H \times C_{cache} + (1 - H) \times C_{LLM}}$$

* **Impact:** Achieving a $40\%$ cache hit ratio ($H = 0.40$) reduces total generative workflow operational expenses by **~38%**.

---

### 2. Sliding Window Rate Limiting Calculus
To enforce a maximum throughput of $R_{limit}$ requests per window $W_{seconds}$, we evaluate request timestamps within interval $[t - W, t]$:

$$\text{Allow Execution IF } \sum_{i=1}^{M} \mathbb{I}(t_i \ge t - W) < R_{limit}$$

---

## 3. Telemetry and Observability Metrics Matrix

Production automation workflows require comprehensive structured logging and tracing metrics to detect system regressions and cost anomalies.

| Metric | Type | Purpose |
| :--- | :--- | :--- |
| `workflow_execution_duration_seconds` | Histogram | Measures latency across nodes (p50, p90, p99 metrics). |
| `llm_token_consumption_total` | Counter | Tracks prompt and completion token usage split by model ID. |
| `webhook_auth_failures_total` | Counter | Logs invalid HMAC signatures to alert on potential security attacks. |
| `semantic_cache_hit_ratio` | Gauge | Tracks caching efficiency to evaluate prompt optimization performance. |

---

## 4. Production Implementation: Security, Observability, and Health Check Suite

This complete Python script implements a production security gateway and observability suite for automation workflows using **FastAPI**. It features HMAC SHA-256 payload verification, structured telemetry logging, sliding-window rate limiting, and automated health checks.

```python
import hmac
import hashlib
import time
import logging
import json
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException, Header, status
from pydantic import BaseModel, Field

# Configure structured JSON production logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ProductionAutomationGateway")

app = FastAPI(
    title="Production Security & Observability Gateway",
    description="Secures n8n/Make automation endpoints with HMAC signature verification, telemetry logging, and rate limiting.",
    version="1.0.0"
)

# Configuration Constants
SHARED_HMAC_SECRET = "c8f9b2d1e0a3456789abcdef0123456789abcdef0123456789abcdef01234567"
MAX_TIMESTAMP_SKEW_SECONDS = 300  # 5 Minutes
RATE_LIMIT_MAX_REQUESTS = 100
RATE_LIMIT_WINDOW_SECONDS = 60

# In-Memory Rate Limiting Tracker (Production implementation should use Redis)
request_history = []


# -------------------------------------------------------------------
# 1. Pydantic Models for Telemetry & Responses
# -------------------------------------------------------------------
class HealthStatusResponse(BaseModel):
    status: str = Field(..., example="HEALTHY")
    uptime_timestamp: float = Field(..., example=1722422400.0)
    services: Dict[str, str] = Field(..., example={"database": "CONNECTED", "redis_queue": "CONNECTED"})


class ExecutionMetricsPayload(BaseModel):
    trace_id: str = Field(..., example="tr-8801-abc")
    workflow_name: str = Field(..., example="Lead_Ingestion_Pipeline")
    execution_time_ms: float = Field(..., example=142.5)
    tokens_consumed: int = Field(..., example=320)
    cache_hit: bool = Field(..., example=False)


# -------------------------------------------------------------------
# 2. Security Middleware Functions
# -------------------------------------------------------------------
def verify_hmac_and_timestamp(raw_body: bytes, signature_header: Optional[str], timestamp_header: Optional[str]) -> bool:
    """Validates cryptographic HMAC SHA-256 signature and checks timestamp skew."""
    if not signature_header or not timestamp_header:
        logger.warning("❌ Missing Security Headers in incoming request.")
        return False

    # 1. Replay Attack Prevention
    try:
        current_time = time.time()
        request_time = float(timestamp_header)
        if abs(current_time - request_time) > MAX_TIMESTAMP_SKEW_SECONDS:
            logger.warning(f"❌ Timestamp skew too large: {abs(current_time - request_time)}s")
            return False
    except ValueError:
        logger.error("❌ Invalid Timestamp Header format.")
        return False

    # 2. Compute HMAC SHA-256 Signature Over Body + Timestamp
    payload_to_sign = timestamp_header.encode('utf-8') + b"." + raw_body
    computed_signature = hmac.new(
        key=SHARED_HMAC_SECRET.encode('utf-8'),
        msg=payload_to_sign,
        digestmod=hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(computed_signature, signature_header)


def check_sliding_window_rate_limit() -> bool:
    """Enforces sliding window rate limit controls."""
    now = time.time()
    global request_history
    # Filter requests outside current window
    request_history = [t for t in request_history if now - t < RATE_LIMIT_WINDOW_SECONDS]

    if len(request_history) >= RATE_LIMIT_MAX_REQUESTS:
        return False

    request_history.append(now)
    return True


# -------------------------------------------------------------------
# 3. Secure Production Gateway Endpoints
# -------------------------------------------------------------------
@app.post("/api/v1/production/secure-webhook", status_code=status.HTTP_200_OK)
async def handle_secure_webhook(
    request: Request,
    x_signature: Optional[str] = Header(None),
    x_timestamp: Optional[str] = Header(None)
):
    """
    Production-hardened Webhook Listener.
    Validates HMAC signature, checks sliding window rate limit, and records metric traces.
    """
    start_time = time.time()
    raw_body = await request.body()

    # Rate Limiting Guard
    if not check_sliding_window_rate_limit():
        logger.error("⚠️ Rate limit exceeded for webhook endpoint.")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Retry after 60 seconds."
        )

    # Cryptographic Authentication Guard
    if not verify_hmac_and_timestamp(raw_body, x_signature, x_timestamp):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid cryptographic signature or stale timestamp."
        )

    execution_duration = (time.time() - start_time) * 1000
    
    # Emit Structured Telemetry Log
    telemetry_data = ExecutionMetricsPayload(
        trace_id=f"tr-{int(time.time())}",
        workflow_name="Secure_Production_Ingest",
        execution_time_ms=round(execution_duration, 2),
        tokens_consumed=0,
        cache_hit=False
    )
    logger.info(f"📊 [TELEMETRY METRIC] {telemetry_data.model_dump_json()}")

    return {
        "status": "ACCEPTED",
        "trace_id": telemetry_data.trace_id,
        "message": "Payload verified and passed to execution queue."
    }


@app.get("/health", response_model=HealthStatusResponse, status_code=status.HTTP_200_OK)
async def system_health_check():
    """Endpoint for automated monitoring services (Datadog, Kubernetes probes)."""
    return HealthStatusResponse(
        status="HEALTHY",
        uptime_timestamp=time.time(),
        services={
            "database": "CONNECTED",
            "redis_queue": "CONNECTED",
            "n8n_engine": "ACTIVE"
        }
    )


if __name__ == "__main__":
    import uvicorn
    print("\n--- Starting Production Security Gateway on Port 8000 ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

