# Module 01: Microservices, API Gateway, and Service Mesh

Modern distributed architecture transitions applications from monolithic codebases into decoupled, independently deployable microservices. Managing communication, security, traffic routing, and resilience across dozens or hundreds of microservices requires specialized infrastructure components: **API Gateways** at the edge and **Service Meshes** within the service network.

This module covers communication protocols, edge routing mechanics, mathematical reliability formulas for distributed calls, Envoy proxy configuration, and a resilience implementation featuring Circuit Breakers and Rate Limiting using Python and FastAPI.

---

## 1. Theoretical Foundations

### 1.1 Architectural Overview & Communication Protocols
* **Microservices Patterns**: Decomposition by business capability or bounded context (Domain-Driven Design). Services communicate asynchronously (Event-Driven via Kafka/RabbitMQ) or synchronously (RPC/REST).
* **API Gateway (North-South Traffic)**: The single entry point for external clients. Handles edge concerns including TLS termination, authentication/authorization, rate limiting, request routing, and response aggregation.
* **Service Mesh (East-West Traffic)**: Dedicated infrastructure layer managing inter-service communication via sidecar proxies (e.g., Envoy). Handles dynamic service discovery, mTLS encryption, load balancing, traffic splitting, and telemetry collection without modifying application code.

* ---

### 1.2 Mathematical Formulation of Service Reliability & Circuit Breaking

#### Cascading Failure & System Availability Formula
If a composite service depends synchronously on $n$ independent microservices, each with an availability probability $A_i \in [0, 1]$, the total availability $A_{\text{total}}$ without fault tolerance mechanisms is:

$$A_{\text{total}} = \prod_{i=1}^{n} A_i$$

*Example*: If 10 dependent microservices each have $99.5\%$ ($0.995$) availability, overall system availability degrades to:

$$A_{\text{total}} = (0.995)^{10} \approx 0.9511 \quad (95.11\%)$$

#### Circuit Breaker State Transition Math
Let $E_t$ be the error rate within a sliding time window $W$ consisting of $N$ requests:

$$E_t = \frac{\sum_{i=1}^{N} \mathbb{I}(\text{Response}_i = \text{Error})}{N}$$

The Circuit Breaker transitions from **CLOSED** to **OPEN** when:

$$\text{Transition to OPEN} = \begin{cases} 1 & \text{if } N \ge N_{\text{min}} \text{ and } E_t \ge \theta_{\text{error}} \\ 0 & \text{otherwise} \end{cases}$$

Where $N_{\text{min}}$ is the minimum request volume threshold and $\theta_{\text{error}}$ is the maximum allowable error percentage (e.g., $0.50$ for $50\%$).

---

## 2. Service Mesh Envoy Proxy Configuration

Below is an production-grade Envoy configuration file (`envoy.yaml`) that acts as a sidecar proxy handling upstream routing, retries, circuit breaking, and health checks.

### `envoy.yaml`

```yaml
static_resources:
  listeners:
  - name: listener_http
    address:
      socket_address:
        address: 0.0.0.0
        port_value: 10000
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": [type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager](https://type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager)
          stat_prefix: ingress_http
          route_config:
            name: local_route
            virtual_hosts:
            - name: backend_service
              domains: ["*"]
              routes:
              - match:
                  prefix: "/api/v1/users"
                route:
                  cluster: user_service_cluster
                  retry_policy:
                    retry_on: "5xx,connect-failure,refused-stream"
                    num_retries: 3
                    per_try_timeout: 2s
          http_filters:
          - name: envoy.filters.http.router
            typed_config:
              "@type": [type.googleapis.com/envoy.extensions.filters.http.router.v3.Router](https://type.googleapis.com/envoy.extensions.filters.http.router.v3.Router)

  clusters:
  - name: user_service_cluster
    connect_timeout: 0.25s
    type: STRICT_DNS
    lb_policy: ROUND_ROBIN
    circuit_breakers:
      thresholds:
      - priority: DEFAULT
        max_connections: 1024
        max_pending_requests: 100
        max_requests: 1000
        max_retries: 3
    load_assignment:
      cluster_name: user_service_cluster
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: user-service-internal
                port_value: 8080
```

## 3. Resilient API Gateway Implementation with Circuit Breaking

This Python snippet implements an edge API Gateway microservice using FastAPI and pybreaker that routes incoming requests to downstream services with built-in circuit breaker fallback protection.

### Prerequisites
``` bash
pip install fastapi uvicorn httpx pybreaker
```

### Python Implementation (api_gateway.py)
``` python
import time
import httpx
import pybreaker
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

app = FastAPI(title="Edge API Gateway", version="1.0.0")

# 1. Define Circuit Breaker Listener for Telemetry
class CircuitBreakerLogger(pybreaker.CircuitBreakerListener):
    def state_change(self, cb, old_state, new_state):
        print(f"[CIRCUIT BREAKER ALERT] State changed from {old_state.name} to {new_state.name}")

# 2. Initialize Circuit Breaker
# Opens circuit after 3 consecutive failures; stays open for 10 seconds before half-open state
db_circuit_breaker = pybreaker.CircuitBreaker(
    fail_max=3,
    reset_timeout=10,
    listeners=[CircuitBreakerLogger()]
)

DOWNSTREAM_USER_SERVICE_URL = "http://localhost:8001/users"

class UserResponse(BaseModel):
    user_id: str
    status: str
    source: str

# 3. Protected Call Function
@db_circuit_breaker
def call_downstream_user_service(user_id: str):
    """
    Simulates or executes an HTTP call to a downstream microservice.
    Decorated with circuit breaker to trip on HTTP errors or connection timeouts.
    """
    with httpx.Client(timeout=1.0) as client:
        response = client.get(f"{DOWNSTREAM_USER_SERVICE_URL}/{user_id}")
        response.raise_for_status()
        return response.json()

# 4. Gateway Endpoint with Fallback Logic
@app.get("/api/v1/gateway/users/{user_id}", response_model=UserResponse)
async def get_user_profile(user_id: str):
    try:
        # Attempt invocation through circuit breaker
        data = call_downstream_user_service(user_id)
        return UserResponse(
            user_id=data.get("user_id", user_id),
            status="SUCCESS",
            source="Primary Downstream Microservice"
        )
    except pybreaker.CircuitBreakerError:
        # Fallback response when the downstream service is unhealthy/circuit open
        print(f"[FALLBACK TRIGGERED] Circuit OPEN for user_id={user_id}")
        return UserResponse(
            user_id=user_id,
            status="DEGRADED",
            source="API Gateway Local Fallback Cache"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Gateway Communication Error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
