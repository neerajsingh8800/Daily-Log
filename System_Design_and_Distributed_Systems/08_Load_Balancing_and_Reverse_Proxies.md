# Module 08: Load Balancing and Reverse Proxies

Load balancers and reverse proxies act as traffic orchestrators in distributed architectures. Positioned between clients and backend application clusters, they distribute network traffic, prevent server overload, manage TLS termination, enforce health checks, and obscure backend service topologies to maximize availability, throughput, and system resilience.

This module covers OSI Layer 4 (L4) vs. Layer 7 (L7) load balancing mechanics, traffic routing algorithms, mathematical models for round-robin weighting and server saturation, active health checking strategies, and a complete L7 Reverse Proxy implementation in Python.

---

## 1. Theoretical Foundations

### 1.1 Architectural Paradigms: L4 vs. L7 Load Balancing

* **Layer 4 (L4) Load Balancing (Transport Layer)**:
  * Operates at the TCP/UDP protocol level without inspecting higher-level application payloads.
  * Directs packet streams based on IP address, TCP/UDP ports, and 5-tuple routing (`Source IP`, `Source Port`, `Destination IP`, `Destination Port`, `Protocol`).
  * Features ultra-low latency, minimal CPU overhead, and high throughput (e.g., HAProxy TCP mode, AWS NLB, IPVS). Does not terminate TLS sessions.

* **Layer 7 (L7) Load Balancing (Application Layer)**:
  * Operates at the HTTP/HTTPS/gRPC protocol level, decrypting and inspecting application payloads.
  * Makes intelligent routing decisions based on HTTP headers, URLs, cookies, request methods, or authorization tokens.
  * Supports TLS termination, path-based routing, header manipulation, and sticky sessions, but incurs higher CPU/memory overhead per request (e.g., NGINX, HAProxy HTTP mode, AWS ALB, Envoy).
 
  * ---

### 1.2 Mathematical Foundations of Traffic Allocation Algorithms

#### 1. Weighted Round-Robin (WRR) Formula
Given a set of $N$ backend servers $S = \{S_1, S_2, \dots, S_N\}$ with corresponding integer capacity weights $W = \{w_1, w_2, \dots, w_N\}$, the total system weight $W_{\text{total}}$ is:

$$W_{\text{total}} = \sum_{i=1}^{N} w_i$$

The probability $P(S_i)$ that an incoming request is routed to server $S_i$ is proportional to its weight:

$$P(S_i) = \frac{w_i}{W_{\text{total}}}$$

*Smooth Weighted Round-Robin (Nginx Algorithm)*: Tracks a running `current_weight` ($cw$) for each server. For each request:
1. $cw_i = cw_i + w_i$ for all servers.
2. Select server $S_{\text{max}}$ with the maximum $cw$.
3. Decrement selected server's weight: $cw_{\text{max}} = cw_{\text{max}} - W_{\text{total}}$.

#### 2. Server Utilization & Queueing Delay (M/M/c Model)
According to Kendall's queueing notation, for a load balancer distributing request arrival rate $\lambda$ across $c$ identical backend servers, each with processing rate $\mu$, the total server traffic intensity (utilization factor $\rho$) is:

$$\rho = \frac{\lambda}{c \cdot \mu}$$

* Stability Constraint: System queues grow infinitely if $\rho \ge 1$. Load balancing strategies must keep $\rho < 0.75$ ($75\%$ capacity utilization) to maintain low queueing latency.

---

## 2. Load Balancing Algorithms Comparison

| Algorithm | Complexity | State Tracking | Primary Advantage | Major Weakness / Risk |
| :--- | :--- | :--- | :--- | :--- |
| **Round-Robin** | $O(1)$ | Simple counter index | Equal request distribution | Ignores backend server capacity/load variations |
| **Weighted Round-Robin** | $O(N)$ | Fixed weight configuration | Accommodates heterogeneous hardware | Static weights don't adjust to runtime spikes |
| **Least Connections** | $O(N)$ | Active connection counts | Dynamic balancing for long queries | Susceptible to stampedes on newly added nodes |
| **IP / Consistent Hash** | $O(1)$ | Client IP or session key | Consistent stateful session affinity | Risk of hotspot imbalances if key distribution is skewed |

---

## 3. Production L7 Reverse Proxy Implementation

This Python script implements a production-style **Layer 7 Reverse Proxy & Load Balancer** featuring asynchronous request forwarding, dynamic Weighted Round-Robin selection, active background health checks, and dynamic circuit breaking.

### Prerequisites

```bash
pip install httpx fastapi uvicorn
```

### Python Implementation (reverse_proxy.py)
```python
import asyncio
import time
from typing import List, Dict, Optional
import httpx
from fastapi import FastAPI, Request, Response, status

app = FastAPI(title="L7 Async Reverse Proxy & Load Balancer")

# -------------------------------------------------------------------
# 1. BACKEND NODE DEFINITION & HEALTH CHECKER
# -------------------------------------------------------------------
class BackendServer:
    def __init__(self, url: str, weight: int = 1):
        self.url = url.rstrip('/')
        self.weight = weight
        self.effective_weight = weight
        self.current_weight = 0
        self.is_healthy = True
        self.active_connections = 0

class LoadBalancer:
    def __init__(self, backends: List[BackendServer]):
        self.backends = backends
        self.client = httpx.AsyncClient(timeout=5.0)

    def select_backend_swrr(self) -> Optional[BackendServer]:
        """Smooth Weighted Round-Robin (Nginx Algorithm) selection."""
        healthy_backends = [b for b in self.backends if b.is_healthy]
        if not healthy_backends:
            return None

        total_weight = 0
        best_backend: Optional[BackendServer] = None

        for backend in healthy_backends:
            backend.current_weight += backend.effective_weight
            total_weight += backend.effective_weight

            if best_backend is None or backend.current_weight > best_backend.current_weight:
                best_backend = backend

        if best_backend:
            best_backend.current_weight -= total_weight

        return best_backend

    async def health_check_loop(self):
        """Background coroutine to evaluate backend health continuously."""
        while True:
            for backend in self.backends:
                try:
                    response = await self.client.get(f"{backend.url}/health", timeout=2.0)
                    if response.status_code == 200:
                        if not backend.is_healthy:
                            print(f"[HEALTH RECOVERED] Backend '{backend.url}' is ONLINE.")
                        backend.is_healthy = True
                    else:
                        backend.is_healthy = False
                except Exception:
                    if backend.is_healthy:
                        print(f"[HEALTH FAILURE] Backend '{backend.url}' is DOWN.")
                    backend.is_healthy = False
            await asyncio.sleep(5.0)


# Initialize Backend Pool
backends_pool = [
    BackendServer("[http://127.0.0.1:8001](http://127.0.0.1:8001)", weight=3),
    BackendServer("[http://127.0.0.1:8002](http://127.0.0.1:8002)", weight=1),
]
lb = LoadBalancer(backends_pool)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(lb.health_check_loop())


# -------------------------------------------------------------------
# 2. L7 REVERSE PROXY FORWARDING ENGINE
# -------------------------------------------------------------------
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_handler(request: Request, path: str):
    backend = lb.select_backend_swrr()
    
    if not backend:
        return Response(
            content="503 Service Unavailable: All backend targets unhealthy.",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    target_url = f"{backend.url}/{path}"
    body = await request.body()
    headers = dict(request.headers)
    headers["x-forwarded-for"] = request.client.host if request.client else "127.0.0.1"
    headers["x-forwarded-proto"] = request.url.scheme

    backend.active_connections += 1
    try:
        async with httpx.AsyncClient() as client:
            proxy_res = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
                params=request.query_params,
                timeout=10.0
            )
        
        return Response(
            content=proxy_res.content,
            status_code=proxy_res.status_code,
            headers=dict(proxy_res.headers)
        )
    except httpx.RequestError as exc:
        print(f"[PROXY ERROR] Failed to connect to '{target_url}': {exc}")
        backend.is_healthy = False  # Immediate circuit breaker trip
        return Response(
            content="502 Bad Gateway: Upstream connection failure.",
            status_code=status.HTTP_502_BAD_GATEWAY
        )
    finally:
        backend.active_connections -= 1


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 4. Architectural Best Practices

* TLS Termination at Edge: Offload expensive TLS/SSL handshake decryption at the load balancer or API Gateway to free up CPU cycles on upstream application servers.
* Connection Pooling (Keep-Alive): Maintain persistent TCP connection pools between the reverse proxy and backends to eliminate latency overhead from per-request TCP handshakes.
* Passive + Active Health Checks: Combine active HTTP health ping loops with passive monitoring (tripping health status immediately on upstream 5xx gateway errors or connection drops).

