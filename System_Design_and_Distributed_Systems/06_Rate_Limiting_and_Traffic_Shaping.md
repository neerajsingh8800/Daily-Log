# Module 06: Rate Limiting and Traffic Shaping

Rate limiting and traffic shaping protect API infrastructure, control resource consumption, prevent Denial-of-Service (DoS) attacks, and enforce multi-tenant usage tier quotas. Rate limiters act as gatekeepers at API Gateways or reverse proxies, shedding load or queuing excess requests before backend servers become overwhelmed.

This module covers rate-limiting algorithms, mathematical models for burst capacity and sliding windows, distributed coordination strategies using Redis with atomic Lua scripts, and a production implementation in Python.

---

## 1. Theoretical Foundations

### 1.1 Algorithmic Paradigms
* **Token Bucket**: Tokens refill into a bucket at a constant rate $r$. Each request consumes 1 token. If the bucket is full, excess tokens are discarded; if empty, requests are rejected. Supports traffic bursting up to bucket capacity $B$.
* **Leaky Bucket**: Requests enter a FIFO queue (bucket) and leak out at a constant processing rate $r$. Smooths out traffic bursts into a steady output stream (Traffic Shaping). If the queue overflows, incoming requests are dropped.
* **Fixed Window Counter**: Divides time into fixed windows (e.g., 1 minute) and increments a counter per window. Simple to implement, but vulnerable to **burst at window boundary** (up to $2 \times$ the limit across boundary seconds).
* **Sliding Window Log**: Tracks timestamped logs of every request in a sorted set (e.g., Redis ZSET). Provides exact precision but requires high memory storage ($O(N)$ space per user).
* **Sliding Window Counter**: Combines Fixed Window counters by taking a weighted average of the current window and previous window counters based on time elapsed. Memory efficient ($O(1)$) with minimal estimation error.

* ---

### 1.2 Mathematical Formulations

#### 1. Token Bucket Capacity & Refill Equation
Let $T_{\text{last}}$ be the timestamp of the last request, $T_{\text{now}}$ be the current time, $B$ be max bucket capacity, and $r$ be token refill rate (tokens/second). The available tokens $Tokens_{\text{current}}$ at time $T_{\text{now}}$ is:

$$Tokens_{\text{current}} = \min \Big( B, \ Tokens_{\text{previous}} + (T_{\text{now}} - T_{\text{last}}) \cdot r \Big)$$

If $Tokens_{\text{current}} \ge 1$, the request is **ALLOWED** and tokens are updated:

$$Tokens_{\text{new}} = Tokens_{\text{current}} - 1$$

#### 2. Sliding Window Counter Approximation Math
Let $C_{\text{prev}}$ be the request count in the previous window, $C_{\text{curr}}$ be the count in the current window, $W$ be window size (seconds), and $t_{\text{elapsed}}$ be elapsed time into the current window. The estimated total requests $N_{\text{est}}$ is:

$$N_{\text{est}} = C_{\text{prev}} \cdot \left( \frac{W - t_{\text{elapsed}}}{W} \right) + C_{\text{curr}}$$

A request is allowed if $N_{\text{est}} < \text{Limit}$.

---

## 2. Rate Limiting Algorithm Comparison

| Algorithm | Memory Complexity | Concurrency / Race Safety | Burst Support | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Token Bucket** | $O(1)$ | Needs Atomic CAS / Lua Script | Yes (up to $B$) | General API rate limiting (AWS, Stripe) |
| **Leaky Bucket** | $O(B)$ | Queue-based synchronization | No (Smooths traffic) | Traffic shaping / Third-party vendor calls |
| **Fixed Window** | $O(1)$ | Redis `INCR` + `EXPIRE` | Burst at boundary | Low-precision simple rate counters |
| **Sliding Log** | $O(N)$ requests | High storage overhead | Yes | Ultra-strict security endpoints (Login/Auth) |
| **Sliding Counter** | $O(1)$ | Dual key lookup | Smooth approximation | High-throughput distributed Edge Gateways |

---

## 3. Production Distributed Rate Limiter (Redis + Lua Script)

To run rate limiting safely in a multi-instance distributed setting without race conditions (Check-Then-Set bugs), processing must occur atomically using a **Redis Lua Script**.

### Prerequisites

```bash
pip install redis fastapi uvicorn
```

### Python Implementation (rate_limiter.py)
```python
import time
import redis
from fastapi import FastAPI, Request, HTTPException, status

app = FastAPI(title="Distributed Rate Limiter Gateway")

# Redis Connection Initialization
r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# -------------------------------------------------------------------
# ATOMIC TOKEN BUCKET LUA SCRIPT FOR REDIS
# -------------------------------------------------------------------
TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local ttl = math.ceil(capacity / refill_rate)

local data = redis.call('HMGET', key, 'tokens', 'last_updated')
local tokens = tonumber(data[1])
local last_updated = tonumber(data[2])

if tokens == nil then
    tokens = capacity
    last_updated = now
else
    local delta = math.max(0, now - last_updated)
    tokens = math.min(capacity, tokens + delta * refill_rate)
    last_updated = now
end

if tokens >= requested then
    tokens = tokens - requested
    redis.call('HMSET', key, 'tokens', tokens, 'last_updated', last_updated)
    redis.call('EXPIRE', key, ttl)
    return {1, math.floor(tokens)}  -- 1 = ALLOWED
else
    redis.call('HMSET', key, 'tokens', tokens, 'last_updated', last_updated)
    redis.call('EXPIRE', key, ttl)
    return {0, math.floor(tokens)}  -- 0 = BLOCKED
end
"""

lua_sha = r.script_load(TOKEN_BUCKET_LUA)

class RateLimiter:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate

    def is_allowed(self, client_id: str, cost: int = 1) -> tuple[bool, int]:
        """
        Executes atomic token bucket check in Redis.
        Returns tuple: (is_allowed, remaining_tokens)
        """
        now = time.time()
        key = f"rate_limit:{client_id}"
        
        result = r.evalsha(
            lua_sha, 
            1, 
            key, 
            self.capacity, 
            self.refill_rate, 
            now, 
            cost
        )
        return bool(result[0]), int(result[1])

# Initialize Rate Limiter: Max Capacity = 10 tokens, Refill = 2 tokens/sec
limiter = RateLimiter(capacity=10, refill_rate=2.0)

# Middleware / Route Protection
@app.get("/api/v1/resource")
async def get_protected_resource(request: Request):
    client_ip = request.client.host if request.client else "127.0.0.1"
    
    allowed, remaining = limiter.is_allowed(client_id=client_ip, cost=1)
    
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later.",
            headers={
                "X-RateLimit-Limit": str(limiter.capacity),
                "X-RateLimit-Remaining": str(remaining),
                "Retry-After": "1"
            }
        )
        
    return {
        "status": "SUCCESS",
        "message": "Access granted to high-throughput endpoint.",
        "tokens_remaining": remaining
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 4. Operational Best Practices & Headers

### Standard Rate Limit Response Headers:

* X-RateLimit-Limit: Maximum allowed capacity in window.
* X-RateLimit-Remaining: Tokens currently left in bucket.
* X-RateLimit-Reset: Unix timestamp when bucket refills completely.
* Retry-After: Seconds client must wait before making another request (on HTTP 429).

* Graceful Degradation: On Redis failure, configure API Gateway to fail open (allow requests with local log alerts) to prevent complete service outages.
