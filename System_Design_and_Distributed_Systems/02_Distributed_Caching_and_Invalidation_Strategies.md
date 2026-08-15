# Module 02: Distributed Caching and Invalidation Strategies

Caching is the primary mechanism for achieving low-latency sub-millisecond data access and protecting backend databases from overwhelming read loads in high-concurrency distributed systems. However, managing distributed caches introduces complex challenges: maintaining cache consistency, selecting proper write/read strategies, and mitigating failure modes such as Cache Stampede (Thundering Herd), Cache Avalanche, and Cache Penetration.

This module covers caching topologies, mathematical hit-ratio formulations, cache eviction mechanics, multi-layer invalidation patterns, and a production Redis-backed hybrid cache with probabilistic early expiration (XFetch) implemented in Python.

---

## 1. Theoretical Foundations

### 1.1 Caching Patterns & Architectural Strategies
* **Cache-Aside (Lazy Loading)**: Application reads from cache first. On cache miss, it fetches from DB, writes to cache, and returns result. (Simple, resilient to cache node failures, but potential stale data).
* **Write-Through**: Writes go directly to the Cache, which synchronously writes to the DB before returning success. (High data consistency, higher write latency).
* **Write-Behind (Write-Back)**: Writes go to the Cache and return immediately. The cache asynchronously flushes batch updates to the DB. (Ultra-fast write throughput, risk of data loss on cache node crashes).
* **Refresh-Ahead**: The cache proactively reloads frequently accessed entries before their TTL expires based on historical access patterns.

* ---

### 1.2 Mathematical Formulations

#### 1. Cache Hit Ratio & System Latency Impact
The effective mean read latency $L_{\text{effective}}$ of a system depends on Cache Hit Ratio $H \in [0, 1]$, Cache Read Latency $L_c$, and Database Read Latency $L_{db}$:

$$L_{\text{effective}} = H \cdot L_c + (1 - H) \cdot (L_c + L_{db})$$

*Example*: If $L_c = 2\text{ms}$, $L_{db} = 50\text{ms}$, and Hit Ratio $H = 95\%$ ($0.95$):

$$L_{\text{effective}} = 0.95(2) + 0.05(2 + 50) = 1.9 + 2.6 = 4.5\text{ms}$$

#### 2. Cache Stampede Mitigation (Probabilistic Early Expiration / XFetch Algorithm)
To prevent the **Thundering Herd** problem (where thousands of concurrent requests query the DB simultaneously when a hot key expires), the **XFetch** probabilistic algorithm recomputes and re-populates the cache *before* true expiration:

$$\text{Recompute Trigger}: -\beta \cdot \delta \cdot \ln(\Delta) > \text{TTL}_{\text{remaining}}$$

Where:
* $\delta$: Time taken to compute the value from the database.
* $\beta > 0$: Aggressiveness constant (default $1.0$).
* $\Delta \sim U(0, 1)$: Uniform random variable between 0 and 1.
* $\text{TTL}_{\text{remaining}}$: Remaining time until full key expiry.

---

## 2. Advanced Failure Modes & Mitigations

| Failure Mode | Description | Mitigation Strategy |
| :--- | :--- | :--- |
| **Cache Stampede** | Concurrent requests hit DB simultaneously when hot key expires. | Mutex Locks (Singleflight) / XFetch Probabilistic Expiration |
| **Cache Avalanche** | Massive number of keys expire at the exact same second. | Add Random Jitter to TTL (e.g., $\text{TTL} = \text{base} \pm \text{rand}(0, 300\text{s})$) |
| **Cache Penetration** | Queries for non-existent keys bypass cache and hit DB continuously. | Store Null Values with short TTL / Use **Bloom Filters** |
| **Cache Breakdown** | A single hot key expires under extreme traffic surge. | Background Cron Refresh / Logical Expiry without hard TTL |

---

## 3. Production Distributed Cache with XFetch & Lock Mitigation

This Python implementation provides a production-grade Cache-Aside manager using `redis-py` with **Mutex Singleflight Locks** and **Probabilistic Early Expiration (XFetch)**.

### Prerequisites

```bash
pip install redis pydantic
```

### Python Implementation (distributed_cache.py)
```python
import time
import math
import random
import json
import redis
from typing import Optional, Callable, Any

class DistributedCacheManager:
    def __init__(self, redis_client: redis.Redis, beta: float = 1.0):
        self.redis = redis_client
        self.beta = beta

    def get_with_xfetch(self, key: str, fetch_from_db: Callable[[], Any], ttl_seconds: int) -> Optional[Any]:
        """
        Retrieves a key using the XFetch probabilistic early expiration algorithm
        to prevent Thundering Herd / Cache Stampede.
        """
        raw_data = self.redis.get(key)
        
        if raw_data:
            entry = json.loads(raw_data)
            value = entry["value"]
            delta = entry["delta"]           # Time taken to compute DB query in seconds
            expiry_timestamp = entry["expiry"]
            
            ttl_remaining = expiry_timestamp - time.time()
            
            # XFetch Decision Formula: -beta * delta * ln(random(0,1)) > ttl_remaining
            random_val = random.random()
            if random_val > 0 and (-self.beta * delta * math.log(random_val)) > ttl_remaining:
                print(f"[XFETCH TRIGGERED] Early background recomputation for hot key: '{key}'")
                self._recompute_and_set(key, fetch_from_db, ttl_seconds)
            
            return value

        # Cache Miss: Acquire lock to compute value (Mutex Lock Pattern)
        lock_key = f"lock:{key}"
        acquired_lock = self.redis.set(lock_key, "locked", nx=True, ex=5)

        if acquired_lock:
            try:
                print(f"[CACHE MISS] Fetching key '{key}' from Database...")
                return self._recompute_and_set(key, fetch_from_db, ttl_seconds)
            finally:
                self.redis.delete(lock_key)
        else:
            # Wait briefly and retry reading from cache after lock holder populates it
            time.sleep(0.05)
            return self.get_with_xfetch(key, fetch_from_db, ttl_seconds)

    def _recompute_and_set(self, key: str, fetch_from_db: Callable[[], Any], ttl_seconds: int) -> Any:
        start_time = time.time()
        value = fetch_from_db()
        delta = time.time() - start_time
        
        # Add random jitter to TTL to prevent Cache Avalanche (±10% variation)
        jitter = random.randint(-int(ttl_seconds * 0.1), int(ttl_seconds * 0.1))
        effective_ttl = max(1, ttl_seconds + jitter)
        expiry_timestamp = time.time() + effective_ttl

        payload = {
            "value": value,
            "delta": delta,
            "expiry": expiry_timestamp
        }
        
        self.redis.setex(key, effective_ttl, json.dumps(payload))
        return value

# Usage Example / Simulation
if __name__ == "__main__":
    # Connect to local Redis instance (Fallback to mock object if unavailable)
    try:
        r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        r.ping()
        print("Connected to Redis successfully.")
    except redis.ConnectionError:
        print("Redis unavailable. Run `docker run -p 6379:6379 redis` to test live.")
        exit(0)

    cache = DistributedCacheManager(r)

    # Simulated expensive Database query function
    def expensive_db_query():
        time.sleep(0.2)  # Simulate 200ms DB delay
        return {"user_id": 101, "name": "Alice", "role": "Architect"}

    cache_key = "user_profile:101"

    # 1. Initial Access -> Cache Miss
    data = cache.get_with_xfetch(cache_key, expensive_db_query, ttl_seconds=10)
    print("Fetched Result 1:", data)

    # 2. Subsequent Access -> Cache Hit
    data = cache.get_with_xfetch(cache_key, expensive_db_query, ttl_seconds=10)
    print("Fetched Result 2 (Cached):", data)
```
