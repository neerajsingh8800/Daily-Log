# Module 11: Distributed Locking and Leases

In distributed systems, executing concurrent operations safely across independent processes requires synchronization. A **Distributed Lock** ensures mutual exclusion across multiple nodes, while a **Lease** extends lock semantics with bounded time windows to prevent deadlocks caused by crashed clients or network partitions.

This module covers distributed locking paradigms (Redis Redlock, etcd leases, ZooKeeper ephemeral nodes), Fencing Tokens for split-brain safety, mathematical formulations for lease validity windows under clock drift, and a production-grade Redlock client implementation with automatic lease renewal (heartbeat watchdog) in Python.

---

## 1. Theoretical Foundations

### 1.1 Architectural Paradigms & Consensus-Backed Locks

* **Redis Redlock Algorithm**:
  * Runs across $N$ independent, non-replicated Redis master nodes (typically $N=5$).
  * A client attempts to acquire the lock across all instances sequentially using a random, globally unique lock value (token) with a Time-To-Live (TTL).
  * The lock is acquired if the client receives ACKs from a majority quorum ($Q = \lfloor N/2 \rfloor + 1$) within a total elapsed time less than the TTL.

* **etcd Leases (Raft-Based Consistency)**:
  * Uses a central consensus cluster (Raft) to issue timed leases associated with key-value entries.
  * Clients stream heartbeat RPCs to keep the lease active. If the client fails or partitions, the lease expires and etcd automatically revokes all attached keys.

* **ZooKeeper Ephemeral Sequence Nodes (Paxos/ZAB-Based)**:
  * Clients create an ephemeral node (`/lock/request_`) inside a parent znode. Ephemeral nodes auto-delete if the client's session disconnects.
  * To avoid the **Herd Effect** (where all nodes wake up simultaneously), each client watches only the immediately preceding sequential node in the lock queue (`WATCH /lock/request_(i-1)`).

### 1.2 Mathematical Foundations & Fencing Tokens

### 1. Effective Lease Validity Equation (Redlock Safety Window)
Let $T_{\text{TTL}}$ be the initial lock lease duration, $T_{\text{start}}$ be the timestamp before sending the first acquire request, and $T_{\text{end}}$ be the timestamp after receiving the majority quorum response.

Let $\Delta T_{\text{drift}}$ be the maximum expected clock drift between nodes, calculated as:

$$\Delta T_{\text{drift}} = (T_{\text{TTL}} \cdot \delta_{\text{clock}}) + \mathtt{drift\_margin}$$

Where $\delta_{\text{clock}}$ is the clock drift percentage (e.g., 0.01%). The effective time $T_{\text{valid}}$ remaining on the lease is:

$$T_{\text{valid}} = T_{\text{TTL}} - (T_{\text{end}} - T_{\text{start}}) - \Delta T_{\text{drift}}$$

A lock is considered successfully held **only** if:

$$T_{\text{valid}} > 0$$

#### 2. Fencing Tokens for Storage Protection
A distributed lock cannot guarantee mutual exclusion if a process experiences a Stop-The-World (STW) garbage collection pause or network delay that exceeds $T_{\text{valid}}$.

To ensure safety, a strong lock service issues a monotonically increasing **Fencing Token** ($k_1 < k_2 < k_3$) alongside every lock acquisition. Downstream storage engines validate token order and reject writes containing outdated tokens:

$$\text{Storage Action} = \begin{cases} \text{ACCEPT}, & \text{if } \text{Token}_{\text{incoming}} > \text{Token}_{\text{stored}} \\ \text{REJECT}, & \text{if } \text{Token}_{\text{incoming}} \le \text{Token}_{\text{stored}} \end{cases}$$

---

## 2. Distributed Lock Engines Comparison

| Dimension / Metric | Redis (Redlock) | etcd (Leases) | ZooKeeper (Ephemeral Nodes) |
| :--- | :--- | :--- | :--- |
| **Consensus Protocol** | None (Independent Masters) | Raft | ZAB (ZooKeeper Atomic Broadcast) |
| **Consistency Guarantee** | Asynchronous / Probabilistic | Linearizable | Sequential Consistency |
| **Failure Detection** | TTL Expiration | Lease Heartbeat TTL | Session Timeout / Ping Loss |
| **Lock Thundering Herd** | High (Polling overhead) | Low (Watch API) | Low (Watcher on previous znode) |
| **Primary Use Case** | Ephemeral, high-throughput caching locks | Infrastructure cluster coordination | Strict distributed leader election |

---

## 3. Production Distributed Lock Implementation (Redlock + Auto-Renew Watchdog)

This Python script implements a **Multi-Node Redlock Client** featuring atomic Lua scripts, deterministic lease duration safety checks, auto-renewing background watchdog threads, and monotonically generated fencing tokens.

### Prerequisites

```bash
pip install redis pydantic
```

### Python Implementation (distributed_lock.py)
```python
import time
import uuid
import threading
from typing import List, Optional
import redis

# -------------------------------------------------------------------
# LUA SCRIPTS FOR ATOMIC LOCK ACQUISITION & RELEASE
# -------------------------------------------------------------------
ACQUIRE_LUA = """
if redis.call('exists', KEYS[1]) == 0 then
    redis.call('hset', KEYS[1], 'owner', ARGV[1], 'fencing_token', ARGV[2])
    redis.call('pexpire', KEYS[1], ARGV[3])
    return {1, ARGV[2]}
end
return {0, 0}
"""

RELEASE_LUA = """
if redis.call('hget', KEYS[1], 'owner') == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""

RENEW_LUA = """
if redis.call('hget', KEYS[1], 'owner') == ARGV[1] then
    return redis.call('pexpire', KEYS[1], ARGV[2])
else
    return 0
end
"""

class RedlockNode:
    def __init__(self, host: str, port: int):
        self.client = redis.Redis(host=host, port=port, socket_timeout=0.1)
        self.acquire_script = self.client.register_script(ACQUIRE_LUA)
        self.release_script = self.client.register_script(RELEASE_LUA)
        self.renew_script = self.client.register_script(RENEW_LUA)

class DistributedLock:
    _global_fencing_counter = 0
    _counter_lock = threading.Lock()

    def __init__(self, nodes: List[RedlockNode], resource_name: str, ttl_ms: int = 10000):
        self.nodes = nodes
        self.resource_name = f"lock:{resource_name}"
        self.ttl_ms = ttl_ms
        self.quorum = (len(nodes) // 2) + 1
        self.owner_id = str(uuid.uuid4())
        self.fencing_token: Optional[int] = None
        self.is_locked = False
        self._watchdog_thread: Optional[threading.Thread] = None
        self._stop_watchdog = threading.Event()

    @classmethod
    def _generate_fencing_token(cls) -> int:
        with cls._counter_lock:
            cls._global_fencing_counter += 1
            return cls._global_fencing_counter

    def acquire((self)) -> bool:
        start_time = time.time() * 1000
        token = self._generate_fencing_token()
        acks = 0

        for node in self.nodes:
            try:
                res = node.acquire_script(
                    keys=[self.resource_name],
                    args=[self.owner_id, token, self.ttl_ms]
                )
                if res[0] == 1:
                    acks += 1
            except Exception:
                continue

        elapsed_time = (time.time() * 1000) - start_time
        drift = (self.ttl_ms * 0.01) + 2  # 1% drift factor + 2ms margin
        validity_time = self.ttl_ms - elapsed_time - drift

        if acks >= self.quorum and validity_time > 0:
            self.is_locked = True
            self.fencing_token = token
            self._start_watchdog()
            print(f"[LOCK ACQUIRED] Resource '{self.resource_name}' | Token={self.fencing_token} | Validity={validity_time:.2f}ms")
            return True
        else:
            self.release()
            return False

    def release(self):
        self._stop_watchdog.set()
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join()

        for node in self.nodes:
            try:
                node.release_script(keys=[self.resource_name], args=[self.owner_id])
            except Exception:
                continue
        self.is_locked = False
        print(f"[LOCK RELEASED] Resource '{self.resource_name}'")

    def _start_watchdog(self):
        """Heartbeat worker to extend lease automatically while job executes."""
        self._stop_watchdog.clear()
        renew_interval = (self.ttl_ms / 1000.0) / 3.0

        def watchdog_loop():
            while not self._stop_watchdog.wait(timeout=renew_interval):
                renewals = 0
                for node in self.nodes:
                    try:
                        if node.renew_script(keys=[self.resource_name], args=[self.owner_id, self.ttl_ms]):
                            renewals += 1
                    except Exception:
                        continue
                if renewals < self.quorum:
                    print("[WATCHDOG WARNING] Failed to renew lock lease across quorum!")

        self._watchdog_thread = threading.Thread(target=watchdog_loop, daemon=True)
        self._watchdog_thread.start()


# -------------------------------------------------------------------
# SIMULATION / VERIFICATION RUNNER
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Setup mock single local Redis node representing a 1-node cluster test
    nodes_pool = [RedlockNode("localhost", 6379)]
    
    lock_a = DistributedLock(nodes_pool, resource_name="order_payment_101", ttl_ms=6000)
    
    if lock_a.acquire():
        print(f"Executing critical business logic with Fencing Token: {lock_a.fencing_token}")
        time.sleep(2.0)  # Simulate active work (Watchdog renews lock in background)
        lock_a.release()
    else:
        print("Failed to acquire distributed lock.")
```

## 4. Operational Best Practices

* Always Enforce Fencing Tokens: Distributed locks cannot prevent delayed network packets or GC pauses from violating mutual exclusion. Downstream target systems (e.g., PostgreSQL, S3) must check monotonic fencing tokens before committing writes.
* Avoid Long-Running Lock Holds: Keep lock boundaries as brief as possible. For long background jobs, use state machines or asynchronous orchestration patterns (e.g., Sagas) instead of holding an active lock.
* Configure NTP Slewing: Configure Network Time Protocol (NTP) to adjust clock skew smoothly (slewing) rather than jumping time forward or backward, which invalidates TTL calculations.
