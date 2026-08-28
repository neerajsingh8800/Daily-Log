# Module 14: Resilience Engineering

Distributed systems are inherently subject to partial failures, network latency spikes, and transient resource exhaustion. Without defensive patterns, localized failures cascade across microservice topologies, causing widespread system outages.

This module covers core resilience engineering patterns: **Circuit Breakers** (State machine transitions, sliding window failure rates), **Bulkheads** (Thread pool and concurrency isolation), and **Exponential Backoff with Jitter** (Preventing thundering herd problems), complete with mathematical models and a production-grade resilience engine implementation in Python.

---

## 1. Theoretical Foundations

### 1.1 Core Resilience Paradigms

* **Circuit Breaker Pattern**:
  * Acts as an automated electrical circuit switch that monitors outbound service calls.
  * Transitions through three states:
    * **Closed**: Requests pass through normally. Successes and failures are tracked over a sliding time window.
    * **Open**: Requests fail fast immediately without executing the downstream network call, preventing resource drain when a dependency is unhealthy.
    * **Half-Open**: After a reset timeout, a limited trial batch of requests is allowed through to evaluate downstream recovery.

* **Bulkhead Isolation Pattern**:
  * Isolates critical resources into isolated pools (e.g., dedicated thread pools or concurrency semaphores per downstream dependency).
  * Prevents a single degraded downstream service from exhausting all worker threads or memory across the entire host instance.

* **Exponential Backoff with Jitter**:
  * Retries transiently failed operations with exponentially increasing delay intervals.
  * Introduces randomized **Jitter** to desynchronize concurrent client retries and resolve **Thundering Herd** alignment spikes on recovering downstream databases or services.
 
  * ---

## 1.2 Mathematical Foundations

### 1. Exponential Backoff with Full Jitter Formula
For a retry attempt $n \in \{1, 2, \dots, N_{\text{max}}\}$, a base backoff delay $T_{\text{base}}$, and a maximum delay ceiling $T_{\text{max}}$:

The deterministic exponential delay $T_{\text{exp}}$ is:

$$T_{\text{exp}}(n) = \min\left(T_{\text{max}}, \; T_{\text{base}} \cdot 2^{n-1}\right)$$

Applying **Full Jitter** selects a random uniform delay $T_{\text{wait}}$ from the range $[0, T_{\text{exp}}(n)]$:

$$T_{\text{wait}}(n) = \text{Uniform}\left(0, \; \min\left(T_{\text{max}}, \; T_{\text{base}} \cdot 2^{n-1}\right)\right)$$

Applying **Equal Jitter** balances deterministic backoff with randomness:

$$T_{\mathtt{wait\_equal}}(n) = \frac{T_{\text{exp}}(n)}{2} + \text{Uniform}\left(0, \; \frac{T_{\text{exp}}(n)}{2}\right)$$

### 2. Sliding Window Failure Rate Evaluation
For a rolling window of $N_{\text{total}}$ recent executions containing $N_{\text{failures}}$ failed executions:

$$\text{Failure Rate } R_{\text{fail}} = \frac{N_{\text{failures}}}{N_{\text{total}}}$$

The Circuit Breaker transitions from **Closed** to **Open** if:

$$N_{\text{total}} \ge N_{\mathtt{min\_requests}} \quad \text{AND} \quad R_{\text{fail}} \ge \Theta_{\text{threshold}}$$

Where $N_{\mathtt{min\_requests}}$ is the minimum execution volume required before evaluating the failure rate threshold $\Theta_{\text{threshold}} \in (0, 1]$.

---

## 2. Resilience Strategies Comparison

| Strategy | Primary Risk Prevented | Resource Impact | Degradation Behavior |
| :--- | :--- | :--- | :--- |
| **Circuit Breaker** | Cascading failure & resource lockup | Low (In-memory counters) | Fail-Fast (Immediate fallback) |
| **Bulkhead** | Dependency starvation / Thread exhaustion | Medium (Isolated thread pools/semaphores)| Sheds load for specific dependency |
| **Exponential Backoff + Jitter** | Thundering herd retry storms | Low (CPU sleep intervals) | Gradual backoff with randomized delay |

---

## 3. Production Resilience Engine Implementation

This Python script implements an integrated resilience framework containing a **State-Machine Circuit Breaker**, a **Semaphore Bulkhead**, and an **Exponential Backoff with Full Jitter** retry decorator.

### Prerequisites

```bash
pip install asyncio
```

### Python Implementation (resilience_engine.py)
```python
import time
import random
import asyncio
import threading
from enum import Enum
from typing import Callable, Any, Optional, Dict
from collections import deque


class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpenException(Exception):
    """Raised when a request is rejected because the Circuit Breaker is OPEN."""
    pass


class BulkheadFullException(Exception):
    """Raised when a call is shed because the Bulkhead capacity is exhausted."""
    pass


# -------------------------------------------------------------------
# 1. CIRCUIT BREAKER WITH SLIDING WINDOW & HALF-OPEN TRIALS
# -------------------------------------------------------------------
class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: float = 0.5,
        recovery_time_sec: float = 3.0,
        sliding_window_size: int = 10,
        min_requests: int = 5,
        half_open_trials: int = 2
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_time_sec = recovery_time_sec
        self.sliding_window_size = sliding_window_size
        self.min_requests = min_requests
        self.half_open_trials = half_open_trials

        self.state = CircuitState.CLOSED
        self.window = deque(maxlen=sliding_window_size)
        self.last_state_change = time.time()
        self.half_open_successes = 0
        self.lock = threading.Lock()

    def allow_execution(self) -> bool:
        with self.lock:
            now = time.time()
            if self.state == CircuitState.OPEN:
                if now - self.last_state_change >= self.recovery_time_sec:
                    self._transition_to(CircuitState.HALF_OPEN)
                    return True
                return False
            return True

    def record_result(self, success: bool):
        with self.lock:
            self.window.append(1 if success else 0)
            if self.state == CircuitState.HALF_OPEN:
                if success:
                    self.half_open_successes += 1
                    if self.half_open_successes >= self.half_open_trials:
                        self._transition_to(CircuitState.CLOSED)
                else:
                    self._transition_to(CircuitState.OPEN)

            elif self.state == CircuitState.CLOSED:
                if len(self.window) >= self.min_requests:
                    failures = self.window.count(0)
                    failure_rate = failures / len(self.window)
                    if failure_rate >= self.failure_threshold:
                        self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState):
        print(f"[CIRCUIT BREAKER '{self.name}'] State Change: {self.state.value} -> {new_state.value}")
        self.state = new_state
        self.last_state_change = time.time()
        if new_state == CircuitState.HALF_OPEN:
            self.half_open_successes = 0
        elif new_state == CircuitState.CLOSED:
            self.window.clear()


# -------------------------------------------------------------------
# 2. BULKHEAD ISOLATION (CONCURRENCY LIMITER)
# -------------------------------------------------------------------
class Bulkhead:
    def __init__(self, name: str, max_concurrent_calls: int):
        self.name = name
        self.semaphore = asyncio.Semaphore(max_concurrent_calls)

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        if self.semaphore.locked():
            print(f"[BULKHEAD '{self.name}'] Capacity limit reached! Shedding load.")
            raise BulkheadFullException(f"Bulkhead '{self.name}' max capacity reached.")
        
        async with self.semaphore:
            return await func(*args, **kwargs)


# -------------------------------------------------------------------
# 3. EXPONENTIAL BACKOFF WITH FULL JITTER RETRY DECORATOR
# -------------------------------------------------------------------
def with_exponential_backoff_jitter(
    max_retries: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 2.0
):
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise e
                    
                    # Compute Exponential Backoff with Full Jitter
                    exp_delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    jittered_delay = random.uniform(0, exp_delay)
                    
                    print(f"[RETRY WARNING] Attempt {attempt} failed: {e}. Retrying in {jittered_delay:.3f}s...")
                    await asyncio.sleep(jittered_delay)
        return wrapper
    return decorator


# -------------------------------------------------------------------
# SIMULATION / VERIFICATION RUNNER
# -------------------------------------------------------------------
cb = CircuitBreaker("Payment-API", failure_threshold=0.5, recovery_time_sec=1.0, min_requests=4)
bh = Bulkhead("Payment-API-Bulkhead", max_concurrent_calls=2)


@with_exponential_backoff_jitter(max_retries=3, base_delay=0.05, max_delay=0.5)
async def external_rpc_call(should_fail: bool):
    if not cb.allow_execution():
        raise CircuitBreakerOpenException("Circuit Breaker is OPEN. Immediate fail-fast.")
    
    if should_fail:
        cb.record_result(success=False)
        raise RuntimeError("Remote service returned 503 Service Unavailable")
    
    cb.record_result(success=True)
    return "200 OK - Payment Success"


async def main():
    print("--- 1. Simulating Normal Successful Requests ---")
    for _ in range(3):
        res = await bh.execute(external_rpc_call, should_fail=False)
        print(f"Result: {res}")

    print("\n--- 2. Inducing Failures to Trip Circuit Breaker ---")
    for i in range(4):
        try:
            await bh.execute(external_rpc_call, should_fail=True)
        except Exception as err:
            print(f"Call {i+1} caught exception: {err}")

    print(f"\nCurrent Circuit State: {cb.state.value}")

    print("\n--- 3. Attempting Request while Circuit Breaker is OPEN ---")
    try:
        await bh.execute(external_rpc_call, should_fail=False)
    except Exception as err:
        print(f"Immediate Fail-Fast Triggered: {err}")

    print("\n--- 4. Waiting for Circuit Breaker Recovery Period ---")
    await asyncio.sleep(1.2)

    print("\n--- 5. Half-Open Probe Attempt ---")
    res = await bh.execute(external_rpc_call, should_fail=False)
    print(f"Trial Request Result: {res}")
    print(f"Circuit State after successful trial: {cb.state.value}")


if __name__ == "__main__":
    asyncio.run(main())
```

## 4. Operational Best Practices

* Always Combine Patterns: Use Bulkheads to contain failures, Circuit Breakers to prevent resource consumption during prolonged outages, and Exponential Backoff with Jitter for transient errors.
* Tune Sliding Window Sizes: Avoid setting sliding window sizes too small ($N < 10$), which leads to hyper-sensitive state flapping, or too large, which delays fault detection.
* Fallbacks Over Hard Failures: Provide meaningful fallback mechanisms (e.g., serving cached responses, degraded default payloads, or asynchronous queueing) when a Circuit Breaker opens or a Bulkhead sheds load.
