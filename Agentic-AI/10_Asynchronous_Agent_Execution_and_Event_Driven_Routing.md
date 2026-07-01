# Module 10: Asynchronous Agent Execution and Event-Driven Routing

Enterprise-grade AI agents rarely operate inside synchronous loops. Real-world tasks involve waiting for slow LLM token generation, multi-second tool executions, third-party API webhooks, or explicit human-in-the-loop approvals. Synchronous designs block the execution thread, destroying scalability and spiking infrastructure costs.

This module covers the principles of asynchronous agent task management, mathematical modeling of queue capacities, and event-driven architectures built to handle high-concurrency workflows without breaking.

---

## 1. Concurrency Theory: Sync vs. Async vs. Distributed Event Loops

Understanding the core plumbing of runtime execution models is crucial for architecting low-latency agent networks.

### The Asynchronous Event Loop
In a synchronous system, a thread blocks while waiting for an external I/O operation (like an API call to an LLM) to complete. In an asynchronous event loop (e.g., Python's `asyncio`), when a task hits an `await` statement, it relinquishes control back to the loop. The loop handles other execution tasks while waiting for the underlying system socket to signal that data is ready.
### Amdahl's Law in Multi-Agent Workflows
When dealing with multi-agent orchestration (e.g., parallel worker execution), the theoretical speedup of executing multiple sub-agents concurrently is restricted by the serial portion of the program (like sequential orchestration steps or final summarizations). 

$$S_{\text{latency}} = \frac{1}{(1 - P) + \frac{P}{N}}$$

Where:
* $S_{\text{latency}}$ is the theoretical execution speedup.
* $P$ is the proportion of the agent execution workflow that can be made parallel.
* $1 - P$ is the serial execution bottleneck (e.g., sequentially validating guardrails).
* $N$ is the number of parallel processing workers available.

---

## 2. Queueing Theory & Traffic Management

When building an event-driven router where incoming user prompts are transformed into discrete tasks in a queue, we model the system as an **M/M/1 Queue** (Poisson arrivals, Exponential service times, single routing queue) or **M/M/c Queue** ($c$ parallel execution workers) to prevent buffer overflows and message drops.

### Traffic Intensity ($\rho$)
The stability of your agent processing queue is determined by the ratio of the arrival rate ($\lambda$) to the service rate ($\mu$):

$$\rho = \frac{\lambda}{c \cdot \mu}$$

* If $\rho \ge 1$, the queue will grow infinitely, eventually causing out-of-memory errors (OOM) or massive system dropouts.
* For stable, reliable agent execution, system targets should maintain $\rho < 0.8$.

### Average Waiting Time ($W_q$)
In a stable single-worker queue system ($\text{M/M/1}$), the average time an execution task sits dormant in the queue before an agent starts picking it up is modeled as:

$$W_q = \frac{\lambda}{\mu(\mu - \lambda)}$$

---

## 3. Production Implementation

The following implementation showcases a production-ready asynchronous agent orchestrator. It uses `asyncio` to execute tools concurrently, leverages an in-memory `asyncio.Queue` to mimic a distributed broker (like RabbitMQ or Redis), and routes events based on dynamic task definitions.

```python
import asyncio
import time
import random
import uuid
from typing import Dict, Any, List, Optional

# ==========================================
# Event Schemas & Task Definitions
# ==========================================
class AgentEvent:
    def __init__(self, event_type: str, payload: Dict[str, Any]):
        self.event_id: str = str(uuid.uuid4())
        self.event_type: str = event_type
        self.payload: Dict[str, Any] = payload
        self.timestamp: float = time.time()

class AsynchronousAgentRouter:
    def __init__(self, max_concurrent_workers: int = 3):
        # The central ingestion queue for all raw agent execution tasks
        self.task_queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self.semaphore = asyncio.Semaphore(max_concurrent_workers)
        self.is_running = True

    async def emit_event(self, event_type: str, payload: Dict[str, Any]):
        """Ingests an incoming execution task or routing event into the queue."""
        event = AgentEvent(event_type, payload)
        await self.task_queue.put(event)
        print(f"[+] Event Ingested | ID: {event.event_id[:8]} | Type: {event_type}")

    async def _execute_rag_tool(self, task_id: str, query: str) -> str:
        """Simulates an asynchronous, heavy I/O vector database retrieval operation."""
        print(f"   [Tool-Start] Processing RAG lookup for Task: {task_id}")
        # Yields control back to the event loop, allowing other tasks to run simultaneously
        await asyncio.sleep(1.5) 
        return f"Relevant documents context for query '{query}' retrieved successfully."

    async def _execute_web_scrape_tool(self, task_id: str, url: str) -> str:
        """Simulates an asynchronous network operation scraping a remote page."""
        print(f"   [Tool-Start] Scraping live web endpoints for Task: {task_id}")
        await asyncio.sleep(2.0)
        return f"Live snapshot data from {url} parsed into JSON layout."

    async def process_agent_workflow(self, event: AgentEvent):
        """
        Manages individual agent reasoning execution state. 
        Uses a bounded semaphore to prevent API rate-limiting or compute starvation.
        """
        async with self.semaphore:
            task_id = event.event_id[:8]
            payload = event.payload
            event_type = event.event_type
            
            print(f"\n[*] Worker Processing Task: {task_id} | Type: {event_type}")
            start_time = time.time()

            try:
                if event_type == "KNOWLEDGE_RETRIEVAL":
                    # Parallelizing downstream sub-tasks if needed
                    query = payload.get("query", "")
                    rag_task = asyncio.create_task(self._execute_rag_tool(task_id, query))
                    
                    # Simulating an inline async LLM classification call occurring simultaneously
                    await asyncio.sleep(0.5) 
                    
                    # Await the execution result of the I/O tool bound task
                    context_result = await rag_task
                    print(f"   [Worker-Done] Task {task_id} compiled context payload.")
                    
                elif event_type == "WEB_ANALYSIS":
                    url = payload.get("url", "")
                    scrape_result = await self._execute_web_scrape_tool(task_id, url)
                    print(f"   [Worker-Done] Task {task_id} extracted target data structure.")
                    
                else:
                    print(f"   [Worker-Warning] Unknown routing key for Task: {task_id}")

            except Exception as e:
                print(f"   [-] System Failure Processing Task {task_id}: {str(e)}")
            
            elapsed = time.time() - start_time
            print(f"[✓] Task {task_id} Lifecycle Finalized in {elapsed:.2f} seconds.\n")

    async def start_event_loop_listener(self):
        """Infinite worker daemon that continuously pops and schedules jobs from the queue."""
        print("[*] Event-Driven Agent Router Daemon Started. Awaiting signals...")
        
        while self.is_running or not self.task_queue.empty():
            try:
                # Wait for the next event matching standard timeouts
                event = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Schedule the task execution concurrently without blocking the next queue pop
                asyncio.create_task(self.process_agent_workflow(event))
                
                # Notify the queue that the item is popped and processing has kicked off
                self.task_queue.task_done()
            except asyncio.TimeoutError:
                # Safe fallback if queue is temporarily empty during idle spikes
                continue

    def stop_router(self):
        self.is_running = False

# ==========================================
# Concurrency Verification Flow
# ==========================================
async def main():
    router = AsynchronousAgentRouter(max_concurrent_workers=2)

    # Launching the background consumer loop daemon
    listener_task = asyncio.create_task(router.start_event_loop_listener())

    # Simulating sudden bursts of concurrent user requests (Poisson arrival style spike)
    print("\n--- Simulating High Concurrency Event Ingestion ---")
    await router.emit_event("KNOWLEDGE_RETRIEVAL", {"query": "Explain quantum vector states."})
    await router.emit_event("WEB_ANALYSIS", {"url": "[https://arxiv.org/abs/agent-frameworks](https://arxiv.org/abs/agent-frameworks)"})
    await router.emit_event("KNOWLEDGE_RETRIEVAL", {"query": "Fetch active relational table layouts."})
    
    # Allow workers some runtime execution clearance
    await asyncio.sleep(4.0)
    
    print("--- Simulating Secondary Ingestion Burst ---")
    await router.emit_event("WEB_ANALYSIS", {"url": "[https://github.com/trending](https://github.com/trending)"})
    
    # Wait out the completion of remaining tasks in queue buffers
    await asyncio.sleep(3.0)
    
    # Gracefully shut down daemon listener thread
    router.stop_router()
    await listener_task
    print("[*] All concurrent workflows terminated gracefully.")

if __name__ == "__main__":
    # Standard runtime entry point for async systems
    asyncio.run(main())
```
