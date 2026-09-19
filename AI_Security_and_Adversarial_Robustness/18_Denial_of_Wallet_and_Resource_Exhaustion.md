# 18: Denial of Wallet (DoW) and Algorithmic Resource Exhaustion

## 1. Overview & Threat Surface

While traditional Denial of Service (DoS) attacks aim to crash infrastructure by exhausting CPU or network bandwidth, AI-native applications face a distinct financial and operational vulnerability: **Denial of Wallet (DoW)** and **Algorithmic Resource Exhaustion**. 

In cloud-hosted LLM deployments, inference costs scale directly with consumed input and output tokens, recursive tool calls, and long-context processing windows. Attackers leverage adversarial prompts designed to force models into high-cost execution states—such as recursive agent loops, verbose reasoning chain expansions, and maximum-length context inflations—without triggering traditional network-level rate limits.

### Resource Exhaustion Vulnerability Taxonomy

| Attack Vector | Mechanism | Impact |
| :--- | :--- | :--- |
| **Recursive Agent Loops** | Exploiting decision logic to trap autonomous agents in infinite loop tool calls. | Exponential API billing spikes, exhausted thread pools, and application lockup. |
| **Context Padding / Inflation** | Filling inputs with large benign text blocks to force long-context transformer processing. | High prompt-token pricing and degraded inference response times for legitimate users. |
| **Sycophancy & Verbosity Triggers** | Prompting model to emit long code blocks, step-by-step chain-of-thought, or repetitive text. | Exceeding token limits and maximizing output token costs. |
| **Algorithmic Complexity Exploits** | Triggering heavy non-linear tasks (e.g., recursive tool calls or complex regex execution). | Host CPU/RAM starvation in self-hosted inference runtimes (e.g., vLLM, Ollama). |

---

## 2. Threat Mechanics: Unbounded Execution & Cost Amplification

1. **Infinite Loop Injection:** An attacker submits a task to a multi-step research agent:  
   `"Research this topic. If you find conflicting sources, perform another search until 100% consensus is reached across all web pages."`  
   Because web data inherently contains conflicting opinions, the agent loops indefinitely, triggering hundreds of API calls.
2. **Output Token Exhaustion:** An attacker inputs a prompt designed to generate maximum-length responses:  
   `"Provide a list of 10,000 unique names with a detailed 500-word biography for each name."`  
   Without strict output token limits, a single request can consume thousands of completion tokens.
3. **Context Window Inflation:** Attackers inject massive hidden blocks of text inside RAG search queries to force expensive embedding and re-ranking computations on every user interaction.

---

## 3. Defense Architecture for Financial & Execution Limits

To mitigate Denial of Wallet attacks, applications must enforce **Semantic Cost Circuit Breakers**, strict execution budgets, and rate limits at the application layer rather than relying solely on network firewalls.

1. **Token & Context Bounds:** Enforce hard constraints on input prompt size and output completion limits at both the API wrapper and model configuration levels.
2. **Recursion Depth Counters:** Track and cap the maximum number of consecutive tool iterations an autonomous agent can perform per user session.
3. **Semantic Cost Circuit Breakers:** Monitor real-time API spending per user/tenant and dynamically throttle or trip execution when anomalous usage spikes occur.
4. **Time-To-First-Byte (TTFB) & Execution Timeouts:** Set strict time budgets on agent execution pipelines to kill runaway processes automatically.

---

## 4. Hands-On Python Implementations

### Example 1: Bounded Agent Execution Controller with Recursion & Token Limits

```python
import time
from typing import Dict, Any, Callable

class ResourceExhaustionException(Exception):
    pass

class BoundedAgentController:
    def __init__(self, max_iterations: int = 5, max_total_tokens: int = 4000, execution_timeout_sec: float = 10.0):
        self.max_iterations = max_iterations
        self.max_total_tokens = max_total_tokens
        self.execution_timeout_sec = execution_timeout_sec

    def execute_agent_loop(
        self, 
        task_prompt: str, 
        step_callback: Callable[[int, str], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Executes an agent workflow while enforcing maximum iteration bounds, 
        cumulative token budgets, and execution timeouts.
        """
        start_time = time.time()
        iteration_count = 0
        total_tokens_used = 0
        execution_log = []

        while iteration_count < self.max_iterations:
            # 1. Check Execution Timeout
            elapsed = time.time() - start_time
            if elapsed > self.execution_timeout_sec:
                raise ResourceExhaustionException(
                    f"EXECUTION TIMEOUT: Agent exceeded maximum execution time of {self.execution_timeout_sec}s."
                )

            iteration_count += 1

            # 2. Execute Step via Callback
            step_result = step_callback(iteration_count, task_prompt)
            tokens_in_step = step_result.get("tokens_used", 0)
            total_tokens_used += tokens_in_step
            execution_log.append(step_result)

            # 3. Check Cumulative Token Limits
            if total_tokens_used > self.max_total_tokens:
                raise ResourceExhaustionException(
                    f"TOKEN BUDGET EXCEEDED: Accumulated {total_tokens_used} tokens (Limit: {self.max_total_tokens})."
                )

            # Check if task naturally completed
            if step_result.get("status") == "COMPLETED":
                return {
                    "status": "SUCCESS",
                    "iterations": iteration_count,
                    "total_tokens": total_tokens_used,
                    "logs": execution_log
                }

        # If loop exits without completion
        raise ResourceExhaustionException(
            f"MAX RECURSION REACHED: Agent failed to terminate within {self.max_iterations} iterations."
        )

# Example Usage
if __name__ == "__main__":
    controller = BoundedAgentController(max_iterations=3, max_total_tokens=1500, execution_timeout_sec=2.0)

    # Simulated Step Callback representing an Agent execution step
    def mock_agent_step(iteration: int, prompt: str) -> Dict[str, Any]:
        return {
            "iteration": iteration,
            "status": "RUNNING", # Simulates an agent trapped in a loop
            "tokens_used": 600,
            "action": f"Executed search step {iteration}"
        }

    try:
        controller.execute_agent_loop("Research infinite loop topic", mock_agent_step)
    except ResourceExhaustionException as e:
        print(f"Intercepted Denial of Wallet Attempt:\n{e}")
```
