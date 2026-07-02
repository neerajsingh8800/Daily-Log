# Module 11: Cost Optimization and Token Management Strategies

Scaling AI agents from local prototypes to production enterprise systems introduces a critical constraint: **Unit Economics**. Continuous autonomous loops (like ReAct or Plan-and-Execute) can rack up massive API bills if left unchecked, as they consume tokens on every reasoning step, tool observation, and reflection iteration.

This module covers the principles of token economics, strategies for context pruning, and implementing intelligent model routing to drastically reduce Total Cost of Ownership (TCO) for agentic systems.

---

## 1. The Economics of Agentic LLM Calls

In an agentic workflow, cost is a function of both the input context (which grows with every step) and the generated reasoning outputs. 

### Total Cost Formulation
For a single agent execution trace spanning $k$ reasoning steps, the Total Cost ($C_{\text{total}}$) is calculated as:

$$C_{\text{total}} = \sum_{i=1}^{k} \left( T_{\text{in}}^{(i)} \cdot c_{\text{in}} + T_{\text{out}}^{(i)} \cdot c_{\text{out}} \right)$$

Where:
* $T_{\text{in}}^{(i)}$ is the number of input tokens at step $i$.
* $T_{\text{out}}^{(i)}$ is the number of generated output tokens at step $i$.
* $c_{\text{in}}$ and $c_{\text{out}}$ are the respective costs per token for the chosen model.

Because $T_{\text{in}}$ typically includes the entire conversation history and previous tool outputs, $T_{\text{in}}$ grows linearly (or quadratically in poorly designed systems) with $i$.

---

## 2. Token Management & Optimization Strategies

To prevent exponential cost bloat, production systems must actively manage the payload sent to the LLM.

### 1. Context Window Pruning (FIFO & Summarization)
Instead of appending all past steps into the system prompt, apply a sliding window or a periodic summarization trigger. When a threshold (e.g., 4000 tokens) is hit, a smaller, cheaper model (like Haiku or Llama-3-8B) summarizes the early episodic memory into a dense paragraph.

### 2. Semantic Caching
If your agent handles repetitive tasks (e.g., "Summarize today's server errors"), querying the LLM every time is a waste. By vectorizing the user prompt and comparing it against a cache of previous prompts (using Cosine Similarity), you can return a cached response if the similarity crosses a high threshold (e.g., 95%).

### 3. Dynamic Model Routing (Cascade Routing)
Not all agent tasks require a heavyweight model like GPT-4o or Claude 3.5 Sonnet.
* **Tier 1 (Cheap/Fast):** Formatting JSON, extracting regex, simple routing, summarization.
* **Tier 2 (Heavy/Expensive):** Complex multi-step planning, code generation, ambiguous reasoning.

---

## 3. Production Implementation: Token Budgeting & Model Routing

The following implementation demonstrates a `TokenBudgetManager` to track costs per trace and a `DynamicModelRouter` that cascades tasks to the cheapest capable model based on task complexity and token count limits.

```python
import tiktoken
import time
from typing import Dict, Any, List, Optional
from enum import Enum

# ==========================================
# Enums and Configurations
# ==========================================
class ModelTier(Enum):
    TIER_1_CHEAP = "gpt-3.5-turbo"     # Fast, cheap, for simple tasks
    TIER_2_HEAVY = "gpt-4o"            # Expensive, high reasoning capabilities

# Costs defined per 1,000 tokens (Hypothetical standard rates)
PRICING_TABLE = {
    ModelTier.TIER_1_CHEAP.value: {"input": 0.0005, "output": 0.0015},
    ModelTier.TIER_2_HEAVY.value: {"input": 0.0050, "output": 0.0150},
}

class TokenBudgetManager:
    def __init__(self, max_budget_usd: float = 0.50):
        self.total_cost_usd = 0.0
        self.max_budget_usd = max_budget_usd
        # Using tiktoken for accurate OpenAI token counting
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def record_usage(self, model: str, input_tokens: int, output_tokens: int):
        if model not in PRICING_TABLE:
            raise ValueError(f"Pricing for model {model} not configured.")
        
        cost_in = (input_tokens / 1000.0) * PRICING_TABLE[model]["input"]
        cost_out = (output_tokens / 1000.0) * PRICING_TABLE[model]["output"]
        trace_cost = cost_in + cost_out
        
        self.total_cost_usd += trace_cost
        print(f"   [Budget] Trace Cost: ${trace_cost:.5f} | Total Session Cost: ${self.total_cost_usd:.5f}")

        if self.total_cost_usd >= self.max_budget_usd:
            raise RuntimeError(f"CIRCUIT BREAKER: Token budget of ${self.max_budget_usd} exceeded!")

class DynamicModelRouter:
    def __init__(self, budget_manager: TokenBudgetManager):
        self.budget_manager = budget_manager

    def determine_optimal_model(self, prompt: str, task_complexity: str) -> str:
        """
        Routes the prompt to the most cost-effective model based on explicit 
        complexity flags and prompt length.
        """
        token_count = self.budget_manager.count_tokens(prompt)
        print(f"   [Router] Evaluating prompt length: {token_count} tokens.")

        # Logic: Simple tasks or massive context summaries go to the cheap model
        if task_complexity == "low" or (task_complexity == "medium" and token_count > 5000):
            print(f"   [Router] Routing to {ModelTier.TIER_1_CHEAP.value} (Cost Optimization)")
            return ModelTier.TIER_1_CHEAP.value
        else:
            print(f"   [Router] Routing to {ModelTier.TIER_2_HEAVY.value} (High Reasoning Required)")
            return ModelTier.TIER_2_HEAVY.value

    def simulate_llm_call(self, prompt: str, task_complexity: str) -> str:
        """Simulates an LLM API call while tracking precise costs."""
        model = self.determine_optimal_model(prompt, task_complexity)
        
        input_tokens = self.budget_manager.count_tokens(prompt)
        
        # Simulate processing time and generated output
        time.sleep(0.5)
        simulated_output = f"Processed agentic response using {model}."
        output_tokens = self.budget_manager.count_tokens(simulated_output)
        
        # Record the cost to the budget manager
        self.budget_manager.record_usage(model, input_tokens, output_tokens)
        
        return simulated_output

# ==========================================
# Execution Verification Flow
# ==========================================
if __name__ == "__main__":
    # Initialize a strict budget for this agent trace ($0.05 limit)
    budget_manager = TokenBudgetManager(max_budget_usd=0.05)
    router = DynamicModelRouter(budget_manager)

    print("\n[*] Starting Agent Execution Trace with Cost Tracking\n")

    try:
        # Step 1: Simple formatting task (Should route to cheap model)
        print("--- Step 1: Formatting raw data ---")
        prompt_1 = "Format the following user inputs into a JSON array: [John, 25], [Alice, 30]"
        router.simulate_llm_call(prompt_1, task_complexity="low")

        # Step 2: Complex reasoning (Should route to expensive model)
        print("\n--- Step 2: Complex Plan-and-Execute reasoning ---")
        prompt_2 = "Analyze the provided system logs. Determine the root cause of the OOM exception and write a mitigation plan."
        router.simulate_llm_call(prompt_2, task_complexity="high")

        # Step 3: Massive payload summarization (Should route to cheap model to avoid cost spike)
        print("\n--- Step 3: Summarizing massive log payload ---")
        # Simulating a massive 6000+ token context window
        prompt_3 = "Summarize these logs: " + ("ERROR: Memory limit exceeded. " * 1500)
        router.simulate_llm_call(prompt_3, task_complexity="medium")

    except RuntimeError as e:
        print(f"\n[-] {e}")
        
    print(f"\n[+] Agent trace completed. Final execution cost: ${budget_manager.total_cost_usd:.5f}")
```
