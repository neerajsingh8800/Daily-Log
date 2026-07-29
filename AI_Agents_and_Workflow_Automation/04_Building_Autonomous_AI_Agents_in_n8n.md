# 04: Building Autonomous AI Agents in n8n

This module explores **Autonomous Agent Architectures in Workflow Automation**, covering ReAct reasoning loops, tool binding, memory management, Human-in-the-Loop (HITL) approval gates, and a hands-on production Python agent implementation.

---

## 1. Agentic Architecture vs. Deterministic Workflows

While standard workflows follow explicit, hardcoded control flows (Node A $\rightarrow$ Node B $\rightarrow$ Node C), **Autonomous AI Agents** use Large Language Models as reasoning engines that dynamically determine which tools to invoke and in what order based on user input.

---

## 2. Mathematical Modeling: ReAct Loop Convergence & Cost Calculus

To prevent autonomous agents from getting stuck in infinite tool-calling loops or incurring runaway API charges, workflow engines impose explicit step bounds ($K_{max}$) and evaluate token accumulation.

Let $C_{step}$ be the average token cost per reasoning step, $T_{call}$ be the execution cost per tool API call, and $k$ be the current iteration step ($1 \le k \le K_{max}$):

$$Cost_{Total} = \sum_{k=1}^{K_{max}} \left( C_{step}(k) + T_{call}(k) \right)$$

$$\text{Termination Rule:} \quad \text{Stop if } k \ge K_{max} \quad \text{or} \quad \text{Confidence Score } S \ge 0.95$$

*   **Bounded Loop Guarantee:** Enforcing $K_{max} \le 5$ guarantees deterministic cost upper bounds while allowing sufficient reasoning depth for complex multi-tool tasks.

---

## 3. Human-in-the-Loop (HITL) & Memory Architecture

### 1. Memory Management Paradigms
*   **Window Buffer Memory:** Retains only the last $N$ turns of interaction to conserve context window space.
*   **Vector Store Retrieval Memory:** Persists historical conversation embeddings in a vector database (e.g., Pinecone/Qdrant) and retrieves relevant context via Cosine Similarity.

### 2. Human-in-the-Loop (HITL) Approval Pattern
For high-risk agent operations (e.g., executing database mutations, sending public emails, or invoking financial transactions), the agent execution pauses and emits an approval request (Slack button / Webhook email) before proceeding.

---

## 4. Production Implementation: Autonomous Research & Action Agent in Python

Here is a complete, production-grade Python script implementing an autonomous AI Agent with ReAct reasoning loops, dynamic tool binding (Web Search & Database Query), memory tracking, and bounded step controls.

```python
import os
import json
import logging
from typing import List, Dict, Any, Callable
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AutonomousAgentEngine")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "mock-key-for-development"))


# -------------------------------------------------------------------
# 1. Tool Definitions & Mock Integrations
# -------------------------------------------------------------------
def search_knowledge_base(query: str) -> str:
    """Tool: Searches enterprise knowledge base for internal policy details."""
    logger.info(f"🔧 [TOOL INVOKED] Knowledge Base Search with query: '{query}'")
    mock_db = {
        "refund": "Standard refund policy allows full refunds within 30 days of purchase.",
        "support": "Enterprise support SLA is 2 hours for critical incidents."
    }
    for key, val in mock_db.items():
        if key in query.lower():
            return f"Result: {val}"
    return "Result: No relevant policy records found."


def query_customer_database(email: str) -> str:
    """Tool: Queries user database to fetch transaction history and tier status."""
    logger.info(f"🔧 [TOOL INVOKED] Customer Database Query for: '{email}'")
    if "neeraj" in email.lower():
        return json.dumps({
            "user_id": "USR-9901",
            "tier": "Enterprise VIP",
            "recent_purchase_days_ago": 12,
            "order_value": 450.00
        })
    return json.dumps({"error": "User record not found."})


# Tool Schema Registry for OpenAI Function Calling
TOOLS_REGISTRY: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Searches internal company knowledge base and policies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query term."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_customer_database",
            "description": "Fetches customer tier, purchase history, and details by email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "The user's email address."}
                },
                "required": ["email"]
            }
        }
    }
]

TOOL_FUNCTIONS: Dict[str, Callable] = {
    "search_knowledge_base": search_knowledge_base,
    "query_customer_database": query_customer_database
}


# -------------------------------------------------------------------
# 2. Autonomous Agent Engine (ReAct Pattern)
# -------------------------------------------------------------------
class AutonomousAgent:
    """ReAct Agent executing multi-step reasoning and dynamic tool calls."""

    def __init__(self, system_prompt: str, max_steps: int = 5):
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.memory: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    def run(self, user_goal: str) -> str:
        logger.info(f"🚀 Starting Autonomous Agent Task: '{user_goal}'")
        self.memory.append({"role": "user", "content": user_goal})

        for step in range(1, self.max_steps + 1):
            logger.info(f"🧠 [REASONING STEP {step}/{self.max_steps}] Evaluating context...")

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=self.memory,
                tools=TOOLS_REGISTRY,
                tool_choice="auto",
                temperature=0.0
            )

            message = response.choices[0].message
            self.memory.append(message)

            # Check if agent decided to invoke tools
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)
                    
                    if fn_name in TOOL_FUNCTIONS:
                        # Execute Tool
                        tool_output = TOOL_FUNCTIONS[fn_name](**fn_args)
                        
                        # Append Observation to Memory
                        self.memory.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(tool_output)
                        })
            else:
                # No more tools called; Agent reached final answer
                logger.info("✅ Agent reached conclusive response.")
                return message.content

        return "⚠️ Agent terminated: Reached maximum step limit without conclusion."


# -------------------------------------------------------------------
# 3. Execution Routine
# -------------------------------------------------------------------
def main():
    agent_system_prompt = (
        "You are an autonomous customer support agent. "
        "Use available tools to research customer status and company policy before making decisions. "
        "Provide a polite, well-reasoned final answer."
    )

    agent = AutonomousAgent(system_prompt=agent_system_prompt, max_steps=5)
    
    user_query = "Check if customer neerajrathore5821@gmail.com is eligible for a full refund on his recent order."
    final_result = agent.run(user_query)

    print("\n================ FINAL AGENT OUTPUT ================")
    print(final_result)


if __name__ == "__main__":
    main()
```
