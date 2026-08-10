# Module 11: Agentic Workflows, Execution, and Tracing

As LLMs transition from simple single-turn text generation to autonomous, goal-directed agents, managing execution state, tool reliability, loop convergence, and observability becomes critical. Unlike static chain-of-thought workflows, **agentic workflows** rely on dynamic state machines, persistent memory, cyclic loops, and multi-agent coordination.

This module covers the core control-loop architectures of agentic systems, formal execution reliability metrics, multi-agent state orchestration using **LangGraph**, and end-to-end execution tracing with **LangSmith / OpenInference**.

---

## 1. Theoretical Foundations

### 1.1 Control-Loop Patterns in Agentic Systems
* **ReAct (Reasoning + Acting)**: The foundational loop where an agent generates a thought, selects an external tool action, observes the result, and iterates until a stopping condition is met.
* **Stateful Directed Graphs**: Formalizing agent execution as a state transition system $S_{t+1} = f(S_t, A_t, O_t)$, allowing loops, human-in-the-loop interventions, and dynamic routing based on state evaluation.
* **Multi-Agent Architectures**: Hierarchical or peer-to-peer topologies (e.g., Supervisor-Worker, Router-Executor) where specialized agents communicate via a shared state or queue.

* ### 1.2 Mathematical Formulation of Agent Execution & Reliability

#### Agent Task Completion Rate (Reliability Metric)
Let $T$ be a set of evaluation tasks. For each task $i \in T$, let $C_i \in \{0, 1\}$ denote task completion accuracy, and $K_i$ denote the number of tool execution steps taken. The efficiency-weighted agent success score $S_{agent}$ is defined as:

$$S_{agent} = \frac{1}{\vert{}T\vert{}} \sum_{i=1}^{\vert{}T\vert{}} C_i \cdot \exp\left( -\alpha \max(0, K_i - K_{opt}) \right)$$

Where:
* $K_{opt}$ is the optimal number of steps for the task.
* $\alpha > 0$ is a penalty factor for excessive tool calls or infinite loops.

#### Tool Execution Error Rate (TEER)
Measures the frequency of tool call failures (parse errors, incorrect schemas, runtime exceptions) relative to total tool invocations $N_{invocations}$:

$$\text{TEER} = \frac{\sum_{j=1}^{N_{invocations}} \mathbb{I}(\text{ToolCall}_j = \text{Error})}{N_{invocations}}$$

---

## 2. Production Stateful Agent Workflows with LangGraph

The following script implements a resilient stateful agent system using `LangGraph` with state checkpointing, conditional tool routing, and loop termination safeguards.

### Prerequisites

```bash
pip install langgraph langchain-openai langchain-community langsmith
```

### Python Execution Script(agentic_workflow.py)
```python
import os
from typing import Annotated, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# 1. Define State Schema
class AgentState(TypedDict):
    # 'add_messages' appends new messages rather than overwriting
    messages: Annotated[list[BaseMessage], add_messages]
    loop_count: int

# 2. Define Custom Tools
@tool
def search_system_logs(query: str) -> str:
    """Searches cluster logs for recent operational errors."""
    if "database" in query.lower():
        return "ERROR 500: Database Connection Timeout at 2026-08-10 10:15:00 UTC"
    return "LOG: System operational, CPU load 42%, Memory usage 68%"

@tool
def trigger_alert_pager(channel: str, message: str) -> str:
    """Sends an incident alert notification to an operational team."""
    return f"SUCCESS: Alert dispatched to channel '{channel}' with payload: '{message}'"

tools = [search_system_logs, trigger_alert_pager]
tool_node = ToolNode(tools)

# 3. Define LLM Model with Tool Binding
model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

# 4. Define Graph Nodes
def call_agent(state: AgentState):
    """Agent node that processes messages and decides next action."""
    messages = state["messages"]
    loop_count = state.get("loop_count", 0) + 1
    
    # System prompt enforcing safe execution
    system_prompt = SystemMessage(
        content="You are an SRE Incident Agent. Investigate system errors using tools and notify operations if needed."
    )
    
    response = model.invoke([system_prompt] + messages)
    return {"messages": [response], "loop_count": loop_count}

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Conditional Edge Router to control execution flow or break infinite loops."""
    messages = state["messages"]
    last_message = messages[-1]
    loop_count = state.get("loop_count", 0)

    # Infinite Loop Prevention Safeguard
    if loop_count > 6:
        print(" Emergency Stop: Maximum loop execution threshold reached.")
        return END

    # Route to Tool execution if LLM requested tool call
    if last_message.tool_calls:
        return "tools"
    
    return END

# 5. Construct State Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("agent", call_agent)
workflow.add_node("tools", tool_node)

# Add Edges
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

# Compile Execution Graph
app = workflow.compile()

# 6. Execute Workflow Example
if __name__ == "__main__":
    initial_input = {
        "messages": [HumanMessage(content="Check system logs for database errors, and alert the 'sre-oncall' team if found.")],
        "loop_count": 0
    }

    print("--- Starting Agent Execution ---")
    for event in app.stream(initial_input):
        for node_name, state_update in event.items():
            print(f"\n[Node Executed]: {node_name}")
            if "messages" in state_update:
                latest_msg = state_update["messages"][-1]
                print(f"Content: {latest_msg.content}")
                if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                    print(f"Tool Calls Initiated: {latest_msg.tool_calls}")
```

## 3. Agent Execution Tracing & Observability with LangSmith

Tracing stateful agents requires recording step-by-step executions, input/output prompts, latencies, token consumption, and tool outcomes.

### Enabling OpenTelemetry / LangSmith Environment Tracing

Add these environment variables to automatically capture traces without modifying business logic:
```BASH
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="ls__your_langsmith_api_key_here"
export LANGCHAIN_PROJECT="sre-agent-production"
```

### Programmatic Tracing & Custom Span Logging

```python
from langsmith import traceable

@traceable(name="sre_incident_investigation_run", run_type="chain")
def run_traced_agent_workflow(user_query: str):
    """Wraps agent execution with custom span tracing for LangSmith observability."""
    inputs = {
        "messages": [HumanMessage(content=user_query)],
        "loop_count": 0
    }
    
    result = app.invoke(inputs)
    
    # Extract final answer
    final_output = result["messages"][-1].content
    return {
        "final_output": final_output,
        "total_steps": result["loop_count"]
    }

if __name__ == "__main__":
    response = run_traced_agent_workflow("Investigate database status and notify team.")
    print("\nTraced Execution Complete:")
    print(response)
```
