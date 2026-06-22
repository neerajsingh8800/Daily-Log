# The ReAct Framework and Reasoning Loops

Traditional LLM interactions rely on simple single-turn inference or linear chains. However, complex problem-solving requires autonomous execution paths where the model can observe runtime realities, plan next moves, and interact with external systems. The **ReAct (Reasoning and Acting)** framework combines conversational reasoning trace generation with task-specific action execution loops.

This document explores the architectural mechanics of tight execution loops, the attention budget constraints of multi-turn trajectories, and an autonomous implementation from scratch.

---

## 1. The ReAct Paradigm: Thought, Action, Observation

The core philosophy of ReAct is an imitation of human problem-solving patterns: alternating between *thinking* about a problem and *acting* to gather more information.

* **Thought:** The model analyzes the current state, establishes sub-goals, and reasons about what information is missing.
* **Action:** The model interacts with the physical or digital environment by emitting a structured tool invocation request (e.g., executing a SQL query or a web search).
* **Observation:** The system executes the tool, captures the output, and appends it back into the model's prompt memory space as ground-truth reality.

By interleaving these steps, the model continuously corrects its execution path based on live feedback, mitigating common hallucinations seen in long static generations.

---

## 2. Attention Dynamics & Trajectory Context Windows

Every iteration of the **Thought-Action-Observation** loop appends new data to the active context window. This builds an execution **Trajectory**.

Let $T_i$ represent the conversation history context at loop iteration $i$. The state transition across iterations is formalized as:

$$T_i = T_{i-1} \mathbin{\Vert} \text{Thought}_i \mathbin{\Vert} \text{Action}_i \mathbin{\Vert} \text{Observation}_i$$

Where $\mathbin{\Vert}$ denotes string concatenation. 

### The Token Decay Constraint
Because attention complexity scales quadratically ($O(L^2)$) relative to token sequence length $L$, long trajectories pose two major enterprise constraints:
1.  **Context Bloat:** Repeatedly parsing expanding histories balloons compute resource costs during the prompt prefill phase.
2.  **Early Trajectory Loop Halting:** If an agent gets caught in a logical loop (e.g., repeatedly calling a failing API tool), it can exhaust its maximum context window allocation, crashing the runtime. Production systems must implement a hard limit constraint ($i \le I_{\text{max}}$) to prevent runaway iteration failures.

---

## 3. Production-Grade Implementation: Autonomous ReAct Engine From Scratch

Below is a complete, self-contained Python runtime simulating an autonomous ReAct reasoning agent. It manages its own loop context, uses string matching to isolate thought components, executes a mock calculations tool, and stops once a definitive final answer is determined.

```python
import time
from typing import List, Dict, Any, Tuple

class CoreReActAgent:
    """Pure Python autonomous ReAct execution loop handling thinking traces and tools."""
    def __init__(self, max_loops: int = 5):
        self.max_loops = max_loops
        self.trajectory_history: List[str] = []

    def _execute_calculator_tool(self, execution_args: str) -> str:
        """Mock computational tool executing safe isolated math lookups."""
        try:
            # Clean string format constraints
            sanitized = execution_args.strip(" \"'")
            result = eval(sanitized, {"__builtins__": None}, {})
            return f"Success: Result calculation matches {result}"
        except Exception as e:
            return f"Error: Failed to process calculation expression. Details: {str(e)}"

    def _simulate_llm_inference_step(self, loop_index: int) -> str:
        """
        Simulates structured multi-turn model completions emitting 
        explicit Thought/Action/Final Answer syntax.
        """
        if loop_index == 0:
            return (
                "Thought: I need to calculate the standard operating profit margin. "
                "First, I must compute total revenue minus total operating costs.\n"
                "Action: calculate[250000 - 175000]"
            )
        elif loop_index == 1:
            return (
                "Thought: I have the net operational profit value (75000). Now I must divide "
                "this value by the initial revenue base (250000) and scale to get the percentage.\n"
                "Action: calculate[75000 / 250000 * 100]"
            )
        else:
            return (
                "Thought: I have determined the exact proportion profile.\n"
                "Final Answer: The enterprise operating profit margin scales to exactly 30.0%."
            )

    def run(self, user_objective: str):
        """Executes the core autonomous sequence framework."""
        print(f"=== Starting ReAct Execution Loop ===")
        print(f"Objective: {user_objective}\n")
        
        self.trajectory_history.append(f"Objective: {user_objective}")
        
        for current_step in range(self.max_loops):
            print(f"--- Iteration Loop #{current_step + 1} ---")
            
            # Step 1: Run model inference based on active trajectory status
            llm_output = self._simulate_llm_inference_step(current_step)
            print(llm_output)
            self.trajectory_history.append(llm_output)
            
            # Check for termination token anchor
            if "Final Answer:" in llm_output:
                print("\n[Target Termination Achieved Successfully]")
                break
                
            # Step 2: Parse and route tool invocation requests
            if "Action:" in llm_output:
                action_line = [line for line in llm_output.split("\n") if line.startswith("Action:")][0]
                # Extract parameters enclosed inside structural brackets
                tool_call = action_line.replace("Action:", "").strip()
                tool_name = tool_call.split("[")[0]
                tool_args = tool_call.split("[")[1].split("]")[0]
                
                if tool_name == "calculate":
                    print(f" >> [System Tool Execution]: Invoking '{tool_name}' with arguments: {tool_args}")
                    observation = self._execute_calculator_tool(tool_args)
                    print(f" << [Observation Response Captured]: {observation}")
                    self.trajectory_history.append(f"Observation: {observation}")
                else:
                    observation = f"Error: Tool name '{tool_name}' is unregistered."
                    self.trajectory_history.append(f"Observation: {observation}")
                    
            time.sleep(0.1) # Small cooling padding trace
        else:
            print("\n[Runtime Warning: Agent execution halted due to max loop bounds limit constraints.]")


# --- Verification Engine Hook ---
if __name__ == "__main__":
    agent_instance = CoreReActAgent(max_loops=4)
    target_prompt = "Compute operational margins for Q3 asset sheets showing $250k revenue and $175k expenses."
    agent_instance.run(user_objective=target_prompt)
```
