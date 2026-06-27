# Agent Evaluation Frameworks and Trajectory Testing

Evaluating autonomous agents differs fundamentally from evaluating standard LLMs or RAG applications. While standard pipelines measure static input-output pairs, agent evaluation must analyze the **trajectory**—the multi-step sequence of thoughts, tool selections, and self-corrections an agent executes to arrive at an answer. An agent might reach the correct conclusion via a highly inefficient or hazardous execution path.

---

## 1. Theoretical Foundations: Output vs. Trajectory

When validating enterprise-grade agents, we analyze performance across two distinct dimensions:

| Dimension | Focus Area | Metrics Explored |
| :--- | :--- | :--- |
| **Output Evaluation** | *What* the agent concluded. | Accuracy, completeness, semantic alignment with ground truth. |
| **Trajectory Evaluation** | *How* the agent got there. | Tool call accuracy, loop count efficiency, redundancy, step-by-step reasoning quality. |

### Core Evaluation Challenges
1. **Non-Determinism:** The same agent prompt can take entirely different pathways across execution runs.
2. **Infinite Loops:** Poorly optimized ReAct loops can get trapped calling the same tool repeatedly, draining API budgets.
3. **State Dependency:** Agent performance depends heavily on mock environment states (e.g., mock databases or APIs).

---

## 2. Mathematical Formalization: Trajectory Efficiency & Similarity

To quantify how well an agent executes a task compared to an expert or a baseline path, we can formalize **Trajectory Efficiency ($E_t$)** and **Levenshtein Distance ($D_L$)** over tool call tokens.

### A. Trajectory Efficiency Score
Let $T_{\text{actual}}$ be the number of steps taken by the agent, and $T_{\text{optimal}}$ be the minimal theoretical steps required to resolve the objective.

$$E_t = \max\left(0, 1 - \frac{T_{\text{actual}} - T_{\text{optimal}}}{T_{\text{max_allowed}}}\right)$$

* A score of $1.0$ indicates perfect execution efficiency.
* A score approaching $0.0$ signifies heavy loop redundancy or near-failure.

### B. Trajectory Sequence Alignment
If an agent's tool execution sequence is represented as an ordered list of strings (e.g., `["Search_DB", "Calculator", "Final_Answer"]`), we measure the distance from a golden baseline sequence using an adaptation of the Levenshtein distance matrix formula for sequences:

$$D(i, j) = \begin{cases} 
\max(i, j) & \text{if } \min(i, j) = 0, \\
\min \begin{cases} 
D(i-1, j) + 1 \\ 
D(i, j-1) + 1 \\ 
D(i-1, j-1) + \text{cost} 
\end{cases} & \text{otherwise.} 
\end{cases}$$

Where $\text{cost} = 0$ if the tool tokens align perfectly ($T_{\text{actual}}[i] == T_{\text{golden}}[j]$), else $\text{cost} = 1$.

---

## 3. Production Implementation: Trajectory Evaluator Engine

Below is a complete implementation using an **LLM-as-a-Judge** framework to evaluate a recorded agent execution path against a golden standard dataset.

```python
import os
import json
from typing import List, Dict, Any

# Mocking an LLM client wrapper for evaluation scoring
class EvaluationLLM:
    def evaluate_trajectory(self, objective: str, golden_path: List[str], actual_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Uses an LLM prompt to dissect agent step trajectories, verifying logic and tool correctness.
        """
        # Formulate structured critique context
        steps_dump = json.dumps(actual_steps, indent=2)
        
        # Simulated LLM response parsing based on industry-standard grading criteria
        # In production, this would call client.chat.completions.create with structured outputs (Pydantic models)
        has_loops = "Loop detected" if len(actual_steps) > len(golden_path) + 2 else "No loops"
        
        # Mock calculation reflecting the evaluation criteria
        tool_accuracy = 1.0 if actual_steps[0].get("tool") == golden_path[0] else 0.5
        
        return {
            "reasoning_alignment_score": 0.85 if has_loops == "No loops" else 0.40,
            "tool_selection_accuracy": tool_accuracy,
            "critique": f"Agent correctly targeted baseline goals. {has_loops} during execution pipeline analysis."
        }

class AgentTrajectoryEvaluator:
    def __init__(self):
        self.judge = EvaluationLLM()

    def compute_efficiency(self, actual_steps_count: int, optimal_steps_count: int, max_steps: int = 10) -> float:
        """
        Calculates mathematical trajectory efficiency score.
        """
        if actual_steps_count <= optimal_steps_count:
            return 1.0
        return max(0.0, 1.0 - ((actual_steps_count - optimal_steps_count) / max_steps))

    def run_evaluation_suite(self, test_case: Dict[str, Any], agent_run_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes a comprehensive evaluation over an agent run trace.
        """
        objective = test_case["objective"]
        golden_path = test_case["golden_tool_sequence"]
        
        actual_tool_sequence = [step["tool"] for step in agent_run_trace if "tool" in step]
        
        # 1. Quantify Trajectory Efficiency
        efficiency_score = self.compute_efficiency(
            actual_steps_count=len(agent_run_trace),
            optimal_steps_count=len(golden_path)
        )
        
        # 2. Leverage LLM-As-A-Judge for Semantics & Logic
        judge_metrics = self.judge.evaluate_trajectory(
            objective=objective,
            golden_path=golden_path,
            actual_steps=agent_run_trace
        )
        
        # 3. Compile Composite Score
        composite_score = (efficiency_score * 0.4) + (judge_metrics["tool_selection_accuracy"] * 0.6)
        
        return {
            "test_objective": objective,
            "metrics": {
                "trajectory_efficiency": efficiency_score,
                "tool_selection_accuracy": judge_metrics["tool_selection_accuracy"],
                "reasoning_alignment": judge_metrics["reasoning_alignment_score"],
                "composite_pass_score": composite_score
            },
            "judge_critique": judge_metrics["critique"],
            "status": "PASS" if composite_score >= 0.75 else "FAIL"
        }

# --- Execution Test Verification ---
if __name__ == "__main__":
    evaluator = AgentTrajectoryEvaluator()
    
    # Define reference gold test standard
    golden_test_case = {
        "objective": "Fetch user transaction balance from SQL and calculate 10% tax adjustment.",
        "golden_tool_sequence": ["execute_sql_query", "math_calculator"]
    }
    
    # Scenario: Agent ran and hit an unneeded retrieval loop before answering
    simulated_agent_trace = [
        {"step": 1, "thought": "Need to look up current account balance indices.", "tool": "execute_sql_query"},
        {"step": 2, "thought": "I fetched 5000. Now let me verify schema again just in case.", "tool": "execute_sql_query"},
        {"step": 3, "thought": "Applying mathematical calculation factors.", "tool": "math_calculator"}
    ]
    
    report = evaluator.run_evaluation_suite(test_case=golden_test_case, agent_run_trace=simulated_agent_trace)
    print("=== Production Trajectory Evaluation Report ===")
    print(json.dumps(report, indent=2))
```
