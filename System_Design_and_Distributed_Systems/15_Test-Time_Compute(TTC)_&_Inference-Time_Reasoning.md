# Module 15: Test-Time Compute (TTC) & Inference-Time Reasoning

As Large Language Model (LLM) scaling laws for pre-training reach diminishing returns, state-of-the-art architectures (such as OpenAI's o1 and DeepSeek-R1) shift compute investment toward **Inference-Time Search and Reasoning**. Rather than relying purely on single-pass forward generation, **Test-Time Compute (TTC)** allows models to dynamically allocate FLOPs during inference via tree searches, self-correction loops, and verification scoring.

This module covers the core mechanics of Test-Time Compute, including Monte Carlo Tree Search (MCTS) over reasoning paths, Process Reward Models (PRMs) vs. Outcome Reward Models (ORMs), self-consistency consensus math, and a production-grade MCTS inference engine implementation in Python.

---

## 1. Theoretical Foundations

### 1.1 Test-Time Compute Paradigms

* **Chain-of-Thought (CoT) Tree Search**:
  * Represents multi-step reasoning as a dynamic search tree where nodes are intermediate thought steps and edges are valid token generation transitions.
  * Replaces greedy autoregressive decoding ($O(L)$ sequential steps) with structured exploration algorithms like **Beam Search**, **A* Search**, or **Monte Carlo Tree Search (MCTS)**.

* **Process Reward Models (PRMs) vs. Outcome Reward Models (ORMs)**:
  * **Outcome Reward Model (ORM)**: Evaluates a candidate trajectory only at the final step ($r \in \{0, 1\}$), providing sparse signals that struggle on complex multi-step math or code generation tasks.
  * **Process Reward Model (PRM)**: Evaluates the correctness of *each individual reasoning step* $s_t$, giving fine-grained dense rewards $r_t \in [0, 1]$ to guide search algorithms toward promising branches before reaching a conclusion.

* **Self-Consistency & Majority Voting**:
  * Samples $K$ independent reasoning paths at temperature $T > 0$ and aggregates final answers using weighted consensus scoring.
  * Scales accuracy logarithmically with the number of generated rollouts.

---

### 1.2 Mathematical Foundations

#### 1. Upper Confidence Bound for Trees (UCT) Selection
During the selection phase of MCTS, child node $i$ is picked to balance **exploitation** (high PRM reward) and **exploration** (infrequently visited paths) according to the UCT formula:

$$\text{UCT}(i) = Q(i) + c_{\text{puct}} \cdot P(i) \cdot \frac{\sqrt{N_{\text{parent}}}}{1 + N(i)}$$

Where:
* $Q(i)$: Average Process Reward Model score for node $i$.
* $P(i)$: Prior probability of selecting step $i$ output by the base LLM generator.
* $N_{\text{parent}}$: Total visit count of the parent node.
* $N(i)$: Visit count of child node $i$.
* $c_{\text{puct}}$: Exploration coefficient balancing risk vs. reward.

#### 2. Process Reward Expected Trajectory Valuation
For a sequence of $T$ reasoning steps $S = (s_1, s_2, \dots, s_T)$ with per-step PRM scores $r(s_t) \in [0, 1]$, the trajectory value $V(S)$ is calculated using cumulative geometric mean scoring:

$$V(S) = \left( \prod_{t=1}^{T} r(s_t) \right)^{\frac{1}{T}} = \exp\left( \frac{1}{T} \sum_{t=1}^{T} \ln r(s_t) \right)$$

#### 3. Test-Time Compute Scaling Law
Let $C_{\text{train}}$ be the training compute investment and $C_{\text{test}}$ be the test-time FLOPs budget allocated per query. The total reasoning error rate $E(C_{\text{test}})$ decays according to a power-law relationship:

$$E(C_{\text{test}}) \approx \alpha \cdot C_{\text{test}}^{-\gamma} + E_{\text{floor}}$$

Where $\gamma \in [0.3, 0.6]$ is the test-time compute scaling exponent and $E_{\text{floor}}$ is the residual irreducible error of the base model.

---

## 2. Reasoning Paradigms Comparison

| Metric / Dimension | Greedy Decoding | Self-Consistency (Maj@K) | Beam Search + PRM | MCTS + PRM (o1/DeepSeek-R1 style) |
| :--- | :--- | :--- | :--- | :--- |
| **Test-Time Compute FLOPs** | $1\times$ (Baseline) | $K \times$ (Linear) | $B \times L$ (Medium) | Dynamic ($10\times - 1000\times$) |
| **Search Mechanism** | Autoregressive Greedy | Independent Sampling | Breadth-First Search | Dynamic Tree Exploration |
| **Backtracking Support** | No | No | Limited | Full (State Rollbacks) |
| **Step-Level Verification** | No | No | Yes (PRM) | Yes (PRM + Value Estimation) |
| **Optimal Use Case** | Low-latency chat | Multiple-choice evaluation | Short multi-step problems | Complex mathematical & code proofs |

---

## 3. Production MCTS Test-Time Compute Engine Implementation

This Python script implements a **Monte Carlo Tree Search (MCTS) Test-Time Reasoning Engine** guided by a simulated Process Reward Model (PRM) and an LLM step generator.

### Prerequisites

```bash
pip install numpy pydantic
```

### Python Implementation (ttc_reasoning_engine.py)
```python
import math
import random
import time
from typing import List, Optional, Dict
import numpy as np


# -------------------------------------------------------------------
# 1. NODE STRUCTURE FOR THE REASONING SEARCH TREE
# -------------------------------------------------------------------
class MCTSNode:
    def __init__(self, step_text: str, parent: Optional['MCTSNode'] = None, prior_prob: float = 1.0):
        self.step_text = step_text
        self.parent = parent
        self.children: List['MCTSNode'] = []
        
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.prior_prob: float = prior_prob
        self.is_terminal: bool = False
        self.prm_score: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def uct_score(self, c_puct: float = 1.41) -> float:
        if not self.parent:
            return 0.0
        
        exploration = c_puct * self.prior_prob * (math.sqrt(self.parent.visits) / (1 + self.visits))
        return self.mean_value + exploration


# -------------------------------------------------------------------
# 2. SIMULATED BASE LLM GENERATOR & PROCESS REWARD MODEL (PRM)
# -------------------------------------------------------------------
class LLMStepGenerator:
    """Simulates an LLM proposing multi-step reasoning steps."""
    def generate_candidate_steps(self, current_path: List[str]) -> List[tuple[str, float]]:
        depth = len(current_path)
        if depth >= 3:
            return [("Final Answer: Therefore, x = 42.", 0.95)]
        
        candidates = [
            (f"Step {depth+1}: Simplify equation by isolating terms.", 0.6),
            (f"Step {depth+1}: Apply integration by parts across boundary.", 0.3),
            (f"Step {depth+1}: [Flawed step] Assume x = 0 unconditionally.", 0.1)
        ]
        return candidates


class ProcessRewardModel:
    """Evaluates the semantic validity of individual reasoning steps."""
    def evaluate_step(self, step_text: str) -> float:
        if "[Flawed step]" in step_text:
            return 0.05
        elif "Final Answer" in step_text:
            return 1.0
        elif "Isolating terms" in step_text:
            return 0.88
        return 0.50


# -------------------------------------------------------------------
# 3. MONTE CARLO TREE SEARCH (MCTS) INFERENCE ENGINE
# -------------------------------------------------------------------
class MCTSReasoningEngine:
    def __init__(
        self,
        generator: LLMStepGenerator,
        prm: ProcessRewardModel,
        c_puct: float = 1.41,
        max_rollouts: int = 30
    ):
        self.generator = generator
        self.prm = prm
        self.c_puct = c_puct
        self.max_rollouts = max_rollouts

    def solve(self, prompt: str) -> str:
        root = MCTSNode(step_text=prompt)

        for rollout in range(self.max_rollouts):
            node = self._select(root)
            
            if not node.is_terminal:
                node = self._expand(node)
                
            value = self._evaluate(node)
            self._backpropagate(node, value)

        best_path = self._extract_best_trajectory(root)
        return "\n".join([node.step_text for node in best_path])

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.children and not node.is_terminal:
            node = max(node.children, key=lambda child: child.uct_score(self.c_puct))
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        current_path = self._get_path_to_node(node)
        candidates = self.generator.generate_candidate_steps(current_path)

        for step_text, prior in candidates:
            child = MCTSNode(step_text=step_text, parent=node, prior_prob=prior)
            child.prm_score = self.prm.evaluate_step(step_text)
            
            if "Final Answer" in step_text:
                child.is_terminal = True
                
            node.children.append(child)

        return node.children[0] if node.children else node

    def _evaluate(self, node: MCTSNode) -> float:
        path = self._get_path_to_node(node)
        if not path:
            return 0.0
        
        # Calculate trajectory PRM value using geometric mean
        scores = [self.prm.evaluate_step(n.step_text) for n in path if n.parent is not None]
        if not scores:
            return 0.0
        
        log_sum = sum(math.log(max(s, 1e-6)) for s in scores)
        geom_mean = math.exp(log_sum / len(scores))
        return geom_mean

    def _backpropagate(self, node: MCTSNode, value: float):
        curr: Optional[MCTSNode] = node
        while curr is not None:
            curr.visits += 1
            curr.value_sum += value
            curr = curr.parent

    def _get_path_to_node(self, node: MCTSNode) -> List[MCTSNode]:
        path = []
        curr: Optional[MCTSNode] = node
        while curr is not None:
            path.append(curr)
            curr = curr.parent
        return path[::-1]

    def _extract_best_trajectory(self, root: MCTSNode) -> List[MCTSNode]:
        path = [root]
        curr = root
        while curr.children:
            curr = max(curr.children, key=lambda child: child.visits)
            path.append(curr)
        return path


# -------------------------------------------------------------------
# SIMULATION RUNNER
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Initializing Test-Time Compute (TTC) MCTS Reasoning Engine ---")
    
    generator = LLMStepGenerator()
    prm = ProcessRewardModel()
    engine = MCTSReasoningEngine(generator=generator, prm=prm, max_rollouts=25)

    prompt = "Problem: Solve for x in equation 3x + 6 = 132."
    print(f"Input Prompt: {prompt}\n")

    start_time = time.time()
    solution_path = engine.solve(prompt)
    elapsed = (time.time() - start_time) * 1000

    print("--- Optimal Trajectory Extracted via MCTS Search ---")
    print(solution_path)
    print(f"\nExecution Time: {elapsed:.2f}ms across 25 dynamic tree rollouts.")
```

## 4. Operational Best Practices

* Dynamic Rollout Budgets: Adjust test-time search rollouts based on query difficulty metrics (e.g., standard arithmetic gets $5$ rollouts, while competitive programming problems get $100+$ rollouts).
* Early Branch Pruning: Prune tree branches immediately if the per-step PRM score falls below a threshold ($\text{PRM}(s_t) < 0.15$), conserving FLOPs for high-probability trajectories.
* Prefix KV-Cache Reuse: Reuse pre-computed KV-cache states across common parent nodes during tree exploration to eliminate redundant autoregressive prefill compute.
