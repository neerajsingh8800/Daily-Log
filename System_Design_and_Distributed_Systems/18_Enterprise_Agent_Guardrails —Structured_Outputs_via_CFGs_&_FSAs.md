# Module 18: Enterprise Agent Guardrails — Structured Outputs via CFGs & FSAs

In enterprise production environments, Large Language Models (LLMs) driving automated workflows must strictly produce deterministic, syntactically valid outputs (such as JSON schemas, SQL queries, or API payloads). Relying solely on prompt instructions or soft-retry validation loops is insufficient due to non-zero parsing failure rates.

Modern enterprise guardrails guarantee *100% syntactically correct structured generation* at the sampling level. By compiling formal Context-Free Grammars (CFGs) or regular expressions into Finite State Automata (FSAs) and dynamic Pushdown Automata (PDAs), inference engines dynamically modify the model's output logit vectors at each decoding step $t$, setting probability scores for invalid tokens to $-\infty$.

---

## 1. Theoretical Foundations

### 1.1 Constrained Decoding & Logit Masking Mechanics

* **Guaranteed Zero-Syntax Generation**:
  * Instead of evaluating output syntax *after* generation, guided decoding enforces grammatical constraints *during* token sampling.
  * Ensures LLM agents produce well-formed structured data (Pydantic objects, JSON schemas, SQL statements) without parsing exceptions.

* **Finite State Automata (FSA) & Pushdown Automata (PDA)**:
  * **FSA (Regular Languages / Regex)**: Manages fixed-structure constraints (e.g., date formats, email strings, specific JSON field types).
  * **PDA / CFG (Context-Free Languages)**: Manages nested structures (e.g., recursive JSON objects, arrays, nested SQL queries) using a stack mechanism to track state transitions.

* **Logit Masking Execution Loop**:
  1. At step $t$, the LLM predicts an unconstrained logit vector $z_t \in \mathbb{R}^{\vert{}V\vert{}}$ over vocabulary $V$.
  2. The FSA/PDA inspects its current state $q_t$ and identifies the set of valid next tokens $V_{\text{valid}}(q_t) \subseteq V$.
  3. A logit mask $M_t \in \{0, -\infty\}^{\vert{}V\vert{}}$ is generated, where $M_{t, i} = 0$ for $i \in V_{\text{valid}}(q_t)$ and $-\infty$ otherwise.
  4. Softmax is computed over $z_t + M_t$, restricting probability distribution strictly to valid tokens.

---

### 1.2 Mathematical Foundations

#### 1. Constrained Softmax Probability Distribution
Given raw unconstrained logit values $z_t(v)$ for each vocabulary token $v \in V$, state $q_t \in Q$, and set of valid next tokens $V_{\text{valid}}(q_t) \subseteq V$:

$$P(x_t = v \mid x_{<t}, q_t) = \begin{cases} \frac{\exp(z_t(v))}{\sum_{u \in V_{\text{valid}}} \exp(z_t(u))}, & \text{if } v \in V_{\text{valid}} \\ 0, & \text{otherwise} \end{cases}$$

#### 2. Pushdown Automaton State Transition Mapping
A Pushdown Automaton enforcing a Context-Free Grammar is defined as a tuple $M = (Q, \Sigma, \Gamma, \delta, q_0, Z_0, F)$, where:

* $Q$: Finite set of control states.
* $\Sigma$: Alphabet of valid input tokens ($V$).
* $\Gamma$: Stack alphabet.
* $\delta$: Transition function $Q \times (\Sigma \cup \{\epsilon\}) \times \Gamma \rightarrow P(Q \times \Gamma^*)$.

The valid token set $V_{\text{valid}}(q_t, \gamma_t)$ for state $q_t$ and stack top $\gamma_t \in \Gamma$ is:

$$V_{\text{valid}}(q_t, \gamma_t) = \{ v \in \Sigma \mid \exists (q', \alpha) \in \delta(q_t, v, \gamma_t) \}$$

---

## 2. Structured Output Enforcement Comparison

| Strategy | Failure Rate | Processing Overhead | Backtracking | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Prompt Engineering (Few-Shot)** | High (~5-15%) | 0 | No | Quick prototyping |
| **JSON Mode (Soft-Retry)** | Low (~1-3%) | High (Re-runs full forward pass) | No | Simple JSON payloads |
| **Logit Masking via FSA/CFG** | **0.00% (Guaranteed)** | Low (Per-step state lookup) | No | Mission-critical enterprise workflows |
| **Speculative Decoding + Grammar** | **0.00% (Guaranteed)** | Very Low (Parallel validation) | Yes | High-throughput production APIs |

---

## 3. Production Constrained Decoding Guardrail Implementation

This Python module implements a **Token-Level Finite State Automaton (FSA) Guardrail** that forces an LLM logit sampler to strictly generate JSON-compliant key-value schemas.

### Prerequisites

```bash
pip install numpy pydantic
```

### Python Implementation (guided_decoding_fsa.py)
```python
import math
import numpy as np
from typing import List, Dict, Set, Optional

# -------------------------------------------------------------------
# 1. FINITE STATE AUTOMATON (FSA) FOR JSON SCHEMAS
# -------------------------------------------------------------------
class JSONAutomaton:
    """A State Machine enforcing strict JSON formatting rules at token boundaries."""
    def __init__(self):
        # States: 0=START, 1=OPEN_BRACE, 2=KEY_STRING, 3=COLON, 4=VALUE_STRING, 5=CLOSE_BRACE
        self.state = 0

    def get_valid_tokens(self, vocab: Dict[str, int]) -> Set[int]:
        valid_ids = set()
        for token_text, token_id in vocab.items():
            if self.state == 0 and token_text == "{":
                valid_ids.add(token_id)
            elif self.state == 1 and token_text.startswith('"') and token_text.endswith('"'):
                valid_ids.add(token_id)
            elif self.state == 2 and token_text == ":":
                valid_ids.add(token_id)
            elif self.state == 3 and (token_text.isdigit() or (token_text.startswith('"') and token_text.endswith('"'))):
                valid_ids.add(token_id)
            elif self.state == 4 and token_text == "}":
                valid_ids.add(token_id)
        return valid_ids

    def step(self, token_text: str):
        if self.state == 0 and token_text == "{":
            self.state = 1
        elif self.state == 1 and token_text.startswith('"'):
            self.state = 2
        elif self.state == 2 and token_text == ":":
            self.state = 3
        elif self.state == 3:
            self.state = 4
        elif self.state == 4 and token_text == "}":
            self.state = 5


# -------------------------------------------------------------------
# 2. LOGIT MASKING ENGINE
# -------------------------------------------------------------------
class ConstrainedLogitProcessor:
    def __init__(self, vocab: Dict[str, int], automaton: JSONAutomaton):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.automaton = automaton

    def process_logits(self, logits: np.ndarray) -> np.ndarray:
        valid_token_ids = self.automaton.get_valid_tokens(self.vocab)
        masked_logits = np.full_like(logits, fill_value=-np.inf)
        for t_id in valid_token_ids:
            masked_logits[t_id] = logits[t_id]
        return masked_logits


# -------------------------------------------------------------------
# 3. SIMULATED LLM SAMPLING LOOP WITH GUARDRAIL MASKING
# -------------------------------------------------------------------
class GuardrailedLLM:
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}

    def predict_logits(self) -> np.ndarray:
        """Simulates raw unconstrained LLM logits."""
        return np.random.randn(len(self.vocab))

    def generate_constrained_json(self, logit_processor: ConstrainedLogitProcessor) -> str:
        generated_tokens = []
        while logit_processor.automaton.state != 5:
            # 1. Compute raw logits
            raw_logits = self.predict_logits()
            # 2. Apply FSA Constrained Logit Masking
            masked_logits = logit_processor.process_logits(raw_logits)
            # 3. Compute Softmax
            exp_logits = np.exp(masked_logits - np.max(masked_logits))
            probs = exp_logits / np.sum(exp_logits)
            # 4. Sample token
            selected_id = int(np.random.choice(len(self.vocab), p=probs))
            selected_text = self.inv_vocab[selected_id]
            # 5. Advance FSA State Machine
            logit_processor.automaton.step(selected_text)
            generated_tokens.append(selected_text)
        return "".join(generated_tokens)


# -------------------------------------------------------------------
# SIMULATION / VERIFICATION RUNNER
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("--- 1. Initializing Token Vocabulary & FSA Automaton ---")
    mock_vocab = {
        "{": 0, "}": 1, ":": 2, ",": 3,
        '"status"': 4, '"code"': 5, '"success"': 6,
        "200": 7, "500": 8, "INVALID_TOKEN": 9
    }
    fsa = JSONAutomaton()
    processor = ConstrainedLogitProcessor(vocab=mock_vocab, automaton=fsa)
    llm = GuardrailedLLM(vocab=mock_vocab)

    print("\n--- 2. Executing Logit-Masked Guided Generation Loop ---")
    output_json = llm.generate_constrained_json(processor)
    print(f"\nGuaranteed Valid JSON Payload: {output_json}")
    print("Zero JSON decoding exceptions guaranteed by logit masking!")
```

## 4. Operational Best Practices

* Pre-compile Grammar Indexes: Convert Pydantic schemas or CFG specifications into FSA state transition tables at server initialization to eliminate compilation latency during request processing.
* Trie-Based Vocabulary Indexing: Use prefix Tries over the tokenizer vocabulary to quickly identify all valid token IDs for a given state prefix string in $O(K)$ time.
* Combine with Context Caching: Store the prefix grammar state alongside the KV-cache state to enable instant resumption of structured generation across dynamic multi-turn agent conversations.
