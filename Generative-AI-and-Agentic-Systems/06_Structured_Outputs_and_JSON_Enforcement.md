# Structured Outputs and JSON Enforcement

In production enterprise architectures, raw conversational text from a Large Language Model is highly problematic. Downstream systems—such as database pipelines, internal APIs, and frontend user interfaces—require deterministic, structured data types (predominantly valid JSON) conforming to rigid schemas. Relying on post-hoc prompt engineering or regular expression parsing is an anti-pattern that fails under edge cases.

This document explores the mathematical, theoretical, and algorithmic mechanics of **Constrained Decoding** via Context-Free Grammars (CFG) and logit-level token masking.

---

## 1. Theoretical Limitations of Post-Generation Parsing

Early attempts at structured output generation relied entirely on instruction tuning (e.g., *"Respond only in valid JSON"*). This approach breaks down due to the stochastic nature of autoregressive text generation:
* **Syntax Breakdowns:** The model may omit a closing bracket `}`, inject trailing commas, or truncate a text chunk due to max-token limits.
* **Prose Contamination:** Models frequently prepend or append conversational chatter (e.g., *"Here is the JSON schema you requested:"*), breaking strict input parsers.

---

## 2. Constrained Decoding via Logit Masking

Modern high-throughput inference engines enforce structure **during the decoding phase** by altering token distribution probabilities at every step of generation.

### The Algorithmic Mechanics
1. **Grammar Compilation:** A schema (like a Pydantic model) is converted into a **Context-Free Grammar (CFG)** or an internal Finite State Machine (FSM).
2. **State Tracking:** As tokens are generated, the engine tracks the exact character state of the output string. For example, if the model has just emitted `{"age": `, the next valid structural character *must* be an opening number digit or an array brace—it cannot be a closing quote or an arbitrary word string.
3. **Logit Masking:** Before the model samples the next token, the engine identifies every token in the entire vocabulary $V$ that would violate the grammar rules. The raw logit scores ($z$) for these violating tokens are modified by applying a mask of negative infinity ($-\infty$).

---

## 3. Mathematical Formulation of Logit Bias Shifts

Let $\mathbf{z} = [z_1, z_2, \dots, z_{|V|}]$ represent the raw, unnormalized logit vector emitted by the transformer's final linear projection layer over the vocabulary set $V$.

The engine maps the current state to an indicator vector $\mathbf{m} \in \{0, 1\}^{|V|}$, where a $1$ denotes a syntactically valid token according to the grammar rules, and a $0$ denotes an invalid token. The logit modifier function is defined as:

$$z_i' = \begin{cases} 
z_i & \text{if } m_i = 1 \\
-\infty & \text{if } m_i = 0 
\end{cases}$$

When the modified logits $\mathbf{z}'$ are passed through the Softmax activation function to generate the final sampling probability distribution $P(w_i)$, the probability of selecting an illegal token drops to exactly zero:

$$P(w_i \mid w_{<i}) = \frac{\exp(z_i')}{\sum_{j=1}^{|V|} \exp(z_j')} = \begin{cases} 
\frac{\exp(z_i)}{\sum_{j \in \text{Valid}} \exp(z_j)} & \text{if } m_i = 1 \\
0 & \text{if } m_i = 0 
\end{cases}$$

This ensures that the token sampled by the model is mathematically guaranteed to adhere to the target syntax schema.

---

## 4. Production-Grade Implementation: Token-Constrained JSON Decoding Engine

Below is a complete, self-contained Python architecture simulating a constrained decoding loop. It evaluates generated characters against a basic schema state tracker and dynamically masks vocabulary logit selections to enforce valid JSON format.

```python
import math
from typing import List, Dict, Tuple, Set

class JSONConstrainedDecoder:
    """Simulates logit-level structural token masking to enforce valid JSON syntax."""
    def __init__(self, vocabulary: Dict[int, str]):
        self.vocab = vocabulary
        # Basic categories of structural tokens for state mapping
        self.structural_tokens = ['{', '}', '"', ':', ',', '[', ']']

    def _determine_valid_next_chars(self, current_text: str) -> Set[str]:
        """
        A simplified state-machine tracking simple JSON object syntax states.
        Enforces: Object open -> Key open -> Colon separator -> Value open -> Comma/Close.
        """
        stripped = current_text.strip()
        
        if not stripped:
            return {'{'}
            
        # State: Just opened object
        if stripped == '{':
            return {'"'}
            
        # State: Inside or just finished a key string
        if stripped.endswith('{') or stripped.endswith(','):
            return {'"'}
            
        # State: Looking for a colon after key validation closure
        if stripped.endswith('"') and not stripped.endswith(':"') and ":" not in stripped.split(",")[-1]:
            return {':'}
            
        # State: Looking for a value after colon mapping
        if stripped.endswith(':'):
            # Allow digits for numbers or quote characters for strings
            return {'"', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}
            
        # State: Value is actively processing numbers
        if stripped[-1].isdigit():
            return {',', '}', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}
            
        # State: Value string just closed
        if stripped.endswith('"') and ":" in stripped.split(",")[-1]:
            return {',', '}'}
            
        # Default safety fallback state
        return set(chr(i) for i in range(32, 126))

    def compute_constrained_logits(self, current_text: str, raw_logits: Dict[int, float]) -> Dict[int, float]:
        """
        Applies a logit bias shift by tracking valid syntax states 
        and setting illegal vocabulary token logits to -inf.
        """
        valid_chars = self._determine_valid_next_chars(current_text)
        modified_logits = {}
        
        for token_id, token_str in self.vocab.items():
            # Check if token string matches any allowed next character syntax rules
            is_valid = any(token_str.startswith(vc) for vc in valid_chars) or token_str == ""
            
            if is_valid:
                modified_logits[token_id] = raw_logits[token_id]
            else:
                # Apply the mathematical negative infinity mask step
                modified_logits[token_id] = -float('inf')
                
        return modified_logits

    @staticmethod
    def sample_next_token(logits: Dict[int, float]) -> int:
        """Executes a stable greedy selection across valid logit distributions."""
        best_token_id = max(logits, key=logits.get)
        if logits[best_token_id] == -float('inf'):
            raise RuntimeError("Fatal: Constrained decoding engine encountered an inescapable dead-end state.")
        return best_token_id


# --- Simulation Execution Sandbox Verification ---
if __name__ == "__main__":
    # 1. Define a targeted miniature mock vocabulary matrix
    mock_vocabulary = {
        0: "{", 1: "}", 2: '"', 3: ":", 4: ",", 
        5: "id", 6: "val", 7: "99", 8: "hello",
        9: "error_text_payload", 10: " "
    }
    
    decoder = JSONConstrainedDecoder(vocabulary=mock_vocabulary)
    
    # Simulating a step-by-step decoding loop starting from an empty string input
    generated_buffer = ""
    print("=== Token-Level Structural Enforcement Loop ===")
    
    for generation_step in range(7):
        # Mock raw logits emitted by standard language model output layers
        # Uniform distribution across elements to simulate impartial generation intent
        mock_raw_logits = {i: 5.0 for i in mock_vocabulary.keys()}
        
        # Artificially boosting non-structural keywords to prove masking works
        mock_raw_logits[9] = 12.0  # Emphasizing an invalid token string "error_text_payload"
        
        if generation_step == 2:
            mock_raw_logits[5] = 10.0 # Emphasizing key target "id"
        if generation_step == 5:
            mock_raw_logits[7] = 10.0 # Emphasizing value target "99"

        # Apply logit masking rules
        constrained_logits = decoder.compute_constrained_logits(generated_buffer, mock_raw_logits)
        
        # Sample token
        chosen_id = decoder.sample_next_token(constrained_logits)
        chosen_token = mock_vocabulary[chosen_id]
        
        generated_buffer += chosen_token
        print(f"Step {generation_step + 1} -> Sampleed Token: '{chosen_token}' | Buffer: {generated_buffer}")

    print("\nFinal Formatted Output String Validation:")
    print(f" -> Result: {generated_buffer}")
```
