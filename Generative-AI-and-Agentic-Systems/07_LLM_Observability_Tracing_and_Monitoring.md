# LLM Observability: Tracing and Monitoring

Transitioning Large Language Models (LLMs) from localized sandboxes to multi-step enterprise pipelines introduces massive operational visibility challenges. Because LLM generations are non-deterministic and chain together asynchronous components (retrievers, vector indexes, cross-encoders, and API boundaries), standard application logging (APM) fails. Debugging requires tracking hierarchical call structures, execution velocities, and granular cost vectors.

This document explores the structural, mathematical, and telemetry frameworks necessary to achieve production-grade observability across complex Generative AI pipelines.

---

## 1. The Core Metrics Matrix

To monitor performance accurately, an observability engine tracks operational efficiency across three foundational categories:

### I. Time to First Token (TTFT)
* **Definition:** The duration between submitting a prompt to the network and receiving the absolute first generated token fragment.
* **System Component:** Directly measures the efficiency of the **Prefill Phase**. High TTFT points to network transfer delays or overloaded GPU compute queues processing oversized context inputs.

### II. Inter-Token Latency (ITL)
* **Definition:** The average time elapsed between generating each subsequent token inside an active stream.
* **System Component:** Directly measures the efficiency of the **Decode Phase**. High ITL indicates memory-bandwidth bottlenecks, unoptimized serving engines, or inefficient key-value caching allocations.

### III. Input vs. Output Token Footprints
* **Definition:** Absolute tracking of input token counts ($N_{\text{in}}$) vs. output token counts ($N_{\text{out}}$). 
* **System Component:** Directly maps financial resource consumption and efficiency configurations.

---

## 2. Mathematical Framework for Performance Analysis

To extract actionable telemetry from systemic traces, we apply explicit cost and speed equations.

### Token Processing Velocity
The overall generation throughput (velocity $V$) measured in tokens per second is formulated as:

$$V = \frac{N_{\text{out}} - 1}{\text{Total Latency} - \text{TTFT}}$$

### Total Cost Accumulation
Given localized market rates for token units (where $C_{\text{in}}$ is the cost per million input tokens and $C_{\text{out}}$ is the cost per million output tokens), the financial cost $\mathcal{C}$ for a single isolated pipeline execution step is mapped as:

$$\mathcal{C} = \left( \frac{N_{\text{in}} \cdot C_{\text{in}}}{10^6} \right) + \left( \frac{N_{\text{out}} \cdot C_{\text{out}}}{10^6} \right)$$

### The Graph Trace Topology
Multi-step pipelines (like RAG systems or agent routing configurations) are modeled as a **Directed Acyclic Graph (DAG)** of spans. Each execution unit is encapsulated in a span containing a unique `span_id`, a structural parent reference pointer `parent_span_id`, timestamps, and a semantic metadata map.

---

## 3. Production-Grade Implementation: Decorator-Driven Hierarchical Tracing Engine

Below is a complete, self-contained Python observability framework. It uses context managers and function decorators to automatically calculate TTFT, track token volumes, compute execution pricing, and output structured, hierarchical span dependency trees.

```python
import time
import uuid
import random
from typing import List, Dict, Any, Optional

class TraceLogger:
    """Central registry responsible for assembling and rendering execution trace trees."""
    def __init__(self):
        self.spans: Dict[str, Dict[str, Any]] = {}

    def start_span(self, name: str, parent_id: Optional[str] = None) -> str:
        span_id = str(uuid.uuid4())
        self.spans[span_id] = {
            "span_id": span_id,
            "parent_id": parent_id,
            "name": name,
            "start_time": time.time(),
            "end_time": None,
            "metadata": {}
        }
        return span_id

    def end_span(self, span_id: str, metadata: Optional[Dict[str, Any]] = None):
        if span_id in self.spans:
            self.spans[span_id]["end_time"] = time.time()
            if metadata:
                self.spans[span_id]["metadata"].update(metadata)

    def print_trace_tree(self, current_id: Optional[str] = None, indent: int = 0):
        """Recursively parses parent pointers to output an industry-standard visual trace hierarchy."""
        for s_id, span in self.spans.items():
            if span["parent_id"] == current_id:
                duration = 0.0
                if span["end_time"] and span["start_time"]:
                    duration = span["end_time"] - span["start_time"]
                
                meta_str = f" | Meta: {span['metadata']}" if span["metadata"] else ""
                print(f"{'  ' * indent}└── 🟩 [{span['name']}] Duration: {duration:.4f}s{meta_str}")
                
                # Recurse down to discover children linked to this span node
                self.print_trace_tree(current_id=s_id, indent=indent + 2)


# Instantiate a global registry tracker object
trace_registry = TraceLogger()
active_span_stack: List[str] = []

def observed_span(span_name: str):
    """Decorator to automatically wrap complex pipeline function blocks into trace spans."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            parent_id = active_span_stack[-1] if active_span_stack else None
            span_id = trace_registry.start_span(name=span_name, parent_id=parent_id)
            active_span_stack.append(span_id)
            
            try:
                result = func(*args, **kwargs)
                # If the function returns execution metadata, capture it dynamically
                meta = result if isinstance(result, dict) and "tokens_in" in result else None
                trace_registry.end_span(span_id, metadata=meta)
                return result
            finally:
                active_span_stack.pop()
        return wrapper
    return decorator


# --- Mocking Pipeline Components with Instrumented Telemetry ---

@observed_span("Stage-1 Retrieval Sweep")
def mock_retrieval_step() -> Dict[str, Any]:
    time.sleep(0.15) # Simulating database query network latencies
    return {"retrieved_chunks": 5}

@observed_span("Stage-2 Cross-Encoder Rerank")
def mock_rerank_step():
    time.sleep(0.08) # Simulating dynamic scoring compute cycles

@observed_span("LLM Auto-Regressive Generation")
def mock_llm_generation() -> Dict[str, Any]:
    # Phase 1: Simulate the compute-heavy Prefill latency loop (TTFT)
    ttft_latency = 0.12
    time.sleep(ttft_latency)
    
    # Phase 2: Simulate streaming decode cycles
    input_tokens = 450
    output_tokens = 60
    
    # Cost parameters for calculation: e.g., $1.50/M input, $5.00/M output tokens
    c_in_rate = 1.50
    c_out_rate = 5.00
    financial_cost = ((input_tokens * c_in_rate) / 1e6) + ((output_tokens * c_out_rate) / 1e6)
    
    # Accumulate inter-token generation delays
    time.sleep(0.18) 
    
    return {
        "tokens_in": input_tokens,
        "tokens_out": output_tokens,
        "ttft_seconds": round(ttft_latency, 4),
        "computed_cost_usd": f"${financial_cost:.6f}"
    }

@observed_span("Complete RAG Execution Flow")
def run_monitored_rag_pipeline():
    """Root function organizing pipeline stages sequentially under one parent trace trace context."""
    mock_retrieval_step()
    mock_rerank_step()
    metrics = mock_llm_generation()
    return metrics


# --- Operational Verification Sandbox ---
if __name__ == "__main__":
    print("=== Executing Live Monitored RAG Application Pipeline ===")
    run_monitored_rag_pipeline()
    
    print("\n=== Accumulated Telemetry Graph Tree Output ===")
    trace_registry.print_trace_tree()
```
