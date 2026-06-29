# Production Observability and Trace Analysis

Debugging a standard web application involves looking at linear stack traces. Debugging an autonomous agent, however, requires dissecting an asynchronous, non-deterministic graph of loop iterations, tool calls, and LLM completions. If an agent fails on step 7 of an execution loop, traditional logging falls short. You need structured, graph-aware tracing.

---

## 1. Theoretical Foundations: Spans, Traces, and Directed Acyclic Graphs (DAGs)

To monitor complex multi-agent or ReAct loops in production, observability frameworks structure execution data using distributed tracing principles:

### Key Concepts
* **Trace:** The complete end-to-end journey of a user request through the agent system.
* **Span:** A single unit of contiguous work within a trace (e.g., an LLM call, a tool execution, or a vector database lookup). Spans contain start/end timestamps, status codes, and custom metadata attributes.
* **DAG Representation:** Multi-agent architectures form Directed Acyclic Graphs where parent-child relationships map out exactly which agent spawned another or which thought prompted a tool call.

---

## 2. Mathematical Formalization: Aggregated Latency and Cost Accumulation

In multi-step loops, calculating performance requires tracking both synchronous dependencies and parallel executions across sub-spans.

### A. Critical Path Latency
For an agent trace containing a set of sequential or parallel spans $S$, the total system execution latency $L_{\text{total}}$ is governed by the length of the **Critical Path** (the longest path of dependent operations):

$$L_{\text{total}} = \max_{P \in \text{Paths}} \sum_{s \in P} \Delta t_s$$

Where $\Delta t_s = t_{\text{end}, s} - t_{\text{start}, s}$ is the duration of an individual span $s$.

### B. Cumulative Stepwise Token Cost
The financial footprint of a complex agent trajectory across $M$ distinct LLM interaction points within a trace is computed dynamically as:

$$\text{Cost}_{\text{trace}} = \sum_{j=1}^{M} \left( P_{\text{in}} \cdot X_{\text{in}, j} + P_{\text{out}} \cdot Y_{\text{out}, j} \right)$$

Where:
* $P_{\text{in}}, P_{\text{out}}$ are the pricing rates per unit token for input and output components.
* $X_{\text{in}, j}, Y_{\text{out}, j}$ are the count of context window input tokens and generated output tokens for completion span $j$.

---

## 3. Production Implementation: Structured Trajectory Tracing Engine

Below is a complete, production-grade Python tracing architecture designed to build nested telemetry records for complex agent loops without external third-party service dependencies.

```python
import time
import json
import uuid
from typing import List, Dict, Any, Optional

class AgentSpan:
    def __init__(self, name: str, span_type: str, parent_id: Optional[str] = None):
        self.span_id: str = str(uuid.uuid4())[:8]
        self.parent_id: Optional[str] = parent_id
        self.name: str = name
        self.span_type: str = span_type
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None
        self.attributes: Dict[str, Any] = {}
        self.status: str = "RUNNING"

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def end(self, status: str = "SUCCESS") -> None:
        self.end_time = time.time()
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "type": self.span_type,
            "duration_ms": round((self.end_time - self.start_time) * 1000, 2) if self.end_time else 0,
            "status": self.status,
            "attributes": self.attributes
        }

class AgentTrackerTelemetry:
    def __init__(self):
        self.trace_id: str = str(uuid.uuid4())[:12]
        self.all_spans: List[AgentSpan] = []
        self.span_stack: List[str] = []

    def start_span(self, name: str, span_type: str) -> AgentSpan:
        """
        Starts a new execution span, automatically linking it to its parent span if nested.
        """
        parent_id = self.span_stack[-1] if self.span_stack else None
        span = AgentSpan(name, span_type, parent_id)
        self.all_spans.append(span)
        self.span_stack.append(span.span_id)
        return span

    def end_current_span(self, status: str = "SUCCESS", attributes: Optional[Dict[str, Any]] = None) -> None:
        """
        Closes out the active execution span and pops it off the nesting tracker hierarchy.
        """
        if not self.span_stack:
            return
        
        current_id = self.span_stack.pop()
        for span in self.all_spans:
            if span.span_id == current_id:
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, v)
                span.end(status)
                break

    def export_trace_dag(self) -> str:
        """
        Compiles the collected linear spans into a structured JSON execution graph.
        """
        return json.dumps({
            "trace_id": self.trace_id,
            "total_spans": len(self.all_spans),
            "spans": [span.to_dict() for span in self.all_spans]
        }, indent=2)

# --- Execution Verification Loop ---
if __name__ == "__main__":
    telemetry = AgentTrackerTelemetry()
    
    print("[1] Initialization: Root trace window opened...")
    # Root Operation
    telemetry.start_span(name="Execute_Financial_Analysis", span_type="agent_root")
    time.sleep(0.05)
    
    # Nested Action A: Vector Memory Search
    telemetry.start_span(name="Fetch_Historical_Context", span_type="memory_retrieval")
    time.sleep(0.02)
    telemetry.end_current_span(attributes={"collection": "user_preferences", "matches_found": 3})
    
    # Nested Action B: ReAct Execution Loop
    telemetry.start_span(name="ReAct_Core_Planning", span_type="llm_reasoning")
    time.sleep(0.08)
    
    # Sub-Nested Tool Execution within ReAct Loop
    telemetry.start_span(name="Execute_SQL_Query", span_type="tool_execution")
    time.sleep(0.04)
    telemetry.end_current_span(status="SUCCESS", attributes={"rows_returned": 42, "target_table": "transactions"})
    
    # End ReAct Core Span
    telemetry.end_current_span(attributes={"input_tokens": 4200, "output_tokens": 310})
    
    # Close Root Trace Window
    telemetry.end_current_span()
    
    print("\n=== Exporting Production Trace Execution Graph ===")
    print(telemetry.export_trace_dag())
```
