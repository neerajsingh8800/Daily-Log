# 24: LLM Observability Audit Logging and Forensic SIEM Integration

## 1. Overview & Threat Surface

Securing modern generative AI and autonomous agent applications requires moving beyond traditional application performance monitoring (APM) toward **AI-Native Observability, Structured Audit Logging, and Security Information and Event Management (SIEM) Integration**. Because LLM interactions are stochastic, multi-step, and stateful, traditional web access logs (which capture only HTTP methods, paths, and status codes) are insufficient for detecting prompt injections, tool hijacking, data exfiltration, or denial-of-wallet attacks.

An enterprise LLM deployment without deep forensic telemetry is blind to sophisticated adversarial maneuvering occurring within multi-turn conversations and autonomous agent execution traces.

### LLM Observability & SIEM Threat Telemetry Taxonomy

| Telemetry Dimension | Critical Log Fields | Security Detection Use Case |
| :--- | :--- | :--- |
| **Token Usage & Cost Metrics** | `prompt_tokens`, `completion_tokens`, `latency_ms`, `estimated_cost_usd` | Detecting Denial of Wallet (DoW) attacks and runaway infinite agent loops. |
| **Prompt & Completion Payloads** | `sanitized_prompt_hash`, `prompt_injection_flag`, `guardrail_status` | Identifying persistent jailbreak attempts and system prompt extraction vectors. |
| **Agent Tool Execution Trace** | `tool_name`, `tool_arguments`, `caller_role`, `execution_status` | Catching excessive agency violations, unauthorized privilege escalation, and tool abuse. |
| **Context & Retrieval Metadata** | `rag_chunks_retrieved`, `similarity_scores`, `datasource_ids` | Tracking vector database poisoning, data exfiltration, and sensitive data leakage. |

---

## 2. Threat Mechanics: Blind Spots in Traditional Logging

1. **The Black Box Problem:** A user interacts with an enterprise chatbot over 15 turns. On turn 14, an indirect prompt injection causes the agent to invoke an unauthorized database deletion tool. Traditional HTTP loggers record a single `200 OK` POST request to `/v1/chat/completions` with no visibility into the intermediate tool call payload.
2. **Obfuscated Token Costs:** Attackers flood a model with nested recursive requests or high-context padding. Without real-time token telemetry streaming to SIEM pipelines, security teams discover the financial breach only when monthly cloud billing alarms trigger.
3. **Missing Context Traceability:** In RAG pipelines, if sensitive internal documents are leaked in a response, standard logs do not record which exact vector chunks or enterprise files were retrieved and fed into the context window.

---

## 3. Defense Architecture for OpenTelemetry & SIEM Ingestion

To establish comprehensive forensic audit trails, applications must instrument every LLM call, guardrail inspection, and tool execution using OpenTelemetry (OTel) standards, exporting structured JSON security events to enterprise SIEM platforms (e.g., Splunk, Elastic, Datadog).

1. **Standardized Telemetry Schemas:** Embed unique session identifiers, user roles, token counts, and security evaluation flags into every log record.
2. **Real-Time SIEM Correlation Rules:** Write detection rules in the SIEM to flag anomalies (e.g., > 3 consecutive guardrail failures, tool execution by unverified roles, or token consumption spikes exceeding 10,000 tokens per minute).
3. **Payload Hashing & DLP Redaction:** Hash raw prompt inputs for indexing while scrubbing sensitive PII or secrets before sending logs to centralized log management platforms.

---

## 4. Hands-On Python Implementations

### Example 1: OpenTelemetry Structured LLM Audit Logger

```python
import json
import time
import hashlib
from typing import Dict, Any, Optional

class LLMAuditLogger:
    def __init__(self, service_name: str = "enterprise-llm-gateway"):
        self.service_name = service_name

    def _hash_payload(self, text: str) -> str:
        """Generates a secure SHA-256 hash of prompt payloads for forensic tracking without storing raw PII."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def create_audit_event(
        self,
        session_id: str,
        user_id: str,
        prompt: str,
        completion: str,
        prompt_tokens: int,
        completion_tokens: int,
        security_status: str,
        tool_invocations: Optional[list] = None
    ) -> str:
        """
        Constructs a structured OpenTelemetry-compliant JSON audit log for SIEM ingestion.
        """
        event = {
            "timestamp": time.time(),
            "service": self.service_name,
            "session_id": session_id,
            "user_id": user_id,
            "telemetry": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "estimated_cost_usd": ((prompt_tokens * 0.0015) + (completion_tokens * 0.0020)) / 1000.0
            },
            "security": {
                "status": security_status,  # e.g., "APPROVED", "BLOCKED_JAILBREAK", "TOOL_EXCESS"
                "prompt_hash": self._hash_payload(prompt),
            },
            "execution": {
                "tool_invocations": tool_invocations or []
            }
        }

        # Return formatted JSON log line ready for SIEM ingestion (Splunk/Elastic/Datadog)
        return json.dumps(event)

# Example Usage
if __name__ == "__main__":
    logger = LLMAuditLogger()

    # Simulated SIEM Audit Log generation for a secure LLM transaction
    audit_json = logger.create_audit_event(
        session_id="sess_991823",
        user_id="usr_4412",
        prompt="Summarize quarterly financial reports.",
        completion="Quarterly revenue increased by 14% year-over-year.",
        prompt_tokens=450,
        completion_tokens=120,
        security_status="APPROVED",
        tool_invocations=[{"tool": "fetch_financial_db", "status": "SUCCESS"}]
    )

    print("Structured SIEM Audit Event Payload:")
    print(audit_json)
```
