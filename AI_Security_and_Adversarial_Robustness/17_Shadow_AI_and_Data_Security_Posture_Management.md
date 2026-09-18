# 17: Shadow AI, AI-SPM, and Data Security Posture Management

## 1. Overview & Threat Surface

The rapid adoption of Generative AI across enterprise environments has introduced significant unmanaged risk. Employees frequently paste proprietary source code, internal financial records, and PII into unvetted consumer AI platforms (**Shadow AI**). Simultaneously, engineering teams spin up unofficial fine-tuning pipelines or API connections using unmonitored keys without central security oversight.

To address these vulnerabilities, organizations deploy **AI Security Posture Management (AI-SPM)** and **Data Security Posture Management (DSPM)**. These frameworks continuously discover unmanaged AI models, map data flows from sensitive datastores into model training pipelines, enforce egress data loss prevention (DLP), and manage the lifecycle of AI credentials.

### Core Security Paradigms

| Paradigm | Focus Area | Core Responsibilities |
| :--- | :--- | :--- |
| **Shadow AI Discovery** | Unsanitized End-User & API Activity | Identifying unauthorized LLM web usage, rogue API keys, and unmapped cloud model deployments. |
| **AI-SPM (AI Security Posture Management)** | Asset & Infrastructure Visibility | Mapping model inventories, tracking model lineage, validating configuration drift, and enforcing API key rotation. |
| **DSPM for AI (Data Security Posture Management)** | Training & Context Data Protection | Classifying data fed into fine-tuning datasets, detecting raw PII in vector indexes, and preventing sensitive egress. |

---

## 2. Threat Mechanics: Unsanitized Egress & Model Misconfigurations

1. **Unmonitored API Key Sprawl:** An engineering team embeds hardcoded API keys for external models into local test scripts, leading to unmonitored, non-compliant third-party processing of sensitive data.
2. **Training Data Contamination (DSPM Risk):** Internal document repositories containing raw employee HR records are fed into an unencrypted RAG pipeline without proper role-based access control (RBAC).
3. **Data Exfiltration via Prompt Egress:** Employees copy trade secrets into consumer AI tools, exposing proprietary intellectual property to third-party vendor training regimes.

---

## 3. Defense Architecture for Enterprise AI Governance

1. **Inline Egress Proxy:** Intercept all outbound web traffic destined for known AI endpoints to block unauthorized payload transfers.
2. **Automated AI-SPM Asset Discovery:** Regularly scan cloud accounts (AWS, Azure, GCP) to enumerate hosted models, endpoints, and vector databases.
3. **DSPM Data Pipeline Guardrails:** Inspect training data buckets and vector embedding inputs to sanitize sensitive fields before ingestion.

---

## 4. Hands-On Python Implementations

### Example 1: Shadow AI Inline Egress Proxy & DLP Inspector

```python
import re
import json
from typing import Dict, Any, Tuple

class SecurityPolicyException(Exception):
    pass

class ShadowAIEgressProxy:
    def __init__(self):
        # Known blocked consumer/unvetted AI endpoints
        self.blocked_endpoints = [
            "api.unapproved-ai-vendor.com",
            "consumer-chat.ai/v1/completion"
        ]
        
        # Regex rules for enterprise PII & sensitive data
        self.sensitive_patterns = {
            "api_key": re.compile(r'(?i)(sk-[a-zA-Z0-9]{32,})'),
            "credit_card": re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
        }

    def Inspect_outbound_request(self, target_url: str, request_body: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Inspects outbound requests to detect Shadow AI usage and sensitive data exfiltration.
        """
        # 1. Check against endpoint policy
        for blocked_domain in self.blocked_endpoints:
            if blocked_domain in target_url:
                raise SecurityPolicyException(f"BLOCKED: Destination '{target_url}' is an unapproved Shadow AI service.")

        # 2. Inspect payload content for sensitive data egress
        payload_str = json.dumps(request_body)
        for data_type, pattern in self.sensitive_patterns.items():
            if pattern.search(payload_str):
                raise SecurityPolicyException(f"BLOCKED: Outbound payload contains sensitive data ({data_type}).")

        return True, "Request Authorized"

# Example Usage
if __name__ == "__main__":
    proxy = ShadowAIEgressProxy()

    # Attempt 1: Unauthorized Endpoint
    try:
        proxy.Inspect_outbound_request("[https://api.unapproved-ai-vendor.com/prompt](https://api.unapproved-ai-vendor.com/prompt)", {"prompt": "Hello"})
    except SecurityPolicyException as e:
        print(f"Intercepted Attempt 1: {e}")

    # Attempt 2: Exfiltration of API Key
    try:
        proxy.Inspect_outbound_request("[https://approved-api.openai.com/v1/chat](https://approved-api.openai.com/v1/chat)", {"prompt": "My key is sk-abc123xyz45678901234567890123456"})
    except SecurityPolicyException as e:
        print(f"Intercepted Attempt 2: {e}")
```

