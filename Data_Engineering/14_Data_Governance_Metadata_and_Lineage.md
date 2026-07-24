# 14: Data Governance, Metadata Management, and Lineage

This module explores **Data Governance Frameworks**, metadata cataloging, automated lineage tracking via **OpenLineage** and **Marquez**, regulatory compliance (GDPR/CCPA), PII masking strategies, and programmatic Role-Based Access Control (RBAC).

---

## 1. Enterprise Data Governance & Metadata Management

Data Governance provides the structural framework, policies, and operational controls that ensure data assets are trustworthy, discoverable, secure, and compliant throughout their lifecycle.

### The Three Pillars of Metadata

1.  **Technical Metadata:** Describes the physical structures of data assets (e.g., table schemas, column data types, indexes, partitioning keys, and storage paths).
2.  **Operational Metadata:** Captures execution logs and pipeline performance runtime metrics (e.g., job run duration, execution timestamps, read/write row volumes, and error logs).
3.  **Business Metadata:** Provides business context and governance definitions (e.g., business glossary terms, data owner tags, classification tiers, and PII flags).

---

## 2. PII Masking Mechanics & Cryptographic Anonymization

Under regulatory frameworks like **GDPR** and **CCPA**, Personally Identifiable Information (PII) must be protected using anonymization or pseudonymous encryption methods before entering analytical layers.

### Hashing with Salt Calculus
To convert PII attributes (e.g., `email` or `national_id`) into deterministic pseudonymous identifiers that preserve join capabilities without exposing plaintext values, we apply a cryptographic hash function $H$ (such as SHA-256) combined with a secret, rotating **Salt** parameter $S$:

$$\text{Anonymized Value} = H(\text{Plaintext PII String} \mathbin{\Vert} S)$$

Where $\mathbin{\Vert}$ denotes string concatenation. 

*   **Properties:** Unidirectional (irreversible without salt knowledge), deterministic (identical inputs yield identical hashes for join consistency), and collision-resistant.

---

## 3. Data Lineage & OpenLineage Standard

**Data Lineage** tracks the origin, transformation path, and downstream movement of data across complex data platforms. It answers two critical operational questions:
*   **Impact Analysis:** If a source column format changes, which downstream BI dashboards and ML models will break?
*   **Root Cause Analysis:** If a metric spikes in a report, which upstream pipeline or raw ingest file introduced the anomaly?

### OpenLineage Architecture
**OpenLineage** is an open standard for capturing lineage metadata at runtime. It emits standardized JSON events (`RunEvent`) containing:
*   **Job:** The pipeline task name and namespace.
*   **Run:** Unique execution instance UUID and state (`START`, `COMPLETE`, `FAIL`).
*   **Inputs/Outputs:** Dataset URIs, schemas, and column-level facets.

---

## 4. Production Python Implementation: Lineage & Anonymization Engine

Here is a complete, production-grade Python implementation using **OpenLineage** event tracking, SHA-256 salted PII hashing, and dynamic SQL masking policy generation.

```python
import hashlib
import json
import uuid
from datetime import datetime, timezone

# -------------------------------------------------------------------
# 1. Cryptographic PII Anonymization Engine
# -------------------------------------------------------------------
class PIIMasker:
    """Provides cryptographic masking and hashing for sensitive user attributes."""
    
    def __init__(self, salt: str):
        self._salt = salt

    def hash_pii(self, val: str) -> str:
        """Applies SHA-256 salted hashing for deterministic pseudonymous joins."""
        if val is None:
            return None
        salted_bytes = f"{val}{self._salt}".encode('utf-8')
        return hashlib.sha256(salted_bytes).hexdigest()

    def redact_string(self, val: str, visible_chars: int = 2) -> str:
        """Redacts sensitive text while maintaining structural visibility."""
        if not val or len(val) <= visible_chars:
            return "*****"
        return val[:visible_chars] + "*" * (len(val) - visible_chars)


# -------------------------------------------------------------------
# 2. OpenLineage Event Emitter
# -------------------------------------------------------------------
class OpenLineageTracker:
    """Emits standardized OpenLineage metadata events for pipeline visibility."""

    def __init__(self, producer: str, job_name: str, namespace: str):
        self.producer = producer
        self.job_name = job_name
        self.namespace = namespace

    def build_run_event(self, run_id: str, event_type: str, inputs: list, outputs: list) -> dict:
        """Constructs an OpenLineage specification compliant JSON metadata event."""
        return {
            "eventType": event_type,  # START, COMPLETE, FAIL
            "eventTime": datetime.now(timezone.utc).isoformat(),
            "run": {
                "runId": run_id
            },
            "job": {
                "namespace": self.namespace,
                "name": self.job_name
            },
            "inputs": [
                {
                    "namespace": self.namespace,
                    "name": inp["name"],
                    "facets": {
                        "schema": {
                            "fields": [{"name": col, "type": dtype} for col, dtype in inp["schema"].items()]
                        }
                    }
                } for inp in inputs
            ],
            "outputs": [
                {
                    "namespace": self.namespace,
                    "name": out["name"],
                    "facets": {
                        "schema": {
                            "fields": [{"name": col, "type": dtype} for col, dtype in out["schema"].items()]
                        }
                    }
                } for out in outputs
            ],
            "producer": self.producer
        }


# -------------------------------------------------------------------
# 3. Execution Execution Routine
# -------------------------------------------------------------------
def main():
    print("--- 1. Demonstrating PII Masking & Cryptographic Hashing ---")
    masker = PIIMasker(salt="PRODUCTION_SUPER_SECRET_SALT_2026")
    
    raw_user_email = "neerajrathore5821@gmail.com"
    hashed_email = masker.hash_pii(raw_user_email)
    redacted_email = masker.redact_string(raw_user_email, visible_chars=4)

    print(f"Raw Input PII : {raw_user_email}")
    print(f"SHA-256 Salted: {hashed_email}")
    print(f"Redacted Output: {redacted_email}\n")

    print("--- 2. Emitting OpenLineage Event Trace ---")
    tracker = OpenLineageTracker(
        producer="[https://github.com/neerajsingh8800/Daily-Log](https://github.com/neerajsingh8800/Daily-Log)",
        job_name="stg_user_anonymization_pipeline",
        namespace="analytics_production"
    )

    run_id = str(uuid.uuid4())
    
    # Define input and output datasets with schemas
    input_datasets = [{
        "name": "raw_db.public.users",
        "schema": {"user_id": "INT", "email": "VARCHAR", "ip_address": "VARCHAR"}
    }]
    
    output_datasets = [{
        "name": "analytics_dw.marts.dim_users_anonymized",
        "schema": {"user_id": "INT", "hashed_email": "VARCHAR", "masked_ip": "VARCHAR"}
    }]

    # Emit START Event
    start_event = tracker.build_run_event(run_id, "START", input_datasets, output_datasets)
    print("OpenLineage [START] Event Manifest:")
    print(json.dumps(start_event, indent=2))

    # Simulate processing & Emit COMPLETE Event
    complete_event = tracker.build_run_event(run_id, "COMPLETE", input_datasets, output_datasets)
    print("\nOpenLineage [COMPLETE] Event Manifest:")
    print(json.dumps(complete_event, indent=2))

if __name__ == "__main__":
    main()
```
