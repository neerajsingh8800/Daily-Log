# 13: AI Governance, Compliance, and AIBOM

## 1. Overview & Regulatory Landscape

Deploying Artificial Intelligence and Machine Learning systems in enterprise environments requires strict compliance with international legal frameworks, security standards, and governance protocols. AI Governance bridges technical model validation with corporate risk management, legal compliance, and supply chain accountability.

Unlike traditional software software governance, AI governance must continuously track non-deterministic model behaviors, dataset provenance, fine-tuning lineages, and dynamic runtime safety parameters.

### Major AI Governance Frameworks

| Framework / Regulation | Geographic Scope | Focus Area | Core Requirement |
| :--- | :--- | :--- | :--- |
| **EU AI Act** | European Union | Risk-based classification (Unacceptable, High, Limited, Minimal) | Technical documentation, continuous risk management, post-market monitoring, human oversight. |
| **NIST AI RMF 1.0** | Global / US Standard | Framework across 4 functions: Govern, Map, Measure, Manage | Trustworthy AI characteristics (validity, reliability, safety, privacy, transparency). |
| **ISO/IEC 42001:2023** | International | AI Management System (AIMS) certification standard | Enterprise process integration, risk control implementation, continuous improvement. |
| **OWASP AIBOM** | Open Standard | Software Supply Chain Security for AI Components | Machine-readable bill of materials tracking datasets, base models, hyper-parameters, and dependencies. |

---

## 2. Threat Landscape & Compliance Risks

Inadequate governance introduces operational, security, and severe legal liabilities.

### Key Governance Risks

1. **Unregulated Supply Chain Dependencies (Shadow AI):** Deploying fine-tuned weights or third-party models without tracking licensing, data lineage, or latent backdoors.
2. **Regulatory Non-Compliance Fines:** Violating EU AI Act high-risk requirements (fines up to €35M or 7% of global annual turnover).
3. **Training Data Copyright & PII Contamination:** Ingesting copyrighted materials or un-redacted PII into training or fine-tuning datasets, creating privacy violations under GDPR/CCPA.
4. **Lack of Explainability & Auditability:** Inability to produce deterministic logging or model decision rationale during regulatory inquiries or post-incident analysis.

---

## 3. AI Bill of Materials (AIBOM) Architecture

An **AI Bill of Materials (AIBOM)** extends traditional Software Bill of Materials (SBOM) standards (such as CycloneDX 1.6) to record AI-specific artifacts.

An AIBOM must explicitly record:
* **Model Provenance:** Parent model name, base weights cryptographic hash (SHA-256), quantization status (e.g., GGUF 4-bit, FP16).
* **Dataset Lineage:** Training/fine-tuning dataset sources, checksums, data cleaning steps, license types (e.g., Apache 2.0, MIT, CC-BY-SA).
* **Environment Dependencies:** Exact framework versions (PyTorch, Transformers, vLLM, CUDA drivers).
* **Attestation Signatures:** Digital signatures certifying safety evaluations, bias checks, and human review sign-offs.

---

## 4. NIST AI RMF & EU AI Act Implementation Mechanics

### Risk Classification Matrix (EU AI Act)

To comply with **High-Risk AI System** mandates, an organization must establish an automated governance pipeline that checks:
1. **Pre-Training Gate:** Dataset licensing verification and PII scanning.
2. **Pre-Deployment Gate:** Automated model evaluation for accuracy, toxic output rate, and drift metrics.
3. **Runtime Monitoring:** Logging prompt/completion metadata, dynamic risk triggers, and manual escalation paths.

---

## 5. Hands-On Python Implementations

### Example 1: Automated CycloneDX 1.6 AIBOM Generator for ML Artifacts

```python
import hashlib
import json
import os
import datetime
from typing import Dict, Any, List

class AIBOMGenerator:
    def __init__(self, system_name: str, version: str):
        self.system_name = system_name
        self.version = version
        self.components: List[Dict[str, Any]] = []

    def _hash_file(self, file_path: str) -> str:
        """Computes SHA-256 checksum of model weights or dataset files."""
        if not os.path.exists(file_path):
            return "FILE_NOT_FOUND"
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def add_model_component(
        self, 
        model_name: str, 
        model_path: str, 
        architecture: str, 
        license_type: str,
        quantization: str = "FP16"
    ):
        """Adds model artifact entry to the AIBOM."""
        file_hash = self._hash_file(model_path)
        component = {
            "type": "machine-learning-model",
            "name": model_name,
            "version": self.version,
            "licenses": [{"license": {"id": license_type}}],
            "hashes": [{"alg": "SHA-256", "content": file_hash}],
            "modelParameters": {
                "architecture": architecture,
                "quantization": quantization
            }
        }
        self.components.append(component)

    def add_dataset_component(
        self, 
        dataset_name: str, 
        dataset_path: str, 
        source_url: str, 
        pii_scrubbed: bool,
        license_type: str
    ):
        """Adds training/fine-tuning dataset entry to the AIBOM."""
        file_hash = self._hash_file(dataset_path)
        component = {
            "type": "data",
            "name": dataset_name,
            "version": self.version,
            "licenses": [{"license": {"id": license_type}}],
            "hashes": [{"alg": "SHA-256", "content": file_hash}],
            "externalReferences": [{"type": "distribution", "url": source_url}],
            "properties": [
                {"name": "pii_scrubbed", "value": str(pii_scrubbed)},
                {"name": "data_type", "value": "fine-tuning-jsonl"}
            ]
        }
        self.components.append(component)

    def generate_cyclonedx_bom(self) -> Dict[str, Any]:
        """Exports CycloneDX 1.6 compliant JSON AIBOM representation."""
        bom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "serialNumber": f"urn:uuid:{hashlib.md5(self.system_name.encode()).hexdigest()}",
            "version": 1,
            "metadata": {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "component": {
                    "type": "application",
                    "name": self.system_name,
                    "version": self.version
                }
            },
            "components": self.components
        }
        return bom

# Example Usage Demonstration
if __name__ == "__main__":
    generator = AIBOMGenerator("Enterprise-CustomerSupport-LLM", "v2.1.0")
    
    # Register Base Model
    generator.add_model_component(
        model_name="Llama-3-8B-Instruct",
        model_path="./llama3_weights.bin", # Dummy path for demo
        architecture="Transformer-Decoder",
        license_type="Llama-3-Community",
        quantization="Q4_K_M"
    )
    
    # Register Fine-Tuning Dataset
    generator.add_dataset_component(
        dataset_name="Customer_Support_Sanitized_2026",
        dataset_path="./dataset.jsonl", # Dummy path for demo
        source_url="s3://internal-ai-vault/datasets/cs_2026.jsonl",
        pii_scrubbed=True,
        license_type="Proprietary"
    )
    
    aibom_json = generator.generate_cyclonedx_bom()
    print("Generated CycloneDX 1.6 AIBOM Document:\n")
    print(json.dumps(aibom_json, indent=2))
```

