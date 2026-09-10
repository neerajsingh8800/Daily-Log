# 09: Supply Chain Security for ML Artifacts — Model Serialization, Provenance & SBOMs

Machine learning models and AI pipelines rely heavily on external model hubs, serialized weights, pre-built tokenizers, and open-source datasets. **ML Supply Chain Attacks** occur when adversaries compromise model files, insert malicious execution payloads during serialization, or tamper with dataset registries.

Because legacy model formats like PyTorch `.pt`/`.pth` or Pickle `.pkl` rely on arbitrary Python code execution during deserialization (`__reduce__` exploit vectors), securing the ML pipeline requires cryptographic signing, zero-trust artifact verification, Software Bill of Materials (SBOM) tracking, and safe binary formats like Safetensors.

This module covers deserialization vulnerabilities, Model Signature Verification, ML-SBOM standards (CycloneDX/SPDX), and a production-grade Python ML Supply Chain Auditor.

---

## 1. Theoretical Foundations

### 1.1 The Pickle Deserialization Vulnerability (`__reduce__`)

The standard Python `pickle` module (and by extension, legacy `torch.load` implementations) serializes objects by recording instructions for an unpickler machine to reconstruct the object graph.

* **Mechanics**: An attacker implements the `__reduce__` magic method on a custom class. Upon deserialization, Python invokes the callable returned by `__reduce__` with user-supplied arguments.
* **Exploit Vector**: Attacker injects `os.system` or `subprocess.Popen` inside `__reduce__`. When a user loads `model.pkl` or `weights.bin`, arbitrary shell commands run immediately on the host before any weight values are passed to GPU VRAM.

* ### 1.2 Zero-Trust Serialization & Provenance Tracking

To mitigate supply chain risks, modern AI systems adopt a **Zero-Trust Artifact Pipeline**:

1. **Safe Serialization (`safetensors`)**: Replaces executable pickle files with pure byte-buffer representations. Safetensors prevents code execution by restricting the file structure to raw numerical arrays and JSON header metadata.
2. **Cryptographic Model Provenance & Sigstore**: Models are signed using asymmetric keys or ephemeral OpenID Connect tokens (e.g., Sigstore / Cosign). Model hashes (SHA-256) are committed to immutable transparency logs.
3. **ML Software Bill of Materials (ML-SBOM)**: Documents baseline training code commits, base model weights, tokenizer metadata, and exact dataset digest hashes.

---

## 2. Supply Chain Security Comparison

| Vulnerability Vector | Legacy Format (`.pkl`, `.pt`) | Modern Solution (`.safetensors`) | Mitigation Mechanism |
| :--- | :--- | :--- | :--- |
| **Arbitrary Code Execution (RCE)** | Critical Risk | Zero Risk | Eliminates executable bytecodes from format specification |
| **Memory Mapping Efficiency** | Slow (Copies buffer) | Fast (`mmap` direct copy) | Loads tensor buffers directly into VRAM zero-copy |
| **Integrity Tampering** | High Risk | Cryptographic Digest | SHA-256 header validation & Sigstore signatures |
| **Dependency Confusion** | Medium Risk | Machine-readable SBOM | Pinning hashes in CycloneDX / SPDX manifests |

---

## 3. Production Defense Implementation

This Python module implements an **ML Artifact & Supply Chain Integrity Auditor**. It inspects model files for unsafe unpickling opcodes, validates SHA-256 cryptographic digests, enforces Safetensors compliance, and generates ML-SBOM verification records.

### Prerequisites

```bash
pip install pydantic safetensors
```

### Python Implementation (ml_supply_chain_auditor.py)
```python
import os
import io
import pickle
import pickletools
import hashlib
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel


class MLSBOMManifest(BaseModel):
    artifact_name: str
    file_format: str
    sha256_hash: str
    file_size_bytes: int
    is_safe_format: bool
    passed_integrity_check: bool
    detected_risks: List[str]


class MLSupplyChainAuditor:
    """Audits ML model artifacts for unsafe serialization opcodes and provenance verification."""

    UNSAFE_OPCODES = {"GLOBAL", "REDUCE", "BUILD", "OBJ", "NEWOBJ", "INST"}

    def __init__(self, allowed_formats: Optional[List[str]] = None):
        self.allowed_formats = set(allowed_formats or [".safetensors", ".json", ".onnx"])

    @staticmethod
    def compute_sha256(file_path: str) -> str:
        """Computes SHA-256 digest of a model artifact file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def inspect_pickle_opcodes(self, file_path: str) -> Tuple[bool, List[str]]:
        """Statically inspects a pickle-based binary file for malicious execution opcodes."""
        risks = []
        try:
            with open(file_path, "rb") as f:
                data = f.read()

            # Disassemble pickle stream to inspect structural opcodes
            opcodes = list(pickletools.genops(data))
            for op, arg, pos in opcodes:
                if op.name in self.UNSAFE_OPCODES:
                    risks.append(f"Unsafe Opcode Detected: '{op.name}' with argument '{arg}' at byte position {pos}")

        except Exception as e:
            risks.append(f"Pickle Disassembly Error: {str(e)}")

        is_safe = len(risks) == 0
        return is_safe, risks

    def audit_artifact(self, file_path: str, expected_hash: Optional[str] = None) -> MLSBOMManifest:
        """Full supply chain audit pipeline for ML weight artifacts."""
        file_name = os.path.basename(file_path)
        _, ext = os.path.splitext(file_name)
        file_size = os.path.getsize(file_path)
        actual_hash = self.compute_sha256(file_path)

        risks = []
        is_safe_format = ext.lower() in self.allowed_formats

        if not is_safe_format:
            risks.append(f"Unsafe file extension '{ext}'. Format allows arbitrary execution risk (e.g., pickle).")

        # Hash integrity check
        passed_integrity = True
        if expected_hash:
            if actual_hash.lower() != expected_hash.lower():
                passed_integrity = False
                risks.append(f"Hash Mismatch! Expected: {expected_hash}, Actual: {actual_hash}")

        # If it's a legacy pickle/torch file, run deep opcode inspection
        if ext.lower() in [".pkl", ".pt", ".pth", ".bin"]:
            is_pickle_safe, pickle_risks = self.inspect_pickle_opcodes(file_path)
            if not is_pickle_safe:
                risks.extend(pickle_risks)

        return MLSBOMManifest(
            artifact_name=file_name,
            file_format=ext,
            sha256_hash=actual_hash,
            file_size_bytes=file_size,
            is_safe_format=is_safe_format,
            passed_integrity_check=passed_integrity,
            detected_risks=risks
        )


# -------------------------------------------------------------------
# SIMULATION / VERIFICATION RUNNER
# -------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    auditor = MLSupplyChainAuditor()

    print("=== 1. Generating Synthetic Unsafe Pickle Payload ===")
    class MaliciousPayload:
        def __reduce__(self):
            return (os.system, ("echo HACKED_VIA_PICKLE",))

    unsafe_file = os.path.join(tempfile.gettempdir(), "malicious_model.pth")
    with open(unsafe_file, "wb") as f:
        pickle.dump(MaliciousPayload(), f)

    print("=== 2. Auditing Unsafe Model Artifact ===")
    report_unsafe = auditor.audit_artifact(unsafe_file)
    print(f"Artifact Name   : {report_unsafe.artifact_name}")
    print(f"Safe Format     : {report_unsafe.is_safe_format}")
    print(f"Detected Risks  :\n" + "\n".join(f" - {r}" for r in report_unsafe.detected_risks))

    print("\n=== 3. Generating Synthetic Safe Safetensors Payload ===")
    from safetensors.numpy import save_file
    import numpy as np

    safe_file = os.path.join(tempfile.gettempdir(), "model.safetensors")
    tensors = {"weight": np.zeros((10, 10), dtype=np.float32)}
    save_file(tensors, safe_file)

    print("=== 4. Auditing Safe Model Artifact ===")
    safe_hash = auditor.compute_sha256(safe_file)
    report_safe = auditor.audit_artifact(safe_file, expected_hash=safe_hash)
    print(f"Artifact Name   : {report_safe.artifact_name}")
    print(f"Safe Format     : {report_safe.is_safe_format}")
    print(f"Integrity Check : {report_safe.passed_integrity_check}")
    print(f"SHA-256 Hash    : {report_safe.sha256_hash}")
    print(f"Detected Risks  : {len(report_safe.detected_risks)}")

    # Cleanup
    os.remove(unsafe_file)
    os.remove(safe_file)
```

## 4. Operational Best Practices

* Mandate Safetensors Standard: Disallow .pt, .pth, or .bin weight formats in production deployment pipelines. Enforce automated conversion to .safetensors during CI/CD model packaging.
* Pin Cryptographic SHA-256 Hashes: Always reference exact commit SHAs and SHA-256 digests when fetching models or tokenizers from external registries (e.g., Hugging Face Hub, S3, or MLflow).
* Automate ML-SBOM Generation: Generate machine-readable CycloneDX or SPDX manifests containing exact dataset versions, CUDA container digests, and Python environment locks for every model deployment candidate.
