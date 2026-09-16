# 15: Confidential Computing and Secure Inference Enclaves

## 1. Overview & Threat Landscape

Deploying proprietary Large Language Models (LLMs) and handling sensitive user prompts (e.g., healthcare records, financial metrics, trade secrets) in untrusted public cloud environments creates significant security risks. Standard cloud security controls isolate virtual machines (VMs) from other tenants, but leave data vulnerable in memory to host OS compromises, rogue cloud administrators, hypervisor breaches, and physical hardware tampering.

**Confidential Computing** mitigates these risks by executing sensitive AI inference workloads inside hardware-isolated execution environments called **Trusted Execution Environments (TEEs)** or **Secure Enclaves**.

### Confidential Computing Hardware Paradigms for AI

| Architecture | Vendor | Hardware Isolation Mechanism | Typical AI Workload Use Case |
| :--- | :--- | :--- | :--- |
| **AWS Nitro Enclaves** | Amazon Web Services | Isolated CPU/RAM slices with no network interface, interactive access, or persistent storage. Commits via `vsock`. | CPU-based secure inference, secret handling, key management. |
| **AMD SEV-SNP** | AMD | Secure Encrypted Virtualization with Secure Nested Paging; memory encryption keys managed by dedicated security processor. | Confidential Virtual Machines (CVMs) running PyTorch/vLLM. |
| **Intel SGX / TDX** | Intel | Process-level application enclaves (SGX) and Hardware-isolated Trust Domains (TDX). | Secure microservices, tokenizers, privacy-preserving analytics. |
| **NVIDIA H100/H200 CC** | NVIDIA | On-die GPU memory encryption and hardware-enforced PCIe bus encryption between CPU TEE and GPU memory. | High-performance GPU confidential LLM inference (vLLM, TensorRT-LLM). |

---

## 2. Cryptographic Remote Attestation Mechanics

**Remote Attestation** is the foundational protocol used by a client to verify that a remote enclave is genuinely running on authentic hardware, has not been tampered with, and is running an exact expected software measurement before sending sensitive data or decryption keys.

### Steps of Attestation:
1. **Measurement Generation:** As the enclave boots, the hardware processor computes cryptographic hash measurements of the code, initial memory state, and configuration (stored in Platform Configuration Registers - PCRs).
2. **Attestation Report Signing:** The hardware's internal Security Processor signs a report containing the PCR measurements, nonces, and enclave public keys using a factory-fused private key (Hardware Root of Trust).
3. **Verification:** The client verifies the hardware signature against the hardware vendor’s public key infrastructure (e.g., AMD, Intel, or AWS CA) and confirms that the PCR hash matches the known build artifact.
4. **Key Delivery:** Once attested, an encrypted TLS session (aTLS) is established, and model weights or sensitive user prompts are safely transmitted into the enclave.

---

## 3. Defense Architecture for Secure Inference Pipelines

To implement end-to-end confidential inference, the architecture must bind hardware attestation directly with application-layer network security.

1. **Measured Boot Image:** Build the inference container (OS + Python environment + model code) into a deterministic, verifiable disk image with published PCR hashes.
2. **Hardware Key Release (KMS Integration):** The Key Management Service releases model decryption keys *only* when presented with a valid attestation report matching the authorized PCR measurement.
3. **Encrypted In-Transit & In-Memory:** Prompts are encrypted using keys established during attestation; model weights remain encrypted on disk and are decrypted directly inside TEE memory.
4. **Zero External Access:** Disable root SSH, interactive shells, external networking, and core dumps inside the enclave runtime.

---

## 4. Hands-On Python Implementations

### Example 1: AWS Nitro Enclave `vsock` Secure Communication Proxy

AWS Nitro Enclaves communicate exclusively with the parent VM over a virtual socket (`vsock`) interface rather than standard TCP/IP.

```python
import socket
import struct
import json
from typing import Dict, Any

class NitroVsockProxy:
    def __init__(self, port: int = 5000):
        self.port = port
        self.CID_ANY = -1  # VMADDR_CID_ANY

    def start_enclave_listener(self, handler_callback):
        """
        Runs INSIDE the Nitro Enclave. Listens for incoming inference payloads over vsock.
        """
        # AF_VSOCK socket family (AF_VSOCK = 40 on Linux)
        sock = socket.socket(40, socket.SOCK_STREAM)
        sock.bind((self.CID_ANY, self.port))
        sock.listen(5)
        print(f"[Enclave] Listening on vsock port {self.port}...")

        while True:
            conn, addr = sock.accept()
            print(f"[Enclave] Connection established from CID: {addr[0]}")
            
            # Read 4-byte message length header
            raw_len = conn.recv(4)
            if not raw_len:
                conn.close()
                continue
            
            msg_len = struct.unpack("!I", raw_len)[0]
            data = conn.recv(msg_len)
            
            request = json.loads(data.decode('utf-8'))
            
            # Execute processing inside secure boundary
            response = handler_callback(request)
            
            # Send back response over vsock
            serialized_resp = json.dumps(response).encode('utf-8')
            conn.sendall(struct.pack("!I", len(serialized_resp)) + serialized_resp)
            conn.close()

    @staticmethod
    def send_vsock_request(parent_cid: int, port: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs on the PARENT VM to relay prompts into the secure enclave over vsock.
        """
        sock = socket.socket(40, socket.SOCK_STREAM)
        sock.connect((parent_cid, port))

        serialized = json.dumps(payload).encode('utf-8')
        # Send length-prefixed payload
        sock.sendall(struct.pack("!I", len(serialized)) + serialized)

        # Read response length
        raw_len = sock.recv(4)
        msg_len = struct.unpack("!I", raw_len)[0]
        
        response_bytes = sock.recv(msg_len)
        sock.close()
        return json.loads(response_bytes.decode('utf-8'))

# Example Usage Demonstration
if __name__ == "__main__":
    def secure_inference_handler(request: Dict[str, Any]) -> Dict[str, Any]:
        prompt = request.get("prompt", "")
        # Simulated secure inside-enclave inference execution
        return {
            "status": "success",
            "enclave_processed": True,
            "completion": f"Enclave Response to: '{prompt}'"
        }

    print("Nitro Vsock Communication Interface Initialized.")
```



