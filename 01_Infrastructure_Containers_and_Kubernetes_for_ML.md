# 01: Infrastructure, Containers, and Kubernetes for ML

This module explores **Cloud Infrastructure and Containerization Patterns for ML/DL Workloads**. It covers CUDA containerization, multi-stage Docker builds, Kubernetes GPU operator mechanics, node scheduling topologies, and production manifest automation.

---

## 1. Enterprise ML Infrastructure Topology

Deploying Machine Learning and Deep Learning workloads at scale requires specialized infrastructure orchestration. Unlike CPU-bound web services, ML tasks depend on low-latency GPU access, shared host memory drivers, multi-gigabyte container layers, and dedicated hardware affinity.

### Infrastructure Core Components

*   **NVIDIA Container Toolkit (`nvidia-container-runtime`):** Exposes physical host GPU devices (`/dev/nvidia*`) inside unprivileged Docker containers without embedding driver binaries in the image layer.
*   **Kubernetes Device Plugin:** Advertises GPU compute capacity (`nvidia.com/gpu`) to the Kubernetes scheduler.
*   **Multi-Stage Docker Architecture:** Minimizes runtime image footprints by separating heavy CUDA compilation toolchains from the final Python execution runtime.
*   **Shared Memory Allocation (`/dev/shm`):** Prevents PyTorch `DataLoader` multi-process worker crashes during heavy dataset IPC deserialization.

---

## 2. Scaling & Resource Allocations: GPU Fractioning & Shm Scaling

To avoid out-of-memory (OOM) errors during parallel data loading and distributed matrix operations, shared memory ($S_{shm}$) and GPU memory overhead ($M_{gpu}$) must be calculated before Pod scheduling.

### 1. Shared Memory Calculation
PyTorch multi-process dataloaders use POSIX shared memory buffers. For $W$ dataloader workers and average batch tensor size $B_{bytes}$:

$$S_{shm} > W \times B_{bytes} \times 2$$

$$\text{Example: } W = 8 \text{ workers}, \ B_{bytes} = 512 \text{ MB} \implies S_{shm} > 8 \times 512 \text{ MB} \times 2 = 8.192 \text{ GB}$$

Setting `/dev/shm` to a fixed $16 \text{ GB}$ buffer prevents runtime IPC pipe exhaustion.

---

### 2. GPU Utilization Calculus
Let $R_{req}$ be required compute TFLOPS and $V_{req}$ be VRAM allocation in gigabytes. When allocating fractional or multi-GPU nodes:

$$V_{req} = \text{Model Parameters (B)} \times \text{Precision Factor (Bytes)} + \text{KV Cache Memory}$$

$$\text{FP16 Model (7B Params): } V_{req} = (7 \times 2 \text{ GB}) + 4 \text{ GB (KV Overhead)} = 18 \text{ GB Minimum VRAM}$$

---

## 3. Production Multi-Stage Dockerfile Strategy

A naive PyTorch container image exceeds $15 \text{ GB}$ due to full CUDA toolkits (`nvcc`, static headers, build caches). Using a multi-stage Docker architecture reduces the production runtime footprint to $\sim 3.8 \text{ GB}$.

```dockerfile
# ===================================================================
# Stage 1: Builder Engine (Includes CUDA Compiler Toolchain & Dev Libs)
# ===================================================================
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies & Python build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment isolated directory
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade core pip infrastructure
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 extension bindings and dependencies
RUN pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
COPY requirements.txt .
RUN pip install -r requirements.txt

# ===================================================================
# Stage 2: Runtime Minimal Image (Production Target)
# ===================================================================
FROM nvidia/cuda:12.2.2-base-ubuntu22.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install minimal runtime libraries (libgomp1 for multi-threading optimization)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-compiled virtual environment from Builder Stage
COPY --from=builder /opt/venv /opt/venv

# Set production execution workspace
WORKDIR /app
COPY src/ /app/src/

# Non-root user setup for enterprise security compliance
RUN useradd -m -u 1000 mlrunner && chown -R mlrunner:mlrunner /app
USER mlrunner

ENTRYPOINT ["python3", "-m", "src.main"]
4. Production Kubernetes Manifest: Dedicated GPU Node Affinity & Shm Volume
Here is a complete, production-grade Kubernetes Deployment manifest featuring GPU resource limits, node affinity, tolerations, health probes, and shared memory (/dev/shm) mount configurations.

YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-gpu-inference-engine
  namespace: ml-production
  labels:
    app.kubernetes.io/name: gpu-inference
    app.kubernetes.io/tier: model-serving
spec:
  replicas: 2
  selector:
    matchLabels:
      app: gpu-inference
  template:
    metadata:
      labels:
        app: gpu-inference
    spec:
      # -----------------------------------------------------------------
      # 1. Node Allocation: Tolerations & Target Affinity
      # -----------------------------------------------------------------
      tolerations:
        - key: "[nvidia.com/gpu](https://nvidia.com/gpu)"
          operator: "Exists"
          effect: "NoSchedule"
      
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: accelerator
                    operator: In
                    values:
                      - nvidia-tesla-a100
                      - nvidia-rtx-4090

      # -----------------------------------------------------------------
      # 2. Container Execution Spec
      # -----------------------------------------------------------------
      containers:
        - name: inference-container
          image: my-registry.internal/ml/gpu-inference:v1.0.0
          imagePullPolicy: IfNotPresent
          
          # Environment variables for PyTorch & NVIDIA runtime
          env:
            - name: CUDA_VISIBLE_DEVICES
              value: "0"
            - name: NCCL_DEBUG
              value: "INFO"
            - name: MODEL_NAME
              value: "meta-llama/Llama-3-8b"
          
          # Compute Resource Limits (Exposing Physical GPU)
          resources:
            requests:
              cpu: "8"
              memory: "32Gi"
              [nvidia.com/gpu](https://nvidia.com/gpu): "1"
            limits:
              cpu: "16"
              memory: "64Gi"
              [nvidia.com/gpu](https://nvidia.com/gpu): "1"

          # -------------------------------------------------------------
          # 3. Volume Mounts for Shared Memory IPC
          # -------------------------------------------------------------
          volumeMounts:
            - name: dshm
              mountPath: /dev/shm
            - name: model-cache
              mountPath: /root/.cache

          # Health Probes
          readinessProbe:
            httpGet:
              path: /healthz
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /healthz
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 15

      # -----------------------------------------------------------------
      # 4. Storage Volumes (RAM Disk for Shared Memory)
      # -----------------------------------------------------------------
      volumes:
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: "16Gi"
        - name: model-cache
          persistentVolumeClaim:
            claimName: pvc-model-weights-cache
5. Automated Python Cluster Health Diagnostic Script
This production-grade diagnostic script validates CUDA device availability, tests memory allocations, verifies PyTorch-accelerated matrix multiplication operations, and outputs structured JSON diagnostics for Kubernetes container readiness probes.

Python
import sys
import time
import json
import logging
from typing import Dict, Any

# Configure structured JSON output logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("GPUContainerHealthCheck")


def perform_gpu_diagnostics() -> Dict[str, Any]:
    """Evaluates CUDA runtime availability, VRAM memory metrics, and performs tensor ops."""
    report: Dict[str, Any] = {
        "status": "HEALTHY",
        "timestamp": time.time(),
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "tensor_op_successful": False
    }

    try:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        if not torch.cuda.is_available():
            logger.error("❌ CUDA is not available to the PyTorch runtime container.")
            report["status"] = "UNHEALTHY"
            report["error"] = "CUDA driver/device plugin unavailable."
            return report

        report["device_count"] = torch.cuda.device_count()

        for i in range(report["device_count"]):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            total = props.total_memory / (1024 ** 3)

            device_info = {
                "device_index": i,
                "name": props.name,
                "total_vram_gb": round(total, 2),
                "allocated_vram_gb": round(allocated, 2),
                "reserved_vram_gb": round(reserved, 2)
            }
            report["devices"].append(device_info)

        # Execute Tensor Multiplication Diagnostic Test
        logger.info("🧪 Executing CUDA Tensor Multiplication Diagnostic Test...")
        a = torch.randn(2000, 2000, device="cuda:0", dtype=torch.float32)
        b = torch.randn(2000, 2000, device="cuda:0", dtype=torch.float32)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

        if c.shape == (2000, 2000):
            report["tensor_op_successful"] = True
            logger.info("✅ CUDA Matrix Multiplication executed successfully on GPU.")

    except Exception as err:
        logger.error(f"❌ Error during GPU Container Health Check: {str(err)}")
        report["status"] = "UNHEALTHY"
        report["error"] = str(err)

    return report


if __name__ == "__main__":
    logger.info("🔍 Initializing Infrastructure Container GPU Health Probes...")
    results = perform_gpu_diagnostics()
    
    # Dump JSON telemetry log
    print(json.dumps(results, indent=2))

    if results["status"] != "HEALTHY":
        sys.exit(1)
    sys.exit(0)
```
