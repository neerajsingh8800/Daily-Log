# Inference Optimization Engines

Deploying Large Language Models (LLMs) in production presents severe operational bottlenecks. Because auto-regressive generation requires predicting tokens one by one, the workload is overwhelmingly **memory-bandwidth bound** rather than compute-bound. 

This document explores the architectural mechanics, mathematical constraints, and serving paradigms utilized by modern inference optimization engines like vLLM, TGI, and TensorRT-LLM to maximize throughput and minimize latency.

---

## 1. The Core Bottleneck: Compute-Bound vs. Memory-Bound

To optimize LLM serving, we must analyze operational operational efficiency using the **Roofline Model**. The Roofline model relates operational intensity (FLOPs performed per byte of memory moved) to the hardware's peak performance boundaries.

### Prefill Phase (Compute-Bound)
* **What happens:** The engine processes the entire user prompt simultaneously.
* **Characteristics:** The key-value states for all prompt tokens are computed concurrently. This maximizes parallelization across CUDA cores, making the phase highly efficient and constrained primarily by the GPU's total compute capacity ($\text{TFLOPs}$).

### Decode Phase (Memory-Bound)
* **What happens:** The engine generates tokens sequentially, one step at a time.
* **Characteristics:** To generate a single new token, the entire weight matrix of the model ($W$) and all prior Key-Value caches must be transferred from global High Bandwidth Memory (HBM) to the GPU's localized SRAM registers. The actual arithmetic operations performed on that data are minimal, meaning the hardware spends most of its execution loops stalled, waiting for memory transfers.

---

## 2. Advanced Dynamic Serving Paradigms

Traditional deep learning serving systems use static batching, which forces requests to wait until an entire batch finishes execution. This is highly inefficient for open-ended LLM generation due to large variations in output lengths.

### Continuous Batching (Iteration-Level Scheduling)
Introduced by Orca (Yu et al., 2022), **Continuous Batching** abandons requests-level batching. Instead, the scheduler operates at the **iteration level**. 

As soon as an individual request within an active batch emits its end-of-sequence (`<|endoftext|>`) token, it is dropped from the batch immediately. A newly arrived request's *prefill phase* is then dynamically injected into the vacant slot during the very next forward pass execution loop. This eliminates idle resource waste across mismatched sequence tasks.

---

## 3. PagedAttention Mathematics and Architecture

The primary limitation to maximizing the batch size ($B$) in a continuous batching system is the explosive memory footprint of the **KV Cache**.

### The KV Cache Fragmentation Problem
In standard transformers, memory for a request's KV Cache must be allocated statically and contiguously based on the model's maximum possible sequence length (e.g., 4096 tokens). 
* **Internal Fragmentation:** If a user request only generates 50 tokens, the remaining 4046 tokens allocated are locked out and wasted.
* **Virtual Memory Analog:** This mirrors the classic physical memory allocation problems in early operating systems.

### PagedAttention Mechanics
Developed by vLLM (Kwon et al., 2023), **PagedAttention** resolves fragmentation by decoupling logical sequence tokens from physical memory slots using virtual paging concepts. 

The KV Cache of a sequence is broken into small, fixed-size **blocks** (typically holding 16 tokens). These blocks do not need to be stored contiguously in virtual GPU memory space. A central **Page Table** maps logical token blocks to physical block addresses on the fly.

### Mathematical Formulation
Let $i$ be the logical token index within a sequence. The KV cache values for this token are mapped to a physical block index and slot offset. Let $B_s$ be the block size (number of tokens per block).

The logical block index $\lfloor i / B_s \rfloor$ is mapped via the Page Table $\mathcal{M}$ to a physical block identifier:

$$P_k = \mathcal{M}\left(\left\lfloor \frac{i}{B_s} \right\rfloor\right)$$

The explicit slot offset within that physical block is computed via a modulo operation:

$$\tau = i \pmod{B_s}$$

During the attention calculation loop, the structural lookup for Key vector block sequence fragments is retrieved non-contiguously from physical memory arrays matching:

$$\text{Attention}(Q_i) = \text{Softmax}\left( \frac{Q_i \cdot K_{\mathcal{M}(\lfloor j/B_s \rfloor), (j \pmod{B_s})}^T}{\sqrt{d_k}} \right) \cdot V_{\mathcal{M}(\lfloor j/B_s \rfloor), (j \pmod{B_s})}$$

---

## 4. Architectural Comparison of Modern Serving Engines

| Optimization Dimension | vLLM | TensorRT-LLM | Hugging Face TGI | Ollama |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Target** | High-throughput cloud API endpoints. | Maximum performance on NVIDIA enterprise clusters. | Enterprise production hosting. | Local, localized workstation execution. |
| **Core Memory Architecture** | Custom PagedAttention. | TensorRT optimized Paged KV-Cache management. | PagedAttention wrapper bindings. | Formatted static/dynamic memory limits via llama.cpp. |
| **Kernel Specialization** | Highly customized PyCUDA / Triton extensions. | Proprietary NVIDIA TensorRT compilation graphs. | FlashAttention + custom CUDA optimizations. | Pure C/C++ AVX/AMX CPU/GPU portable logic. |

---

## 5. Implementation: Continuous Batching Engine Simulation

Below is a complete Python implementation demonstrating an iteration-level scheduling algorithm for continuous batching. It simulates the lifecycle of processing mixed prompts, executing concurrent prefill/decode phases, and dynamically freeing slots as requests finish.

```python
import time
import random
from typing import List, Dict, Optional

class InferenceRequest:
    def __init__(self, request_id: int, prompt: str, expected_output_len: int):
        self.request_id = request_id
        self.prompt = prompt
        self.expected_output_len = expected_output_len
        self.tokens_generated = 0
        self.phase = "PREFILL"  # Phases: PREFILL -> DECODE -> FINISHED
        self.start_time = time.time()
        self.end_time: Optional[float] = None

    @property
    def total_latency(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return 0.0

class ContinuousBatchingEngine:
    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size
        self.active_batch: List[InferenceRequest] = []
        self.waiting_queue: List[InferenceRequest] = []
        self.completed_requests: List[InferenceRequest] = []

    def submit_request(self, request: InferenceRequest):
        """Adds incoming user requests directly to the backlog queue."""
        self.waiting_queue.append(request)
        print(f"[Submission] Request {request.request_id} queued (Target Output Len: {request.expected_output_len} tokens).")

    def _step_scheduler(self):
        """Fills vacant processing slots from the waiting queue."""
        while len(self.active_batch) < self.max_batch_size and self.waiting_queue:
            next_req = self.waiting_queue.pop(0)
            self.active_batch.append(next_req)
            print(f" -> [Schedule] Request {next_req.request_id} pulled into active batch.")

    def step_forward_iteration(self):
        """Simulates one unified iteration execution pass of the engine."""
        self._step_scheduler()
        
        if not self.active_batch:
            return

        print(f"\n--- Iteration Loop Pass: {len(self.active_batch)} Active Requests ---")
        
        # Track finished placeholders to clean up at the end of the iteration step
        finished_this_step = []

        for req in self.active_batch:
            if req.phase == "PREFILL":
                # Prefill processes all prompt conditions simultaneously in 1 cycle
                print(f" [Compute] Request {req.request_id} execution: Processing prompt prefill phase.")
                req.phase = "DECODE"
            elif req.phase == "DECODE":
                # Decode appends exactly 1 token per step
                req.tokens_generated += 1
                print(f" [Memory]  Request {req.request_id} execution: Generated decode token {req.tokens_generated}/{req.expected_output_len}.")
                
                if req.tokens_generated >= req.expected_output_len:
                    req.phase = "FINISHED"
                    req.end_time = time.time()
                    finished_this_step.append(req)

        # Evict finished requests to make space for the next iteration
        for req in finished_this_step:
            self.active_batch.remove(req)
            self.completed_requests.append(req)
            print(f" 🎉 [Evict] Request {req.request_id} finalized and evicted from cache block.")

# --- Simulation Sandbox Verification ---
if __name__ == "__main__":
    # Initialize Engine with a max active batch size of 2
    engine = ContinuousBatchingEngine(max_batch_size=2)
    
    # Mocking uneven workload demands
    engine.submit_request(InferenceRequest(request_id=101, prompt="Explain Quantum Physics", expected_output_len=4))
    engine.submit_request(InferenceRequest(request_id=102, prompt="Hi", expected_output_len=1))
    engine.submit_request(InferenceRequest(request_id=103, prompt="Write a Python script for sorting", expected_output_len=3))
    
    # Process iterations until everything is completed
    iteration = 0
    while (engine.active_batch or engine.waiting_queue) and iteration < 10:
        engine.step_forward_iteration()
        iteration += 1
        
    print("\n=== Performance Metrics Overview ===")
    for req in engine.completed_requests:
        print(f"Request {req.request_id} finished successfully. Tokens Generated: {req.tokens_generated}")
```
