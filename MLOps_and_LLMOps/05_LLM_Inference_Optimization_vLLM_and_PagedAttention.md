# 05: LLM Inference Optimization, vLLM, and PagedAttention

This module explores **Large Language Model (LLM) Inference Mechanics, VRAM Memory Management, and High-Throughput Serving Systems**. It covers Key-Value (KV) cache memory fragmentation, the PagedAttention memory management algorithm, continuous batching iteration scheduling, quantization precision tradeoffs, and an automated Python vLLM serving and benchmarking engine.

---

## 1. Enterprise LLM Serving Architecture: Memory Bottlenecks & PagedAttention

Standard transformer autoregressive decoding is inherently memory-bandwidth bound. During generation, each newly generated token requires attending to all prior tokens, necessitating the storage of Key-Value tensor projections in VRAM (**KV Cache**). Naive static memory allocation leads to severe internal and external memory fragmentation (up to $60\text{--}80\%$ wasted VRAM), severely limiting batch sizes and throughput.

### Core Architecture Components

* **KV Cache Memory Fragmentation:** Traditional runtimes reserve contiguous maximum-sequence length memory buffers per request upfront, causing massive VRAM waste when prompts or completions are shorter than max context length.
* **PagedAttention Algorithm:** Inspired by virtual memory and paging in OS design, PagedAttention partitions the KV cache into fixed-size physical blocks (pages). Tokens are mapped dynamically to non-contiguous VRAM physical memory blocks, eliminating internal fragmentation and enabling KV cache sharing (e.g., parallel sampling, system prompts).
* **Continuous Batching (Iteration-Level Scheduling):** Instead of waiting for an entire batch to finish generating (request-level batching), continuous batching evicts completed requests and injects newly arrived requests at every iteration step.

---

## 2. Mathematical Modeling: KV-Cache Sizing & PagedAttention Allocation

### 1. KV-Cache Memory Calculation
For an $L$-layer Transformer model with hidden dimension $H$, number of attention heads $N_{heads}$, sequence length $S$, batch size $B$, and precision factor $P_{bytes}$ (e.g., $2 \text{ bytes}$ for FP16/BF16):

$$M_{KV} = 2 \times L \times H \times S \times B \times P_{bytes}$$

$$\text{Example (Llama-3-8B FP16): } L=32, H=4096, S=4096, B=16, P_{bytes}=2 \implies M_{KV} \approx 17.17 \text{ GB}$$

Without PagedAttention, reserving $S_{max}=8192$ for all $16$ requests requires $34.35 \text{ GB}$ of static VRAM, triggering OOM errors on single GPUs.

---

## 3. Production vLLM Engine Configuration & Benchmarking Script

This complete, production-grade Python script provisions an optimized **vLLM** inference engine with **PagedAttention**, configures tensor parallelism, sets up dynamic continuous batching limits, executes local LLM generation, and benchmark latency/throughput metrics.

```python
import time
import json
import logging
import argparse
from typing import List, Dict, Any

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

# Configure structured enterprise logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("vLLMInferenceEngine")


# -------------------------------------------------------------------
# 1. vLLM Engine Initializer with PagedAttention Configuration
# -------------------------------------------------------------------
def initialize_vllm_engine(
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 4096,
    quantization: str = None
) -> LLM:
    """
    Initializes high-throughput vLLM engine with PagedAttention memory management.
    
    Args:
        model_id: Hugging Face model repository identifier or local directory.
        tensor_parallel_size: Number of GPUs for Tensor Parallelism execution.
        gpu_memory_utilization: Fraction of GPU VRAM allocated to KV-Cache blocks.
        max_model_len: Maximum context window sequence length.
        quantization: Quantization method ('awq', 'gptq', 'fp8', or None).
    """
    logger.info(f"🚀 Initializing vLLM Engine for model: {model_id}...")
    logger.info(f"⚙️ Config: Tensor Parallelism={tensor_parallel_size}, VRAM Util={gpu_memory_utilization}, Max Len={max_model_len}")

    llm = LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        quantization=quantization,
        trust_remote_code=True,
        enforce_eager=False # Uses CUDA Graphs for lower overhead
    )
    
    logger.info("✅ vLLM Engine successfully loaded with PagedAttention enabled.")
    return llm


# -------------------------------------------------------------------
# 2. Batch Generation & Throughput Benchmark Engine
# -------------------------------------------------------------------
def run_llm_benchmark(
    llm: LLM,
    prompts: List[str],
    temperature: float = 0.7,
    max_tokens: int = 256
) -> Dict[str, Any]:
    """Executes high-throughput continuous batch generation and logs benchmark metrics."""
    logger.info(f"⚡ Processing batch of {len(prompts)} prompt requests...")

    # Configure Sampling Parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        presence_penalty=0.0,
        frequency_penalty=0.0
    )

    # Benchmark Generation Execution
    start_time = time.time()
    outputs: List[RequestOutput] = llm.generate(prompts, sampling_params)
    total_time_sec = time.time() - start_time

    total_prompt_tokens = 0
    total_generated_tokens = 0
    results = []

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        num_prompt_tokens = len(output.prompt_token_ids)
        num_gen_tokens = len(output.outputs[0].token_ids)

        total_prompt_tokens += num_prompt_tokens
        total_generated_tokens += num_gen_tokens

        results.append({
            "request_id": output.request_id,
            "prompt_tokens": num_prompt_tokens,
            "generated_tokens": num_gen_tokens,
            "text_preview": generated_text[:100].strip() + "..."
        })

    # Performance Metrics Calculation
    total_tokens = total_prompt_tokens + total_generated_tokens
    tok_per_sec = total_generated_tokens / total_time_sec
    avg_latency_per_request = total_time_sec / len(prompts)

    benchmark_summary = {
        "num_requests": len(prompts),
        "total_execution_time_sec": round(total_time_sec, 3),
        "total_prompt_tokens": total_prompt_tokens,
        "total_generated_tokens": total_generated_tokens,
        "tokens_per_second": round(tok_per_sec, 2),
        "avg_latency_per_request_sec": round(avg_latency_per_request, 3),
        "sample_outputs": results[:2]
    }

    logger.info(f"📊 Benchmark Execution Summary:\n{json.dumps(benchmark_summary, indent=2)}")
    return benchmark_summary


# -------------------------------------------------------------------
# 3. Execution CLI Entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Inference Optimization Benchmark Engine")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="Hugging Face model ID")
    parser.add_argument("--tensor_parallel", type=int, default=1, help="GPUs for Tensor Parallelism")
    parser.add_argument("--vram_util", type=float, default=0.85, help="GPU memory utilization factor")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max output tokens per request")
    
    args = parser.parse_args()

    # Synthetic Benchmark Prompt Workload
    sample_prompts = [
        "Explain the concept of PagedAttention in modern LLM inference systems.",
        "Write a Python function to compute the Fibonacci sequence using dynamic programming.",
        "What are the key differences between FP16, INT8, and AWQ INT4 quantization?",
        "Describe how Continuous Batching solves line-of-head blocking in transformer serving.",
        "Summarize the architecture of a distributed Feature Store for real-time machine learning."
    ] * 4  # 20 total batch requests

    # Initialize Engine & Run Benchmark
    engine = initialize_vllm_engine(
        model_id=args.model,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.vram_util
    )

    run_llm_benchmark(
        llm=engine,
        prompts=sample_prompts,
        max_tokens=args.max_tokens
    )
```
