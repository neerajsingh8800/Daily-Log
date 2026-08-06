# 06: Distributed Fine-Tuning, Ray, DeepSpeed, and LoRA

This module explores **Distributed LLM Training Infrastructure, Memory Offloading Topologies, and Parameter-Efficient Fine-Tuning (PEFT)**. It covers DeepSpeed ZeRO memory optimization stages, Low-Rank Adaptation (LoRA/QLoRA) weight decomposition mechanics, Ray Train cluster orchestration, and a end-to-end distributed training pipeline.

---

## 1. Enterprise Distributed Fine-Tuning Architecture

Training or fine-tuning Large Language Models across multiple GPU nodes requires overcoming severe memory bottlenecks (optimizer states, gradients, parameter weights, and activations). Decoupling compute workers from memory management via DeepSpeed ZeRO and parameter adapter methods (LoRA) enables training billion-parameter models on accessible compute infrastructure.

### Core Architecture Components

* **Ray Train:** An open-source distributed compute framework that orchestrates multi-node, multi-GPU training jobs, handling node discovery, worker process setup, fault tolerance, and communication primitives.
* **DeepSpeed ZeRO (Zero Redundancy Optimizer):**
  * **ZeRO-Stage 1:** Shards optimizer states across data-parallel processes.
  * **ZeRO-Stage 2:** Shards optimizer states and gradients across data-parallel processes.
  * **ZeRO-Stage 3:** Shards optimizer states, gradients, and model parameters across all worker nodes. Offloads to host CPU/NVMe memory when GPU VRAM limits are reached.
* **Low-Rank Adaptation (LoRA):** Freezes pre-trained base model weights $W_0 \in \mathbb{R}^{d \times k}$ and injects trainable rank decomposition matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ where $r \ll \min(d, k)$, reducing trainable parameters by $>99\%$.

---

## 2. Mathematical Modeling: Memory Allocation & LoRA Weight Decomposition

### 1. ZeRO Memory Reduction Calculus
For an $N$-parameter model using standard FP32 Adam optimizer states, traditional data-parallel training requires memory $M_{total}$ per GPU:

$$M_{total} = M_{weights} + M_{gradients} + M_{optimizer\_states}$$

$$M_{total} = (2N) + (2N) + (4N_{\text{master}} + 4N_{\text{momentum}} + 4N_{\text{variance}}) = 16N \text{ bytes}$$

Under **DeepSpeed ZeRO-3** distributed across $K$ GPUs, memory footprint scales down to:

$$M_{\text{ZeRO-3}} = \frac{2N + 2N + 12N}{K} = \frac{16N}{K} \text{ bytes}$$

---

### 2. LoRA Forward Pass Decomposition
During forward propagation, the updated weight matrix $W$ is re-parameterized as:

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B \cdot A) x$$

where $W_0 \in \mathbb{R}^{d \times k}$ is frozen, $A \sim \mathcal{N}(0, \sigma^2)$, $B = 0$ at initialization, and $\alpha$ is a scaling hyperparameter constant.

---

## 3. DeepSpeed Configuration Manifest (`deepspeed_config.json`)

```json
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "allgather_bucket_size": 5e8,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  },
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

## 4. Production Implementation: Distributed Ray + DeepSpeed + PEFT Pipeline

This complete Python script initializes a Ray Cluster, configures Hugging Face PEFT (LoRA), wraps execution inside Ray Train distributed actors, and executes fine-tuning with DeepSpeed ZeRO-3.
```python
import os
import sys
import json
import logging
import argparse
from typing import Dict, Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

import ray
import ray.train
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer
from ray.train.torch import TorchTrainer, TorchConfig

# Configure structured enterprise logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DistributedFineTuningEngine")


# -------------------------------------------------------------------
# 1. Synthetic Dataset Generator for Fine-Tuning
# -------------------------------------------------------------------
def create_synthetic_instruction_dataset(tokenizer, num_samples: int = 200) -> Dataset:
    """Generates synthetic instruction-following dataset for causal LLM fine-tuning."""
    data = [
        {
            "instruction": "Explain DeepSpeed ZeRO-3 memory offloading.",
            "response": "ZeRO-3 shards optimizer states, gradients, and model parameters across all GPUs and offloads to CPU RAM."
        },
        {
            "instruction": "What is the advantage of LoRA?",
            "response": "LoRA freezes the base model weights and trains rank decomposition matrices, reducing trainable params by over 99%."
        }
    ] * (num_samples // 2)

    formatted_texts = [
        f"<s>[INST] {item['instruction']} [/INST] {item['response']}</s>"
        for item in data
    ]

    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )

    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"].clone()
    })
    return dataset


# -------------------------------------------------------------------
# 2. Distributed Worker Training Loop Function
# -------------------------------------------------------------------
def train_loop_per_worker(config: Dict[str, Any]):
    """Per-worker distributed execution loop executed by Ray Train workers."""
    model_id = config.get("model_id", "facebook/opt-125m")
    deepspeed_config_path = config.get("deepspeed_config_path")

    logger.info(f"👷 Worker Rank {ray.train.get_context().get_world_rank()}: Loading Model & Tokenizer...")

    # Load Tokenizer & Base Model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_cache=False
    )

    # Apply LoRA Parameter-Efficient Fine-Tuning Configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"] if "opt" in model_id else ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load Training Dataset
    dataset = create_synthetic_instruction_dataset(tokenizer)

    # Hugging Face Training Arguments with DeepSpeed
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=5,
        max_steps=20,
        fp16=True,
        deepspeed=deepspeed_config_path if os.path.exists(deepspeed_config_path) else None,
        report_to="none",
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    # Integrate Ray Train Callback for Metrics Sync
    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)

    # Execute Distributed Training
    logger.info("⚡ Worker starting training step execution...")
    trainer.train()


# -------------------------------------------------------------------
# 3. Ray Train Cluster Orchestration Engine
# -------------------------------------------------------------------
def launch_distributed_fine_tuning(
    num_workers: int = 2,
    use_gpu: bool = True,
    model_id: str = "facebook/opt-125m"
):
    """Orchestrates distributed training job across Ray cluster workers."""
    logger.info(f"🚀 Launching Ray Distributed Fine-Tuning Job across {num_workers} workers...")

    # Initialize Local or Remote Ray Cluster
    if not ray.is_initialized():
        ray.init()

    # Create Mock DeepSpeed Config File
    ds_config = {
        "fp16": {"enabled": True},
        "zero_optimization": {"stage": 2},
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto"
    }
    ds_config_path = "deepspeed_config_temp.json"
    with open(ds_config_path, "w") as f:
        json.dump(ds_config, f)

    train_config = {
        "model_id": model_id,
        "deepspeed_config_path": ds_config_path
    }

    # Configure Ray TorchTrainer Spec
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_config,
        torch_config=TorchConfig(backend="nccl" if use_gpu else "gloo"),
        scaling_config=ray.train.ScalingConfig(
            num_workers=num_workers,
            use_gpu=use_gpu,
            resources_per_worker={"CPU": 2, "GPU": 1} if use_gpu else {"CPU": 2}
        )
    )

    results = trainer.fit()
    logger.info(f"✅ Distributed Training Job Completed Successfully. Best Checkpoint Metrics: {results.metrics}")

    # Cleanup temporary configuration manifest
    if os.path.exists(ds_config_path):
        os.remove(ds_config_path)


# -------------------------------------------------------------------
# 4. Command Line Execution Entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Fine-Tuning with Ray, DeepSpeed, and LoRA")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of Ray worker processes")
    parser.add_argument("--cpu_only", action="store_true", help="Force CPU Gloo execution mode for testing")
    parser.add_argument("--model_id", type=str, default="facebook/opt-125m", help="Target HF model ID")

    args = parser.parse_args()

    launch_distributed_fine_tuning(
        num_workers=args.num_workers,
        use_gpu=not args.cpu_only,
        model_id=args.model_id
    )
```

