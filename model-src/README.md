---
language:
- en
- ar
license: mit
tags:
- silx-ai
- quasar-preview
- quasar
- foundation-model
- moe
- 18b
- 2b-active
- long-context
- bittensor
- sn24
- decentralized-training
- distillation
- hybrid-transformer
- loop-transformer
- safe-nope
- drope
pipeline_tag: text-generation
library_name: transformers
---

<p align="center">
  <img src="./quasar_banner.png" alt="Quasar-Preview Foundation Model" width="100%">
</p>

# **Quasar-Preview**

**Quasar-Preview** is the first public model in SILX AI’s **Quasar Foundation Model** series.

It is an early preview checkpoint built to demonstrate the direction of the Quasar architecture at real scale: sparse MoE routing, hybrid recurrent/attention layers, and an experimental long-context configuration designed for future memory-based systems.

This is **not the finished Quasar model**.

Quasar-Preview is the first public step in a larger series of Quasar models that will continue scaling through decentralized training, distillation, architecture improvements, and long-context research on **Bittensor SN24**.

---

## TL;DR

- **First public Quasar model**
- **~18B total parameter MoE**
- **~2B active parameter path**
- **Experimental 5M-token context configuration**
- Built with **Loop Transformer + Quasar hybrid attention**
- Includes **Quasar / Raven / GLA** hybrid layers
- Designed for **Bittensor SN24 decentralized distillation**
- Trained on **>1T and <1.5T tokens**
- Long-context extension path has received **<1B tokens** so far
- Early preview checkpoint, not a final production/SOTA model

Quasar-Preview should be understood as an **architecture preview and foundation checkpoint**, not the final endpoint of the Quasar roadmap.

---

# Important Note

Quasar-Preview is an early model from our broader Quasar model series.

It is released to make the architecture public, allow miners and researchers to work with the model, and begin the next phase of decentralized scaling.

This model is:

- An **early preview checkpoint**
- The **first model** in a planned series of Quasar models
- Trained on **>1T and <1.5T tokens**
- Built for **research, distillation, and SN24 training**
- Not yet the final Quasar model
- Not intended to represent the final quality of the Quasar architecture

Performance is expected to improve through:

- Iterative subnet training
- Distillation cycles
- Longer training runs
- Stronger post-training
- More long-context extension training
- Future Quasar architecture updates

---

# Model Overview

| Field | Value |
| --- | --- |
| Model Name | Quasar-Preview |
| Model Family | Quasar Foundation Models |
| Organization | SILX AI |
| Model Type | `quasar_long` |
| Architecture | Quasar Long Hybrid Transformer |
| Total Parameters | ~18B class |
| Active Parameters | ~2B class sparse MoE path |
| Training Stage | Early preview checkpoint |
| Context Config | Experimental 5M-token config |
| Long-Context Method | Safe NoPE / DrOPE-style staging |
| Tokenizer | Quasar tokenizer preserved from checkpoint lineage |
| Primary Use | Research, distillation, SN24 decentralized training |
| License | MIT |

---

# What Is Active In This Checkpoint?

Quasar-Preview includes several architecture paths. Some are active in this checkpoint, while others are included for future Quasar versions.

| Component | Status in Quasar-Preview |
| --- | --- |
| Sparse MoE | Active |
| Quasar hybrid layers | Active |
| GLA branch | Active |
| Raven branch | Active |
| GQA compatibility attention | Active in this checkpoint |
| Safe NoPE / DrOPE-style context config | Active |
| Loop Transformer scaffold | Present |
| Loop execution | Configured as single-loop |
| Looped anchor injection | Disabled |
| Engram memory | Included and loadable, not active by default |
| 5M context | Config exposed, early long-context training only |

The goal of this release is to expose the first working Quasar architecture checkpoint while keeping the model stable for research and SN24 training.

---

# Quick Start

Quasar-Preview uses custom architecture code.

Use `trust_remote_code=True` when loading the model.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "SILX-AI/Quasar-Preview"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "Explain the purpose of long-context models in simple terms."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Inference Notes

Quasar-Preview is an ~18B total parameter MoE checkpoint. Even though the active path is ~2B parameters, the full checkpoint still requires loading the model weights.

Actual memory usage depends on:

- Precision
- Quantization
- Runtime implementation
- Sequence length
- Batch size
- Device mapping
- Whether long-context experiments are enabled

The 5M context configuration is experimental. Do not assume ordinary inference hardware can run full 5M-token contexts without specialized infrastructure.

---

# Quasar-Preview Benchmark Snapshot

These are early benchmark results from the current Quasar checkpoint lineage.

They should be treated as a moving snapshot, not final model quality.

| Category | Benchmark | Quasar-Preview |
| --- | --- | ---: |
| Knowledge | MMLU (5-shot) | **68.40%** |
| Knowledge | MMLU-Pro | **33.20%** |
| Knowledge | GPQA | **25.60%** |
| Commonsense | ARC Challenge | **63.00%** |
| Commonsense | ARC Easy | **80.10%** |
| Commonsense | PIQA | **81.90%** |
| Commonsense | HellaSwag | **74.00%** |
| Science | OpenBookQA | **47.00%** |
| Math | MATH-500 (4-shot) | **71.40%** |

## Evaluation Notes

These results are provided as an early internal snapshot for the current Quasar-Preview checkpoint lineage.

They are not presented as final model quality. Public verification, different harness versions, prompt formats, decoding settings, and evaluation implementations may change the reported numbers.

When comparing Quasar-Preview to other models, please report:

- Evaluation harness
- Harness version or commit
- Prompt format
- Shot count
- Decoding settings
- Whether chain-of-thought prompting was used
- Exact checkpoint version

---

# Training Strategy

Quasar follows a multi-stage training plan.

Quasar-Preview is an early checkpoint from this plan.

## Stage 1 — Base Pretraining

The base model is trained on a broad corpus to build general next-token prediction, reasoning, and language ability.

Goals of this stage:

- Stabilize the sparse MoE path
- Build general language ability
- Train the hybrid Quasar stack
- Establish a checkpoint suitable for distillation and subnet training

Quasar-Preview has been trained on **>1T and <1.5T tokens** so far.

## Stage 2 — Distillation And Capability Training

After base training, Quasar-Preview is improved through task distillation and targeted capability training.

The goal is to make the checkpoint more useful for:

- Reasoning
- Instruction-following
- Commonsense tasks
- Math and science tasks
- SN24 miner distillation
- Future post-training

This release is designed to be a foundation for continued decentralized improvement rather than the final result.

## Stage 3 — Long-Context Extension

Quasar is designed to move toward ultra-long-context reasoning and memory.

The current checkpoint exposes an experimental **5M-token context configuration** using safe NoPE / DrOPE-style staging.

Important: the 5M context path has received **less than 1B tokens** of long-context extension training so far.

This means the config is present, but mature 5M-token reasoning quality should not be expected yet.

The purpose of this stage is to:

- Preserve short-context behavior
- Avoid damaging the base model during extension
- Prepare the architecture for future long-context training
- Enable research on scalable memory and recall

---

# Quasar Long Hybrid Architecture

Quasar is a hybrid transformer architecture designed for long-context research, sparse computation, and decentralized training.

It is built around:

- A Loop Transformer execution scaffold
- Sparse Mixture-of-Experts routing
- Hybrid Quasar / Raven / GLA branch layers
- Optional anchor-state conditioning
- Optional Engram n-gram memory
- Safe NoPE / DrOPE-style long-context configuration

Quasar-Preview is the first public checkpoint in this architecture family.

---

# Technical Specifications

| Component | Value |
| --- | ---: |
| Total parameters | ~18B |
| Active parameters | ~2B |
| Layers | 20 |
| Hidden size | 2048 |
| Intermediate size | 5120 |
| Attention heads | 16 |
| KV heads | 4 |
| Head dim | 128 |
| Vocabulary size | 157,184 |
| Experts | 256 |
| Experts per token | 8 |
| Shared experts | 1 |
| Active hybrid layers | 4-19 |
| Raven slots | 64 |
| Raven top-k | 32 |
| Engram slots config | 2,000,000 |
| Loop count config | 1 |
| Looped injection config | Disabled |
| Max context config | 5,000,000 |
| Safe NoPE cutoff | 512 |

Compatibility note: this checkpoint includes GQA for the current release path. Future Quasar versions may change this component as the architecture evolves.

---

# Looped Transformer Path

Quasar includes a Loop Transformer execution path.

The idea is to reuse the decoder stack across multiple passes, increasing effective computation depth without copying every parameter into a deeper model.

The current checkpoint is configured conservatively:

```text
num_loops: 1
use_looped_injection: false
```

This means Quasar-Preview runs as a single-loop model by default.

The loop machinery is still part of the architecture code and can be enabled in future Quasar configurations.

When looped injection is enabled, Quasar keeps an anchor snapshot of the input embedding stream, usually called **P**, and injects it back into the hidden state during looped execution.

This gives later loop passes a stable reference to the original token stream.

The intended future looped path is:

```text
Token IDs
  |
  v
Embedding Layer
  |
  +--> Anchor P snapshot
  |
  v
Decoder stack
  |
  v
Loop pass 1
  |
  +--> inject gated Anchor P
  |
  v
Loop pass 2 / future passes
  |
  v
Final hidden state
```

The injection gate is initialized near zero so the model can adapt safely instead of suddenly changing behavior.

This gives Quasar a path toward deeper effective reasoning while keeping parameter count controlled.

---

# Core Data Flow

```text
Token IDs
  |
  v
Token Embedding
  |
  +--> Optional Anchor P snapshot
  |
  v
Early Transformer Blocks
  layers 0-3
  |
  v
Hybrid Quasar Blocks
  layers 4-19
  |
  +--> GQA attention path
  |
  +--> Quasar recurrent / linear path
  |
  +--> Raven slot-memory path
  |
  +--> GLA recurrent path
  |
  v
Hybrid Add / Branch Merge
  |
  v
Optional Loop Injection / Next Loop
  |
  v
RMSNorm
  |
  v
LM Head
  |
  v
Next-token logits
```

---

# Hybrid Layer Composition

The active hybrid layers are:

```text
4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
```

The current layerwise branch cycle is:

```text
quasar -> raven -> quasar -> quasar -> gla
```

Across the hybrid stack, this gives:

- **Quasar branch:** 10 layers
- **Raven branch:** 3 layers
- **GLA branch:** 3 layers

The design keeps Quasar as the dominant branch while giving the model targeted recurrent and slot-memory paths.

---

# Quasar + GLA

GLA is used through the bundled Flash Linear Attention stack.

The goal of the GLA branch is to give Quasar a fast recurrent sequence-mixing path that is cheaper than full dense attention at long lengths.

Current GLA-related config:

```text
hybrid_gla_enabled: true
hybrid_gla_expand_k: 1.0
hybrid_gla_expand_v: 1.0
hybrid_use_short_conv: false
```

GLA is not used as a standalone model here.

It is a branch inside Quasar's hybrid layers.

---

# Raven Design

Raven is included as a slot-routed recurrent attention branch.

Current Raven config:

```text
hybrid_raven_enabled: true
hybrid_raven_slots: 64
hybrid_raven_topk: 32
hybrid_raven_decay_type: Mamba2
```

Raven routes hidden states through a fixed number of recurrent memory slots.

In this checkpoint:

- The branch has **64 memory slots**
- It selects **top-32 routes**
- It uses a **Mamba2-style decay**

Raven gives Quasar a memory-like path where sequence information can be compressed into routed recurrent state instead of relying only on dense attention.

---

# Engram Design

Engram is Quasar's conditional n-gram memory module.

It is included in the repository as `engram.py` and supports:

- n-gram orders `[2, 3]`
- 8 Engram heads
- configurable memory slots
- Triton hash-table lookup
- gated projection back into the residual stream

Current Engram config:

```text
engram_slots: 2,000,000
engram_dim: 512
engram_ngram_orders: [2, 3]
engram_num_heads: 8
engram_residual_scale: 0.01
engram_lr_multiplier: 5.0
engram_layers: []
```

`engram_layers` is currently empty.

This means Engram is included and loadable, but not active by default in Quasar-Preview.

Future Quasar versions can enable Engram on selected layers without changing the base model shape.

Engram is intended as a fast recall path for repeated local patterns, while the main model focuses on reasoning and generalization.

---

# Safe NoPE / DrOPE Context Design

The current checkpoint uses safe NoPE as the default long-context configuration.

Current context config:

```text
use_nope: true
long_context_mode: rope_short_nope_long
nope_after_position: 512
max_position_embeddings: 5,000,000
max_seq_length: 5,000,000
max_sequence_length: 5,000,000
rope_scaling: null
rope_theta: 10000
```

The behavior is:

```text
Positions 0-511
  -> normal RoPE

Positions 512+
  -> NoPE identity rotation
     cos = 1
     sin = 0
```

This is a safe DrOPE-style staging design for positional extension.

The goals are:

- Preserve short-context behavior
- Avoid stretching RoPE everywhere
- Avoid allocating a giant 5M RoPE table
- Expose a 5M sequence-length configuration
- Prepare for future long-context training runs

Important: the 5M context path has only received **less than 1B tokens** of long-context extension training so far.

So high-quality 5M-token reasoning should not be expected yet.

This setting is included to expose and continue training the long-context path safely.

---

# Config Snapshot

```json
{
  "model_type": "quasar_long",
  "architectures": ["QuasarLongForCausalLM"],
  "hidden_size": 2048,
  "intermediate_size": 5120,
  "num_hidden_layers": 20,
  "num_attention_heads": 16,
  "num_key_value_heads": 4,
  "head_dim": 128,
  "vocab_size": 157184,
  "num_experts": 256,
  "num_experts_per_tok": 8,
  "num_shared_experts": 1,
  "num_loops": 1,
  "use_looped_injection": false,
  "hybrid_attention_layers": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
  "hybrid_branch_layout": "layerwise",
  "hybrid_layerwise_cycle": ["quasar", "raven", "quasar", "quasar", "gla"],
  "hybrid_replacement_mode": "add",
  "hybrid_eval_mode": "hybrid_add",
  "hybrid_quasar_enabled": true,
  "hybrid_raven_enabled": true,
  "hybrid_gla_enabled": true,
  "hybrid_raven_slots": 64,
  "hybrid_raven_topk": 32,
  "use_nope": true,
  "long_context_mode": "rope_short_nope_long",
  "nope_after_position": 512,
  "max_position_embeddings": 5000000,
  "max_seq_length": 5000000,
  "max_sequence_length": 5000000
}
```

---

# Intended Use

Quasar-Preview is designed as an early foundation checkpoint for the Quasar ecosystem.

It is primarily intended for:

- **Bittensor SN24 miners** participating in decentralized training and knowledge distillation
- **Distillation pipelines** transferring capabilities from stronger teacher models
- **Research on long-context architectures**
- **Research on sparse MoE systems**
- **Hybrid attention research**
- **Agentic system experiments**
- **Memory and recall experiments**
- **Future Quasar model development**

This model is best treated as a research and development checkpoint.

---

# Out-of-Scope Use

Quasar-Preview is not intended to be used as:

- A final production assistant
- A safety-aligned chatbot
- A medical, legal, or financial authority
- A final benchmark-maximized release
- Proof of mature 5M-token reasoning quality
- The final Quasar architecture endpoint

The model may produce incorrect, unsafe, biased, or low-quality outputs.

Use appropriate evaluation, filtering, and safety layers before any deployment.

---

# Limitations

Quasar-Preview is early.

Known limitations:

- It is not the finished Quasar model.
- It is the first model in a broader Quasar series.
- Long-context behavior is experimental.
- The 5M-token context is a configuration path, not yet mature 5M-token reasoning quality.
- The long-context path has received less than 1B tokens of extension training so far.
- Some architecture modules are included for future versions but disabled in this checkpoint.
- Engram is included but not active by default.
- Loop execution is configured as single-loop by default.
- Benchmarks are early checkpoint-lineage snapshots and require public verification.
- The model may hallucinate or produce incorrect answers.
- The model has not completed the full Quasar training roadmap.

---

# Bittensor SN24

Quasar-Preview is designed for the **SN24 Quasar subnet** on Bittensor.

The goal is to create a shared architecture where miners can continuously improve the model through distributed knowledge distillation, evaluation, and iterative training.

SN24 is intended to support:

- Open model improvement
- Competitive distillation
- Decentralized training incentives
- Shared progress on the Quasar architecture
- Long-context and memory-focused model development

Quasar-Preview is the starting checkpoint for this direction.

---

# Roadmap

Quasar-Preview is only the first public model in the Quasar series.

Next Quasar models will continue toward:

- Larger-scale decentralized training
- More training tokens
- Stronger post-training
- Better reasoning performance
- More stable long-context behavior
- More long-context extension training
- Deeper Loop Transformer experiments
- More Raven, GLA, and Engram experimentation
- Improved benchmark performance
- Stronger agentic and memory capabilities

Future releases may change architecture components, routing, loop configuration, long-context training strategy, and active memory modules as the Quasar series evolves.

---

# Release Statement

Quasar-Preview is not the final destination.

It is the first public checkpoint in the Quasar model series and the first public proof of the architecture direction at scale.

The model is early, but it is real, usable, and ready for research, distillation, and decentralized improvement.

This is the beginning of Quasar.
