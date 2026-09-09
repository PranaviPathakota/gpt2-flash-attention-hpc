# Flash Attention for GPT-2 Training on HPC

Benchmarking **cuDNN Flash Attention vs Standard Attention** for GPT-2 pre-training across 1–16 NVIDIA A100 GPUs on [NERSC Perlmutter](https://docs.nersc.gov/systems/perlmutter/). Built on top of [llm.c](https://github.com/karpathy/llm.c) by Andrej Karpathy.

---

## Key Results

Flash Attention delivers **28–47% higher throughput** and **33–63% less memory per GPU**, and is the **only viable option** for training at 4K–8K sequence lengths on 40 GB GPUs.

### 774M Model — All Configurations

| Setup | Attention | Throughput | Memory/GPU | MFU | Max Seq |
|-------|-----------|-----------|------------|-----|---------|
| 1× A100-40GB | Standard | 145K tok/s | 22 GB | 39.1% | 2K |
| 1× A100-40GB | **Flash** | **209K tok/s** | **13 GB** | **56.0%** | **4K** |
| 4× A100-40GB | Standard | 106K tok/s | 29 GB | 42.5% | 2K |
| 4× A100-40GB | **Flash** | **142K tok/s** | **17 GB** | **56.9%** | **4K** |
| 16× A100-40GB | Standard | 340K tok/s | 8.6 GB | 33.9% | 2K |
| 16× A100-40GB | **Flash** | **438K tok/s** | **5.8 GB** | **43.7%** | **8K** |

### Sequence Length Capability (16× A100-40GB, 774M Model)

| Sequence Length | Standard | Flash | Flash Memory/GPU |
|----------------|----------|-------|-----------------|
| 1K | 340K tok/s | 438K tok/s (+29%) | 5.8 GB |
| 2K | 294K tok/s | 433K tok/s (+47%) | 8.2 GB |
| 4K | OOM | 399K tok/s | 13.4 GB |
| 8K | OOM | 316K tok/s | 25.8 GB |

> Standard attention OOMs at 4K/8K even with 16-GPU parallelism. Flash Attention extends training to 8K sequences using only 65% of GPU memory.

---

## What This Project Does

This project integrates **NVIDIA cuDNN 9.0+ Flash Attention** into the [llm.c](https://github.com/karpathy/llm.c) training framework and systematically benchmarks it against standard attention across:

- **3 model sizes:** GPT-2 124M, 350M, 774M parameters
- **3 GPU tiers:** 1 GPU, 4 GPUs (single node), 16 GPUs (4 nodes, multi-node)
- **4 sequence lengths:** 1K, 2K, 4K, 8K tokens
- **2 attention implementations:** Standard (cuBLAS + custom softmax) vs Flash (cuDNN SDPA)

### The Core Change

Flash Attention is integrated via a single compile flag:

```bash
# Standard attention (default)
make train_gpt2cu

# Flash Attention via cuDNN
make train_gpt2cu USE_CUDNN=1
```

The implementation lives in [`src/llmc/cudnn_att.cpp`](src/llmc/cudnn_att.cpp) — a cuDNN frontend graph-based wrapper that supports FP16/BF16, causal masking, and graph caching to minimize recompilation overhead.

---

## Results

### Memory Efficiency

Flash Attention memory scales as **O(T^1.5)** rather than O(T²), enabling practical long-context training:

![Memory Comparison](results/figures/memory_comparison.png)

### Single GPU: Flash vs Standard (124M Model)

![Single GPU Comparison](results/figures/single_gpu_comparison.png)

### Multi-GPU Scaling (774M Model, 4× A100)

![Multi GPU Comparison](results/figures/multi_gpu_774M_comparison.png)

### 16-GPU Results (774M Model, 4 Nodes)

![16 GPU Comparison](results/figures/16gpu_774M_comparison.png)

### Throughput vs Model Size

![Performance vs Model Size](results/figures/performance_vs_model_size.png)

### Scaling from Single to Multi-GPU

![Scaling Comparison](results/figures/single_vs_multi_gpu_comparison.png)

See [`results/SINGLE_GPU_RESULTS.md`](results/SINGLE_GPU_RESULTS.md) and [`results/MULTI_GPU_RESULTS.md`](results/MULTI_GPU_RESULTS.md) for full tables.

---

## Multi-GPU Scaling

| GPUs | Throughput (Flash, 1K seq) | Scaling Efficiency |
|------|--------------------------|-------------------|
| 1× | 209K tok/s | — |
| 4× | 763K tok/s | 91% |
| 16× | 438K tok/s (774M model) | 77% (multi-node) |

16-GPU runs use 4 nodes connected via **Slingshot-11 interconnect** (~200 Gbps inter-node, NVLink intra-node). Communication overhead is ~12–15% per step.

---

## Hardware

**Platform:** [NERSC Perlmutter](https://docs.nersc.gov/systems/perlmutter/) Supercomputer

| Component | Spec |
|-----------|------|
| GPU | NVIDIA A100-SXM4-40GB (Ampere, SM 8.0) |
| CPU | AMD EPYC 7763 (64 cores) |
| GPU Memory | 40 GB HBM2e |
| Intra-node interconnect | NVLink (600 GB/s) |
| Inter-node interconnect | Slingshot-11 (~200 Gbps) |
| CUDA | 12.4 |
| cuDNN | 9.0+ |
| NCCL | 2.24.3 |
| MPI | Cray MPICH 8.1.30 |

---

## How to Build and Run

### Prerequisites

```bash
# cuDNN 9.0+ (for Flash Attention)
sudo apt-get install libcudnn9-dev-cuda-12

# cuDNN frontend (header-only)
git clone https://github.com/NVIDIA/cudnn-frontend
```

### Build

```bash
cd src/

# Standard attention
make train_gpt2cu

# Flash Attention via cuDNN
make train_gpt2cu USE_CUDNN=1 CUDNN_FRONTEND_PATH=/path/to/cudnn-frontend/include
```

### Single GPU Training

```bash
# Standard attention
./train_gpt2cu -b 32 -t 1024 -d 524288 -e "d12"

# Flash Attention
make train_gpt2cu USE_CUDNN=1
./train_gpt2cu -b 32 -t 1024 -d 524288 -e "d12"
```

### Multi-GPU Training (SLURM)

```bash
# 4 GPUs, standard attention
sbatch scripts/multi_gpu_standard.sh

# 16 GPUs (4 nodes), Flash Attention
sbatch scripts/multi_gpu_flash.sh
```

See [`scripts/`](scripts/) for the full SLURM job scripts used on Perlmutter.

---

## Repo Structure

```
.
├── src/                        # Source code (modified llm.c)
│   ├── llmc/
│   │   ├── cudnn_att.cpp       # Flash Attention via cuDNN (key change)
│   │   ├── cudnn_att.h
│   │   ├── attention.cuh       # Standard attention baseline
│   │   └── [other llm.c headers]
│   ├── dev/cuda/
│   │   ├── attention_forward.cu  # 11 attention kernel implementations
│   │   ├── attention_backward.cu
│   │   └── softmax_forward.cu
│   ├── train_gpt2.cu           # Main CUDA training binary
│   └── Makefile
├── scripts/                    # SLURM job scripts for Perlmutter
│   ├── single_gpu_standard.sh
│   ├── single_gpu_flash.sh
│   ├── multi_gpu_standard.sh
│   ├── multi_gpu_flash.sh
│   └── multi_node/             # Multi-node init methods (TCP, filesystem)
├── results/
│   ├── figures/                # Benchmark graphs
│   ├── logs/                   # Representative SLURM job outputs
│   ├── SINGLE_GPU_RESULTS.md
│   └── MULTI_GPU_RESULTS.md
├── analysis/
│   └── analyze_training_logs.py  # Parses SLURM logs to extract metrics
└── report/
    └── CSCE654_Final_Project_Report_Super_Computing.pdf
```

---

## Report

Full write-up: [`report/CSCE654_Final_Project_Report_Super_Computing.pdf`](report/CSCE654_Final_Project_Report_Super_Computing.pdf)

---

## Attribution

This project is built on [llm.c](https://github.com/karpathy/llm.c) by Andrej Karpathy (MIT License). The Flash Attention integration via cuDNN and all benchmarking experiments are original work.
