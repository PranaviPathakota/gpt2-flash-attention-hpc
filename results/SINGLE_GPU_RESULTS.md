# Single GPU Performance Results

**Hardware:** NERSC Perlmutter — 1× NVIDIA A100-SXM4-40GB
**Model:** GPT-2 124M parameters (12 layers, 768 hidden dim, 12 attention heads)
**Batch Configuration:** B=32 sequences per GPU, gradient accumulation to 524,288 tokens

---

## Results

### T=1024 (Sequence Length 1K)

| Metric | Standard Attention | Flash Attention | Improvement |
|--------|-------------------|-----------------|-------------|
| **Throughput** | 144.5K tok/s | 206.9K tok/s | **+43.2%** |
| **MFU** | 38.7% | 55.5% | **+43.4%** |
| **Time/Step** | 3,630 ms | 2,534 ms | **−30.2%** (faster) |
| **Memory/GPU** | 21.7 GB | 12.8 GB | **−41.0%** |

**Training Steps Completed (1 hour):**
- Standard: 973 steps
- Flash: 1,378 steps (1.4× more iterations)

---

### T=2048 (Sequence Length 2K)

| Metric | Standard Attention | Flash Attention |
|--------|-------------------|-----------------|
| **Throughput** | ❌ OOM | 187.5K tok/s |
| **MFU** | ❌ OOM | 53.7% |
| **Time/Step** | ❌ OOM | 2,795 ms |
| **Memory/GPU** | ❌ OOM | 23.1 GB |

> Standard attention OOMs at T=2048 with B=32. Flash Attention fits in 23.1 GB (58% of 40 GB capacity).

**Training Steps Completed (1 hour):**
- Standard: 0 steps (OOM at initialization)
- Flash: 1,242 steps

---

### T=4096 and T=8192

Both Standard and Flash Attention OOM at T=4096 and T=8192 with B=32 on a single 40 GB GPU.

**Why:** At B=32, activation memory alone exceeds 40 GB for sequences ≥4K even with Flash Attention. With fewer sequences per GPU (e.g., B=2 on 16 GPUs), Flash can reach T=8192 — see `MULTI_GPU_RESULTS.md`.

---

## Key Findings

### 1. Flash Attention — 43% Throughput Gain at T=1K
- **+43% throughput** despite same batch size, same model, same GPU
- Gain comes from: fewer HBM reads/writes (Flash fuses QK^T softmax into SRAM tiles, never writing the full attention matrix to HBM)
- MFU increases from 38.7% → 55.5%, approaching theoretical peak

### 2. Memory Savings Enable T=2K
- **Standard**: 21.7 GB at T=1K, OOM at T=2K — quadratic memory growth in attention matrices hits the 40 GB limit
- **Flash**: 12.8 GB at T=1K, 23.1 GB at T=2K — linear activation growth, no quadratic bottleneck
- Flash doubles sequence length capacity on a single GPU

### 3. Hardware Utilization
- Flash MFU (55.5%) is substantially higher than Standard (38.7%)
- Standard is memory-bandwidth limited — large activation writes/reads throttle compute
- Flash's SRAM-based computation keeps data near the tensor cores

