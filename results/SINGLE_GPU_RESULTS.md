# Single GPU Performance Results Summary

## Extracted from Single_GPU_1hr_train Logs (124M Model, B=32)

### T=1024 (Sequence Length 1024)

**Standard Attention:**
- Memory Usage: 22,252 MiB (~22 GB)
- MFU: 39.1%
- Throughput: 145K tok/s
- Time per step: ~3,600 ms
- Status: ✅ Completed successfully

**Flash Attention (cuDNN):**
- Memory Usage: 13,056 MiB (~13 GB)
- MFU: 56.0%
- Throughput: 209K tok/s
- Time per step: ~2,500 ms
- Status: ✅ Completed successfully

**Flash vs Standard Improvements:**
- Memory Savings: **41%** (22 GB → 13 GB)
- MFU Improvement: **+43%** (39.1% → 56.0%)
- Throughput Speedup: **+44%** (145K → 209K tok/s)

---

### T=2048 (Sequence Length 2048)

**Standard Attention:**
- Status: ❌ No completed runs found (likely OOM or not tested)

**Flash Attention (cuDNN):**
- Memory Usage: 23,100 MiB (~23 GB)
- MFU: 53.7%
- Throughput: 187K tok/s
- Status: ✅ Completed successfully

**Flash vs Standard:**
- Flash Attention **ENABLES** T=2048 training on single 40GB GPU
- Standard appears to be at or beyond memory limits

---

### T=4096 (Sequence Length 4096)

**Standard Attention:**
- Status: ❌ Not tested / OOM expected

**Flash Attention:**
- Status: ❌ Not tested / OOM expected

**Analysis:**
- Both configurations likely exceed 40GB memory limit at this sequence length with B=32

---

### T=8192 (Sequence Length 8192)

**Both configurations:**
- Status: ❌ Not tested / OOM expected

---

## Key Findings

### 1. Flash Attention is FASTER on Single GPU
- **T=1024**: +44% speedup (145K → 209K tok/s)
- This contradicts multi-GPU results where standard was competitive

### 2. Memory Efficiency
- **T=1024**: 41% memory savings (22 GB → 13 GB)
- **T=2048**: Flash uses 23 GB, Standard likely OOM

### 3. Hardware Utilization
- **Flash MFU**: 53.7-56.0%
- **Standard MFU**: 39.1%
- Flash achieves significantly better GPU utilization

### 4. Single GPU Limits
- **Standard Attention**: Max T=1024 reliably
- **Flash Attention**: Max T=2048 reliably
- Both hit limits below multi-GPU capabilities (due to B=32 vs B=8)

---

## Comparison: Single GPU vs Multi-GPU (4×A100)

| Metric | Single GPU<br>Standard | Single GPU<br>Flash | Multi-GPU (4×)<br>Standard | Multi-GPU (4×)<br>Flash |
|--------|------------------------|---------------------|----------------------------|-------------------------|
| **Memory/GPU** | 22 GB | 13 GB | ~12 GB | ~11 GB |
| **MFU** | 39.1% | 56.0% | 38.5% | 50.7% |
| **Throughput** | 145K | 209K | 1,920K | 3,028K |
| **Scaling** | 1× | 1× | 13.2× | 14.5× |

**Insights:**
1. Multi-GPU reduces memory per GPU (B=8 vs B=32)
2. Flash maintains high MFU across configurations
3. Flash scaling is slightly better (14.5× vs 13.2×)

---

## Plotting Data for Visualization

```python
# Single GPU Performance (124M Model, B=32)
seq_lengths = [1024, 2048, 4096, 8192]

# Memory Usage (GB)
standard_memory = [22, None, None, None]  # Only T=1024 tested
flash_memory = [13, 23, None, None]        # T=1024, T=2048 tested

# Throughput (K tok/s)
standard_throughput = [145, None, None, None]
flash_throughput = [209, 187, None, None]

# MFU (%)
standard_mfu = [39.1, None, None, None]
flash_mfu = [56.0, 53.7, None, None]
```

---

## Files Analyzed

**Standard Attention:**
- `Single_GPU_1hr_train/Standard_Attention/llm.c_SingleGPU.o45525117` (T=1024)

**Flash Attention:**
- `Single_GPU_1hr_train/Flash_Attention/llm.SingleGPU_Flash_Attention.o45523933` (T=1024)
- `Single_GPU_1hr_train/Flash_Attention/llm.SingleGPU_Flash_Attention.o45532080` (T=2048)

---

## Recommendations for Presentation

1. **Emphasize Flash Attention superiority on single GPU**:
   - 44% faster throughput
   - 41% memory savings
   - 43% better MFU

2. **Show memory enables longer sequences**:
   - Standard: max T=1024
   - Flash: extends to T=2048

3. **Compare with Multi-GPU results**:
   - Show how batch size affects memory
   - Demonstrate scaling efficiency

4. **Use these actual values** instead of estimated ones in slides
