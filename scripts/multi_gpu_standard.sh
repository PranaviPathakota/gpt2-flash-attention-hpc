#!/bin/bash
#SBATCH -A m4012                        # NERSC account
#SBATCH -C "gpu&hbm40g"               # 40GB A100 GPU nodes
#SBATCH -q regular                      # queue
#SBATCH -t 01:00:00                     # time limit
#SBATCH -J llm.c_MultiGPU_4x_Train     # job name
#SBATCH -N 1                            # 1 node (single-node multi-GPU)
#SBATCH --gpus-per-node=4              # 4 GPUs
#SBATCH --ntasks-per-node=4            # 1 MPI task per GPU
#SBATCH --cpus-per-task=32             # OpenMP threads per MPI rank
#SBATCH -o llm.c_MultiGPU_4x_Train.o%j
#SBATCH -e llm.c_MultiGPU_4x_Train.e%j

# =============================================================================
# Training configuration — adjust these for your experiment
# =============================================================================
MODEL="d36"           # d12=124M, d24=350M, d36=774M
BATCH_SIZE=8          # sequences per GPU (8 × 4 GPUs = 32 total)
SEQ_LEN=1024          # context length: 1024, 2048, 4096, 8192
TOTAL_BATCH=524288    # total tokens per step (via gradient accumulation)
MAX_STEPS=20000
LEARNING_RATE=0.0006
GRAD_CLIP=0.1
LR_WARMUP_STEPS=0
VAL_STEPS=500
LOG_STEPS=1000
OUTPUT_DIR="MultiGPU_4x_log${MODEL}_${SEQ_LEN}l"
TRAIN_DATA="dev/data/fineweb10B/fineweb_train_*.bin"
VAL_DATA="dev/data/fineweb10B/fineweb_val_*.bin"
# =============================================================================

# Navigate to src/ (where Makefile.perlmutter and source files live)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/src"

# Load modules (Cray MPICH + NCCL for multi-GPU)
module load gcc/12.2.0
module load python/3.9
module load cudatoolkit/12.4
module load PrgEnv-gnu
module unload openmpi 2>/dev/null || true
module load cray-mpich/8.1.30

echo "Loading NCCL module..."
module load nccl/2.24.3 || { echo "ERROR: Failed to load NCCL module"; exit 1; }

# Export NCCL paths (module sets $NCCL_DIR; Makefile.perlmutter uses $NCCL_HOME)
if [ -n "$NCCL_DIR" ]; then
    export NCCL_HOME=$NCCL_DIR
    export LD_LIBRARY_PATH="$NCCL_HOME/lib:$LD_LIBRARY_PATH"
    export LIBRARY_PATH="$NCCL_HOME/lib:$LIBRARY_PATH"
    export CPATH="$NCCL_HOME/include:$CPATH"
    echo "NCCL_HOME: $NCCL_HOME"
else
    echo "ERROR: NCCL_DIR not set after module load — check nccl module"
    exit 1
fi

echo "=============================================="
echo "Multi-GPU Training — Standard Attention (4 GPUs, 1 node)"
echo "=============================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_JOB_NODELIST"
echo "GPUs:       4 (single node)"
echo "Model:      $MODEL"
echo "Batch/GPU:  $BATCH_SIZE  (total: $((BATCH_SIZE * 4)))"
echo "Seq length: $SEQ_LEN"
echo "Output dir: $OUTPUT_DIR"
echo "Start time: $(date)"
echo "=============================================="

nvidia-smi

conda activate env-llm.c
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment env-llm.c"
    exit 1
fi

export OMP_NUM_THREADS=32
export PYTHONUNBUFFERED=1

# NCCL settings for Perlmutter (Slingshot network)
export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=0
export NCCL_NET="AWS Libfabric"
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd

# CUDA paths
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4
export CUDA_MATH_LIBS=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/12.4/targets/x86_64-linux

# Cray MPICH paths (Makefile.perlmutter reads $MPICH_DIR)
export MPICH_DIR=${CRAY_MPICH_PREFIX:-/opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3}

# GTL (GPU Transport Layer) — required for GPU-aware MPI on Perlmutter
export MPICH_GPU_SUPPORT_ENABLED=1
if [ -d "/opt/cray/pe/mpich/8.1.30/gtl/lib" ]; then
    export GTL_LIBRARY_PATH=/opt/cray/pe/mpich/8.1.30/gtl/lib
    echo "GTL_LIBRARY_PATH: $GTL_LIBRARY_PATH"
else
    echo "WARNING: GTL library not found at expected path — GPU-aware MPI may not work"
fi

export LD_LIBRARY_PATH="$CUDA_MATH_LIBS/lib:$CUDA_HOME/lib64:$MPICH_DIR/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_MATH_LIBS/lib:$CUDA_HOME/lib64:$MPICH_DIR/lib:$LIBRARY_PATH"
export CPATH="$CUDA_HOME/include:$CUDA_MATH_LIBS/include:$MPICH_DIR/include:$CPATH"
export PATH="$CUDA_HOME/bin:$PATH"

[ -n "$GTL_LIBRARY_PATH" ] && export LD_LIBRARY_PATH="$GTL_LIBRARY_PATH:$LD_LIBRARY_PATH"
[ -n "$GTL_LIBRARY_PATH" ] && export LIBRARY_PATH="$GTL_LIBRARY_PATH:$LIBRARY_PATH"

export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache

# Verify training data
ls $TRAIN_DATA 1>/dev/null 2>&1 || { echo "ERROR: Training data not found: $TRAIN_DATA"; exit 1; }
ls $VAL_DATA   1>/dev/null 2>&1 || { echo "ERROR: Validation data not found: $VAL_DATA"; exit 1; }
echo "Training data verified"

# Build — multi-GPU with NCCL + Cray MPICH, no cuDNN
echo "=============================================="
echo "Building train_gpt2cu (standard attention, multi-GPU)..."
make -f Makefile.perlmutter clean
make -f Makefile.perlmutter train_gpt2cu

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed"
    exit 1
fi

# Verify NCCL linking
if ldd ./train_gpt2cu | grep -q nccl; then
    echo "NCCL linked — multi-GPU enabled"
else
    echo "WARNING: NCCL not found in binary — training may run single-GPU only"
fi
echo "Build successful: $(ls -lh train_gpt2cu)"

# Run training — 4 GPUs on 1 node via MPI
echo "=============================================="
echo "Starting 4-GPU training: $MODEL | seq=$SEQ_LEN | batch/GPU=$BATCH_SIZE"
echo "Start time: $(date)"
echo "=============================================="

srun --ntasks=4 \
     --ntasks-per-node=4 \
     --gpus-per-node=4 \
     --cpus-per-task=32 \
     --cpu-bind=cores \
     bash -c "
       [ -n \"$GTL_LIBRARY_PATH\" ] && export LD_LIBRARY_PATH=\"$GTL_LIBRARY_PATH:\$LD_LIBRARY_PATH\"
       export MPICH_GPU_SUPPORT_ENABLED=1
       echo \"Rank \$SLURM_PROCID on \$(hostname)\"
       ./train_gpt2cu \
         -i \"$TRAIN_DATA\" \
         -j \"$VAL_DATA\" \
         -o $OUTPUT_DIR \
         -e $MODEL \
         -b $BATCH_SIZE \
         -t $SEQ_LEN \
         -d $TOTAL_BATCH \
         -r 1 \
         -z 1 \
         -c $GRAD_CLIP \
         -l $LEARNING_RATE \
         -q 0.1 \
         -u $LR_WARMUP_STEPS \
         -x $MAX_STEPS \
         -n $LOG_STEPS \
         -v $VAL_STEPS \
         -s 0 \
         -h 1 \
         -pi mpi
     "

TRAIN_EXIT_CODE=$?
echo "Training completed at $(date) — exit code: $TRAIN_EXIT_CODE"
[ $TRAIN_EXIT_CODE -ne 0 ] && exit $TRAIN_EXIT_CODE
echo "SUCCESS: 4-GPU standard attention training complete"
ls -la "$OUTPUT_DIR"/ 2>/dev/null || true
