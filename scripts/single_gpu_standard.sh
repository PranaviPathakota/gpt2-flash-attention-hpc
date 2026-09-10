#!/bin/bash
#SBATCH -A m4012                        # NERSC account
#SBATCH -C gpu                          # 40GB A100 GPU nodes
#SBATCH -q regular                      # queue
#SBATCH -t 01:00:00                     # time limit
#SBATCH -J llm.c_SingleGPU_Train        # job name
#SBATCH -N 1                            # 1 node
#SBATCH --gpus-per-node=1              # 1 GPU
#SBATCH --ntasks-per-node=1            # no MPI
#SBATCH --cpus-per-task=32             # OpenMP threads
#SBATCH -o llm.c_SingleGPU_Train.o%j
#SBATCH -e llm.c_SingleGPU_Train.e%j

# =============================================================================
# Training configuration — adjust these for your experiment
# =============================================================================
MODEL="d12"           # d12=124M, d24=350M, d36=774M
BATCH_SIZE=32         # sequences per GPU
SEQ_LEN=1024          # context length: 1024, 2048, 4096, 8192
TOTAL_BATCH=524288    # total tokens per step (via gradient accumulation)
MAX_STEPS=20000
LEARNING_RATE=0.0006
GRAD_CLIP=0.1
LR_WARMUP_STEPS=0
VAL_STEPS=500
LOG_STEPS=1000
OUTPUT_DIR="SingleGPU_1x_log${MODEL}_${SEQ_LEN}l"
TRAIN_DATA="dev/data/fineweb10B/fineweb_train_*.bin"
VAL_DATA="dev/data/fineweb10B/fineweb_val_*.bin"
# =============================================================================

# Navigate to src/ (where Makefile.perlmutter and source files live)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/src"

# Load modules (no MPI, no NCCL for single GPU)
module load gcc/12.2.0
module load python/3.9
module load cudatoolkit/12.4
module load PrgEnv-gnu

echo "=============================================="
echo "Single GPU Training — Standard Attention"
echo "=============================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Model:      $MODEL"
echo "Batch size: $BATCH_SIZE"
echo "Seq length: $SEQ_LEN"
echo "Output dir: $OUTPUT_DIR"
echo "Start time: $(date)"
echo "=============================================="

nvidia-smi

# Activate conda environment
conda activate env-llm.c
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment env-llm.c"
    exit 1
fi

export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# CUDA paths (Perlmutter HPC SDK)
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4
export CUDA_MATH_LIBS=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/12.4/targets/x86_64-linux
export LD_LIBRARY_PATH="$CUDA_MATH_LIBS/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_MATH_LIBS/lib:$CUDA_HOME/lib64:$LIBRARY_PATH"
export CPATH="$CUDA_HOME/include:$CUDA_MATH_LIBS/include:$CPATH"
export PATH="$CUDA_HOME/bin:$PATH"

export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache

echo "CUDA_HOME: $CUDA_HOME"
echo "NVCC:      $(which nvcc || echo 'NOT FOUND')"
echo "GCC:       $(gcc --version | head -1)"

# Verify training data
ls $TRAIN_DATA 1>/dev/null 2>&1 || { echo "ERROR: Training data not found: $TRAIN_DATA"; exit 1; }
ls $VAL_DATA   1>/dev/null 2>&1 || { echo "ERROR: Validation data not found: $VAL_DATA"; exit 1; }
echo "Training data verified"

# Build — single GPU: disable NCCL and MPI
echo "=============================================="
echo "Building train_gpt2cu (standard attention, single GPU)..."
make -f Makefile.perlmutter clean
make -f Makefile.perlmutter train_gpt2cu NO_MULTI_GPU=1 NO_USE_MPI=1

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed"
    exit 1
fi
echo "Build successful: $(ls -lh train_gpt2cu)"

# Run training (no srun — single process, no MPI)
echo "=============================================="
echo "Starting training: $MODEL | seq=$SEQ_LEN | batch=$BATCH_SIZE"
echo "Start time: $(date)"
echo "=============================================="

./train_gpt2cu \
    -i "$TRAIN_DATA" \
    -j "$VAL_DATA" \
    -o "$OUTPUT_DIR" \
    -e "$MODEL" \
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
    -h 1

TRAIN_EXIT_CODE=$?
echo "Training completed at $(date) — exit code: $TRAIN_EXIT_CODE"
[ $TRAIN_EXIT_CODE -ne 0 ] && exit $TRAIN_EXIT_CODE
echo "SUCCESS: single GPU standard attention training complete"
ls -la "$OUTPUT_DIR"/ 2>/dev/null || true
