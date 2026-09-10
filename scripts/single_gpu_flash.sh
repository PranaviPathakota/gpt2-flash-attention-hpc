#!/bin/bash
#SBATCH -A m4012                        # NERSC account
#SBATCH -C gpu                          # 40GB A100 GPU nodes
#SBATCH -q regular                      # queue
#SBATCH -t 01:00:00                     # time limit
#SBATCH -J llm.c_SingleGPU_Flash        # job name
#SBATCH -N 1                            # 1 node
#SBATCH --gpus-per-node=1              # 1 GPU
#SBATCH --ntasks-per-node=1            # no MPI
#SBATCH --cpus-per-task=32             # OpenMP threads
#SBATCH -o llm.c_SingleGPU_Flash.o%j
#SBATCH -e llm.c_SingleGPU_Flash.e%j

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
OUTPUT_DIR="SingleGPU_1x_log${MODEL}_${SEQ_LEN}l_FlashAttention"
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
echo "Single GPU Training — Flash Attention (cuDNN)"
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

# cuDNN setup — libraries installed via conda (nvidia-cudnn package)
export CUDNN_LIB="$CONDA_PREFIX/lib"
export CUDNN_INCLUDE="$CONDA_PREFIX/lib/python3.8/site-packages/nvidia/cudnn/include"
export LD_LIBRARY_PATH="$CUDNN_LIB:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDNN_LIB:$LIBRARY_PATH"
export CPATH="$CUDNN_INCLUDE:$CPATH"

# cuDNN frontend (header-only, cloned once)
if [ ! -d "cudnn-frontend" ]; then
    echo "Cloning cuDNN frontend..."
    git clone https://github.com/NVIDIA/cudnn-frontend.git
fi
if [ ! -d "cudnn-frontend/include" ]; then
    echo "ERROR: cuDNN frontend not available at cudnn-frontend/include"
    exit 1
fi
export CUDNN_FRONTEND_PATH="$(pwd)/cudnn-frontend/include"

echo "CUDA_HOME:           $CUDA_HOME"
echo "CUDNN_LIB:           $CUDNN_LIB"
echo "CUDNN_INCLUDE:       $CUDNN_INCLUDE"
echo "CUDNN_FRONTEND_PATH: $CUDNN_FRONTEND_PATH"

# Verify cuDNN is properly installed
if [ ! -f "$CUDNN_LIB/libcudnn.so" ]; then
    echo "ERROR: libcudnn.so not found at $CUDNN_LIB"
    echo "Install with: conda install -c nvidia cudnn"
    exit 1
fi
if [ ! -f "$CUDNN_INCLUDE/cudnn.h" ]; then
    echo "ERROR: cudnn.h not found at $CUDNN_INCLUDE"
    exit 1
fi

# Quick link test
echo "Testing cuDNN linking..."
echo "int main(){return 0;}" > /tmp/quick_test.c
gcc /tmp/quick_test.c -L"$CUDNN_LIB" -lcudnn -o /tmp/quick_test 2>&1
if [ $? -eq 0 ]; then
    echo "cuDNN linking confirmed"
    rm -f /tmp/quick_test.c /tmp/quick_test
else
    echo "ERROR: cuDNN linking failed"
    exit 1
fi

# Verify training data
ls $TRAIN_DATA 1>/dev/null 2>&1 || { echo "ERROR: Training data not found: $TRAIN_DATA"; exit 1; }
ls $VAL_DATA   1>/dev/null 2>&1 || { echo "ERROR: Validation data not found: $VAL_DATA"; exit 1; }
echo "Training data verified"

# Build — single GPU + cuDNN Flash Attention; disable NCCL and MPI
echo "=============================================="
echo "Building train_gpt2cu (Flash Attention, single GPU)..."
make -f Makefile.perlmutter clean

echo "Step 1: Compiling cuDNN attention module..."
make -f Makefile.perlmutter build/cudnn_att.o NO_MULTI_GPU=1 NO_USE_MPI=1 USE_CUDNN=1
if [ $? -ne 0 ]; then
    echo "ERROR: cuDNN attention module compilation failed"
    exit 1
fi

echo "Step 2: Building main training executable..."
make -f Makefile.perlmutter train_gpt2cu NO_MULTI_GPU=1 NO_USE_MPI=1 USE_CUDNN=1
if [ $? -ne 0 ]; then
    echo "ERROR: Build failed"
    exit 1
fi

# Verify cuDNN is linked
if ldd ./train_gpt2cu | grep -q cudnn; then
    echo "cuDNN Flash Attention is ENABLED"
    ldd ./train_gpt2cu | grep cudnn
else
    echo "ERROR: cuDNN not linked in binary"
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
echo "SUCCESS: single GPU Flash Attention training complete"
ls -la "$OUTPUT_DIR"/ 2>/dev/null || true
