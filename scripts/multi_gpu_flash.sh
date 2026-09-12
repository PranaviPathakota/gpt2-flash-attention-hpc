#!/bin/bash
#SBATCH -A <your_account>               # NERSC account
#SBATCH -C "gpu&hbm40g"               # 40GB A100 GPU nodes
#SBATCH -q regular                      # queue
#SBATCH -t 01:00:00                     # time limit
#SBATCH -J llm.c_MultiGPU_4x_Flash     # job name
#SBATCH -N 1                            # 1 node (single-node multi-GPU)
#SBATCH --gpus-per-node=4              # 4 GPUs
#SBATCH --ntasks-per-node=4            # 1 MPI task per GPU
#SBATCH --cpus-per-task=32             # OpenMP threads per MPI rank
#SBATCH -o llm.c_MultiGPU_4x_Flash.o%j
#SBATCH -e llm.c_MultiGPU_4x_Flash.e%j

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
OUTPUT_DIR="MultiGPU_4x_log${MODEL}_${SEQ_LEN}l_FlashAttention"
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
echo "Multi-GPU Training — Flash Attention (4 GPUs, 1 node)"
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

# cuDNN setup — libraries installed via conda (nvidia-cudnn package)
export CUDNN_LIB="$CONDA_PREFIX/lib"
export CUDNN_INCLUDE="$CONDA_PREFIX/lib/python3.8/site-packages/nvidia/cudnn/include"

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

export LD_LIBRARY_PATH="$CUDNN_LIB:$CUDA_MATH_LIBS/lib:$CUDA_HOME/lib64:$MPICH_DIR/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDNN_LIB:$CUDA_MATH_LIBS/lib:$CUDA_HOME/lib64:$MPICH_DIR/lib:$LIBRARY_PATH"
export CPATH="$CUDNN_INCLUDE:$CUDNN_FRONTEND_PATH:$CUDA_HOME/include:$CUDA_MATH_LIBS/include:$MPICH_DIR/include:$CPATH"
export PATH="$CUDA_HOME/bin:$PATH"

[ -n "$GTL_LIBRARY_PATH" ] && export LD_LIBRARY_PATH="$GTL_LIBRARY_PATH:$LD_LIBRARY_PATH"
[ -n "$GTL_LIBRARY_PATH" ] && export LIBRARY_PATH="$GTL_LIBRARY_PATH:$LIBRARY_PATH"

export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache

echo "CUDNN_LIB:           $CUDNN_LIB"
echo "CUDNN_INCLUDE:       $CUDNN_INCLUDE"
echo "CUDNN_FRONTEND_PATH: $CUDNN_FRONTEND_PATH"
echo "MPICH_DIR:           $MPICH_DIR"
echo "GTL_LIBRARY_PATH:    ${GTL_LIBRARY_PATH:-NOT SET}"

# Verify cuDNN
if [ ! -f "$CUDNN_LIB/libcudnn.so" ]; then
    echo "ERROR: libcudnn.so not found at $CUDNN_LIB"
    exit 1
fi
if [ ! -f "$CUDNN_INCLUDE/cudnn.h" ]; then
    echo "ERROR: cudnn.h not found at $CUDNN_INCLUDE"
    exit 1
fi

# Quick cuDNN link test
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

# Build — multi-GPU + cuDNN Flash Attention
echo "=============================================="
echo "Building train_gpt2cu (Flash Attention, multi-GPU)..."
make -f Makefile.perlmutter clean

echo "Step 1: Compiling cuDNN attention module..."
make -f Makefile.perlmutter build/cudnn_att.o USE_CUDNN=1
if [ $? -ne 0 ]; then
    echo "ERROR: cuDNN attention module compilation failed"
    exit 1
fi

echo "Step 2: Building main training executable..."
make -f Makefile.perlmutter train_gpt2cu USE_CUDNN=1
if [ $? -ne 0 ]; then
    echo "ERROR: Build failed"
    exit 1
fi

# Verify cuDNN + NCCL linking
if ldd ./train_gpt2cu | grep -q cudnn; then
    echo "cuDNN Flash Attention ENABLED"
    ldd ./train_gpt2cu | grep cudnn
else
    echo "ERROR: cuDNN not linked in binary"
    exit 1
fi
if ldd ./train_gpt2cu | grep -q nccl; then
    echo "NCCL linked — multi-GPU enabled"
else
    echo "WARNING: NCCL not found in binary"
fi
echo "Build successful: $(ls -lh train_gpt2cu)"

# Run training — 4 GPUs on 1 node via MPI
echo "=============================================="
echo "Starting 4-GPU Flash Attention training: $MODEL | seq=$SEQ_LEN | batch/GPU=$BATCH_SIZE"
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
echo "SUCCESS: 4-GPU Flash Attention training complete"
ls -la "$OUTPUT_DIR"/ 2>/dev/null || true
