#!/bin/bash
#SBATCH -A m4012                 # replace with your NERSC account
#SBATCH -C gpu                  # use "cpu" for CPU nodes, "gpu" for GPU nodes
#SBATCH -q regular              # queue: regular or debug
#SBATCH -t 00:01:00              # time limit (hh:mm:ss)
#SBATCH -J llm.cGpt2-2_Train_Flash_Attention          # job name
#SBATCH -N 1                    # number of nodes
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32        # OpenMP threads per MPI rank
#SBATCH -o llm.cGpt2-2_Train-cudNN.o%j               # stdout file
#SBATCH -e llm.cGpt2-2_Train-cudNN.e%j               # stderr file

# Load required modules
module load gcc/12.2.0
module load python/3.9
module load cuda/12.4
module load PrgEnv-gnu
module load openmpi/5.0.3

echo "=============================================="
echo "🚀 cuDNN Flash Attention Training Script"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME" 
echo "Start time: $(date)"
echo "=============================================="

nvidia-smi

# Activate conda environment
conda activate env-llm.c
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment"
    exit 1
fi

echo "✅ Environment: $CONDA_PREFIX"

# Set up cuDNN environment (the symlinks should already exist from recovery script)
export CUDNN_LIB="$CONDA_PREFIX/lib"
export CUDNN_INCLUDE="$CONDA_PREFIX/lib/python3.8/site-packages/nvidia/cudnn/include"

# Verify cuDNN is available
echo "Verifying cuDNN setup..."
echo "cuDNN libraries: $(ls $CUDNN_LIB/libcudnn* 2>/dev/null | wc -l) files"
echo "cuDNN headers: $(ls $CUDNN_INCLUDE/cudnn*.h 2>/dev/null | wc -l) files"

if [ ! -f "$CUDNN_LIB/libcudnn.so" ] || [ ! -f "$CUDNN_INCLUDE/cudnn.h" ]; then
    echo "❌ cuDNN not properly configured"
    echo "Run the recovery script first: bash advanced_cudnn_recovery.sh"
    exit 1
fi

# Set up cuDNN frontend
if [ ! -d "cudnn-frontend" ]; then
    echo "Cloning cuDNN frontend..."
    git clone https://github.com/NVIDIA/cudnn-frontend.git
fi

if [ ! -d "cudnn-frontend/include" ]; then
    echo "❌ cuDNN frontend not available"
    exit 1
fi

export CUDNN_FRONTEND_PATH="$(pwd)/cudnn-frontend/include"

# Set environment variables
export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4
export CUDA_MATH_LIBS=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/12.4/targets/x86_64-linux

# Enhanced library paths with cuDNN
export LD_LIBRARY_PATH="$CUDNN_LIB:$CUDA_MATH_LIBS/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDNN_LIB:$CUDA_MATH_LIBS/lib:$CUDA_HOME/lib64:$LIBRARY_PATH"
export CPATH="$CUDNN_INCLUDE:$CUDNN_FRONTEND_PATH:$CUDA_HOME/include:$CUDA_MATH_LIBS/include:$CPATH"
export PATH="$CUDA_HOME/bin:$PATH"

# HuggingFace cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache

echo "=============================================="
echo "Final Environment Check:"
echo "CUDNN_LIB: $CUDNN_LIB"
echo "CUDNN_INCLUDE: $CUDNN_INCLUDE"
echo "CUDNN_FRONTEND_PATH: $CUDNN_FRONTEND_PATH"
echo "NVCC: $(which nvcc)"

# Quick cuDNN link test
echo ""
echo "Testing cuDNN linking..."
echo "int main(){return 0;}" > quick_test.c
gcc quick_test.c -L"$CUDNN_LIB" -lcudnn -o quick_test 2>&1
if [ $? -eq 0 ]; then
    echo "✅ cuDNN linking confirmed"
    rm -f quick_test.c quick_test
else
    echo "❌ cuDNN linking failed"
    exit 1
fi

echo "=============================================="

# Check training data
if ! ls dev/data/fineweb10B/fineweb_train_*.bin 1> /dev/null 2>&1; then
    echo "❌ Training data not found"
    exit 1
fi

if ! ls dev/data/fineweb10B/fineweb_val_*.bin 1> /dev/null 2>&1; then
    echo "❌ Validation data not found"
    exit 1
fi

echo "✅ Training data verified"

# Create build directory
mkdir -p build

# Replace Makefile to use our cuDNN configuration
echo "Creating cuDNN Makefile..."
cat > Makefile << 'EOF'
CC ?= clang
CFLAGS = -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes
NVCC_FLAGS = --threads=0 -t=0 --use_fast_math -std=c++17 -O3
NVCC_LDFLAGS = -lcublas -lcublasLt -lnvidia-ml
BUILD_DIR = build
USE_CUDNN = 1

$(shell mkdir -p $(BUILD_DIR))

NVCC := $(shell which nvcc 2>/dev/null)

# GPU compute capability
GPU_COMPUTE_CAPABILITY = $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/\.//g' | sort -n | head -n 1)
ifneq ($(GPU_COMPUTE_CAPABILITY),)
  NVCC_FLAGS += --generate-code arch=compute_$(GPU_COMPUTE_CAPABILITY),code=[compute_$(GPU_COMPUTE_CAPABILITY),sm_$(GPU_COMPUTE_CAPABILITY)]
endif

# cuDNN configuration using environment variables
ifeq ($(USE_CUDNN), 1)
  $(info ✓ Building with cuDNN Flash Attention support)
  NVCC_INCLUDES = -I$(CUDNN_INCLUDE) -I$(CUDNN_FRONTEND_PATH)
  NVCC_LDFLAGS += -L$(CUDNN_LIB) -lcudnn
  NVCC_FLAGS += -DENABLE_CUDNN
  NVCC_CUDNN = $(BUILD_DIR)/cudnn_att.o
  TARGETS = train_gpt2cu $(NVCC_CUDNN)
else
  $(info → Building without cuDNN)
  TARGETS = train_gpt2cu
endif

# Precision
PRECISION ?= BF16
ifeq ($(PRECISION), FP32)
  PFLAGS = -DENABLE_FP32
else ifeq ($(PRECISION), FP16)
  PFLAGS = -DENABLE_FP16
else
  PFLAGS = -DENABLE_BF16
endif

.PHONY: all clean

all: $(TARGETS)

$(NVCC_CUDNN): llmc/cudnn_att.cpp
	@echo "🔧 Compiling cuDNN attention module..."
	$(NVCC) -c $(NVCC_FLAGS) $(PFLAGS) $^ $(NVCC_INCLUDES) -o $@

train_gpt2cu: train_gpt2.cu $(NVCC_CUDNN)
	@echo "🚀 Building train_gpt2cu with cuDNN Flash Attention..."
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) -o $@

clean:
	rm -f train_gpt2cu test_gpt2cu $(BUILD_DIR)/*.o
EOF

echo "✅ cuDNN Makefile created"

# Clean and build with cuDNN
echo "=============================================="
echo "Building with cuDNN Flash Attention..."

make clean

# Export environment variables for make
export CUDNN_INCLUDE
export CUDNN_LIB  
export CUDNN_FRONTEND_PATH

# Build step by step for better debugging
echo "Step 1: Building cuDNN attention module..."
make build/cudnn_att.o

if [ $? -eq 0 ]; then
    echo "✅ cuDNN attention module compiled successfully"
else
    echo "❌ cuDNN attention module compilation failed"
    exit 1
fi

echo "Step 2: Building main training executable..."
make train_gpt2cu

if [ $? -eq 0 ]; then
    echo "🎉 SUCCESS: train_gpt2cu built with cuDNN Flash Attention!"
else
    echo "❌ Main executable build failed"
    exit 1
fi

# Verify the executable and cuDNN linking
if [ -x "./train_gpt2cu" ]; then
    echo "✅ Executable created: $(ls -lh ./train_gpt2cu)"
    
    # Check cuDNN linking
    echo "Checking cuDNN integration..."
    if ldd ./train_gpt2cu | grep -q cudnn; then
        echo "🚀 cuDNN Flash Attention is ENABLED!"
        ldd ./train_gpt2cu | grep cudnn
    else
        echo "❌ cuDNN not properly linked"
        exit 1
    fi
else
    echo "❌ Executable not found"
    exit 1
fi

# Run the training with cuDNN Flash Attention
echo "=============================================="
echo "🚀 Starting Training with cuDNN Flash Attention"
echo "Start time: $(date)"
echo "=============================================="

./train_gpt2cu \
    -i "dev/data/fineweb10B/fineweb_train_*.bin" \
    -j "dev/data/fineweb10B/fineweb_val_*.bin" \
    -o SingleGPU_1x_log124M_1024l_FlashAttention \
    -e "d12" \
    -b 32 \
    -t 1024 \
    -d 524288 \
    -r 1 \
    -z 1 \
    -c 0.1 \
    -l 0.0006 \
    -q 0.1 \
    -u 0 \
    -x 20000 \
    -n 1000 \
    -v 500 \
    -s 0 \
    -h 1

TRAIN_EXIT_CODE=$?

echo "=============================================="
echo "Training completed at $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "🎉 SUCCESS: Training with cuDNN Flash Attention completed!"
    echo "Output directory:"
    ls -la SingleGPU_1x_log124M_1024l_FlashAttention/ 2>/dev/null || echo "Check SingleGPU_1x_log124M_1024l_FlashAttention directory"
    echo ""
    echo "🚀 Flash Attention was used for optimal performance!"
else
    echo "❌ Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo "=============================================="
echo "✨ cuDNN Flash Attention Training Complete! ✨"
echo "=============================================="
