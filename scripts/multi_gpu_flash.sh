#!/bin/bash
#SBATCH -A m4012                 # replace with your NERSC account
#SBATCH -C "gpu&hbm40g"         # Explicitly request 40GB GPU nodes (use quotes!)
#SBATCH -q regular              # queue: regular or debug
#SBATCH -t 01:00:00              # time limit (hh:mm:ss)
#SBATCH -J llm.c_MultiGPU_Flash_Attention          # job name
#SBATCH -N 4                    # number of nodes
#SBATCH --gpus-per-node=4       # 4 GPUs for multi-GPU training
#SBATCH --ntasks-per-node=4     # MPI tasks (one per GPU)
#SBATCH --cpus-per-task=32      # OpenMP threads per MPI rank
#SBATCH -o llm.c_MultiGPU_16x_1k_Flash_Attention.o%j     # stdout file
#SBATCH -e llm.c_MultiGPU_16x_1k_Flash_Attention.e%j     # stderr file

# Load required modules
module load gcc/12.2.0
module load python/3.9
module load cudatoolkit/12.4
module load PrgEnv-gnu
# Use Cray MPICH instead of OpenMPI - it's native to Perlmutter
module unload openmpi 2>/dev/null || true
module load cray-mpich/8.1.30

# Load NCCL module (needed for multi-GPU)
echo "Loading NCCL module..."
module load nccl/2.24.3 || echo "⚠️  Failed to load NCCL module"

# Export NCCL paths explicitly after loading module
if [ -n "$NCCL_DIR" ]; then
    export NCCL_HOME=$NCCL_DIR
    export LD_LIBRARY_PATH=$NCCL_DIR/lib:$LD_LIBRARY_PATH
    export LIBRARY_PATH=$NCCL_DIR/lib:$LIBRARY_PATH
    export CPATH=$NCCL_DIR/include:$CPATH
    echo "✅ NCCL paths set: $NCCL_DIR"
elif module show nccl/2.24.3 2>&1 | grep -q "NCCL_DIR"; then
    # Extract NCCL_DIR from module
    NCCL_DIR=$(module show nccl/2.24.3 2>&1 | grep "setenv.*NCCL_DIR" | awk '{print $3}')
    export NCCL_HOME=$NCCL_DIR
    export LD_LIBRARY_PATH=$NCCL_DIR/lib:$LD_LIBRARY_PATH
    export LIBRARY_PATH=$NCCL_DIR/lib:$LIBRARY_PATH
    export CPATH=$NCCL_DIR/include:$CPATH
    echo "✅ NCCL paths set: $NCCL_DIR"
fi

echo "=============================================="
echo "🚀 Multi-GPU Flash Attention Training Script (16 GPUs across 4 nodes)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST" 
echo "Total GPUs: 16 (4 nodes × 4 GPUs)"
echo "Start time: $(date)"
echo "Python version: $(python --version)"
echo "=============================================="

# Display GPU information
nvidia-smi

# Activate conda environment
echo "Activating conda environment..."
conda activate env-llm.c
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment env-llm.c"
    exit 1
fi
echo "✅ Conda environment activated: $CONDA_PREFIX"

# Set environment variables for optimal performance
export OMP_NUM_THREADS=32
export PYTHONUNBUFFERED=1

# NCCL settings for multi-node training on Perlmutter (Slingshot network)
export NCCL_DEBUG=WARN  # Changed from INFO to WARN - only show warnings/errors
# export NCCL_DEBUG_SUBSYS=INIT,COLL  # Commented out - removes operation-level logging
export NCCL_NET_GDR_LEVEL=PHB        # GPU Direct RDMA at PCIe Host Bridge level
export NCCL_CROSS_NIC=1              # Enable cross-NIC for better performance
export NCCL_COLLNET_ENABLE=0         # Disable CollNet (not needed for basic setup)
export NCCL_NET="AWS Libfabric"      # Use Libfabric for Slingshot (Cray's network)
export FI_CXI_DISABLE_HOST_REGISTER=1  # Needed for Slingshot
export FI_MR_CACHE_MONITOR=userfaultfd # Memory registration cache

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

# Set CUDA paths (use cudatoolkit paths if available, fallback to cuda)
if [ -d "/opt/nvidia/hpc_sdk/Linux_x86_64/cudatoolkit" ]; then
    export CUDA_HOME=$(find /opt/nvidia/hpc_sdk/Linux_x86_64 -name "cudatoolkit" -type d 2>/dev/null | head -1)/12.4
elif [ -d "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4" ]; then
    export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4
else
    export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
fi

export CUDA_MATH_LIBS=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/12.4/targets/x86_64-linux

# Set MPI paths for Cray MPICH (need to set before building)
export MPICH_DIR=${CRAY_MPICH_PREFIX:-/opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3}
export CPATH="${MPICH_DIR}/include:${CPATH}"
export LIBRARY_PATH="${MPICH_DIR}/lib:${LIBRARY_PATH}"

# CRITICAL: Set GTL (GPU Transport Layer) paths for GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1
if [ -d "/opt/cray/pe/mpich/8.1.30/gtl/lib" ]; then
    export GTL_LIBRARY_PATH=/opt/cray/pe/mpich/8.1.30/gtl/lib
    export LD_LIBRARY_PATH="${GTL_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
    export LIBRARY_PATH="${GTL_LIBRARY_PATH}:${LIBRARY_PATH}"
    echo "✅ GTL library path set: $GTL_LIBRARY_PATH"
else
    echo "⚠️  GTL library not found - multi-GPU may not work"
fi

# Enhanced library paths with cuDNN and MPI
export LD_LIBRARY_PATH="$CUDNN_LIB:$CUDA_MATH_LIBS/lib:$CUDA_HOME/lib64:$MPICH_DIR/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDNN_LIB:$CUDA_MATH_LIBS/lib:$CUDA_HOME/lib64:$MPICH_DIR/lib:$LIBRARY_PATH"
export CPATH="$CUDNN_INCLUDE:$CUDNN_FRONTEND_PATH:$CUDA_HOME/include:$CUDA_MATH_LIBS/include:$MPICH_DIR/include:$CPATH"
export PATH="$CUDA_HOME/bin:$PATH"

# Set HuggingFace cache directories
export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache

echo "=============================================="
echo "Environment Setup Complete:"
echo "CUDA_HOME: $CUDA_HOME"
echo "NVCC: $(which nvcc || echo 'NOT FOUND')"
echo "GCC: $(gcc --version | head -1)"
echo "CUDNN_LIB: $CUDNN_LIB"
echo "CUDNN_INCLUDE: $CUDNN_INCLUDE"
echo "CUDNN_FRONTEND_PATH: $CUDNN_FRONTEND_PATH"
echo "MPICH_DIR: $MPICH_DIR"
echo "GTL_LIBRARY_PATH: ${GTL_LIBRARY_PATH:-NOT SET}"
echo "MPICH_GPU_SUPPORT_ENABLED: $MPICH_GPU_SUPPORT_ENABLED"
echo ""
echo "Multi-GPU Requirements:"
echo "MPI: $(which mpirun || which mpiexec || echo 'NOT FOUND')"
echo "MPI Type: Cray MPICH (native to Perlmutter)"
echo "NCCL check:"
if [ -n "$NCCL_HOME" ] && [ -f "$NCCL_HOME/lib/libnccl.so" ]; then
    echo "  ✅ NCCL found via NCCL_HOME: $NCCL_HOME"
    ls -la $NCCL_HOME/lib/libnccl.so* | head -3
elif [ -n "$NCCL_DIR" ] && [ -f "$NCCL_DIR/lib/libnccl.so" ]; then
    echo "  ✅ NCCL found via NCCL_DIR: $NCCL_DIR"
    ls -la $NCCL_DIR/lib/libnccl.so* | head -3
elif find /usr -name "*libnccl*" -type f 2>/dev/null | grep -q .; then
    echo "  ✅ NCCL found in /usr"
    find /usr -name "*libnccl*" -type f 2>/dev/null | head -3
elif [ -n "$CUDA_HOME" ] && find $CUDA_HOME/lib64 -name "*libnccl*" 2>/dev/null | grep -q .; then
    echo "  ✅ NCCL found in CUDA"
    find $CUDA_HOME/lib64 -name "*libnccl*" 2>/dev/null | head -3
else
    echo "  ❌ NCCL NOT FOUND - multi-GPU training WILL BE DISABLED"
    echo "  This is CRITICAL for multi-GPU training!"
    echo "  Make sure NCCL module loaded correctly"
fi
echo "Number of available GPUs per node: $(nvidia-smi -L | wc -l)"
echo "=============================================="

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

# NCCL detection for multi-GPU support
ifeq ($(NO_MULTI_GPU), 1)
  $(info → Multi-GPU manually disabled)
else
  # Check NCCL_HOME first (set by module or environment)
  ifdef NCCL_HOME
    NCCL_FOUND := $(shell [ -f "$(NCCL_HOME)/lib/libnccl.so" ] && echo "exists")
    ifeq ($(NCCL_FOUND), exists)
      $(info ✓ NCCL found in NCCL_HOME, multi-GPU enabled)
      NVCC_INCLUDES += -I$(NCCL_HOME)/include
      NVCC_LDFLAGS += -L$(NCCL_HOME)/lib
      NVCC_FLAGS += -DMULTI_GPU
      NVCC_LDFLAGS += -lnccl
    endif
  else
    # Fallback: Check using multiple methods for different Linux distributions
    NCCL_FOUND := $(shell (command -v rpm >/dev/null 2>&1 && rpm -qa 2>/dev/null | grep -q nccl) || (command -v dpkg >/dev/null 2>&1 && dpkg -l 2>/dev/null | grep -q nccl) || (find /usr -name "*libnccl*" -type f 2>/dev/null | head -1 | grep -q .) || (find $(CUDA_HOME)/lib64 -name "*libnccl*" 2>/dev/null | head -1 | grep -q .) && echo "exists")
    ifeq ($(NCCL_FOUND), exists)
      $(info ✓ NCCL found in system paths, multi-GPU enabled)
      NVCC_FLAGS += -DMULTI_GPU
      NVCC_LDFLAGS += -lnccl
    else
      $(warning ✗ NCCL not found, multi-GPU will be disabled!)
      $(warning   Set NCCL_HOME or load nccl module for multi-GPU support)
    endif
  endif
endif

# MPI detection for multi-GPU support
OPENMPI_DIR ?= /usr/lib/x86_64-linux-gnu/openmpi
ifeq ($(NO_USE_MPI), 1)
  $(info → MPI manually disabled)
else
  # Check for Cray MPICH first (using MPICH_DIR from environment)
  ifdef MPICH_DIR
    MPI_TEST := $(shell [ -f "$(MPICH_DIR)/include/mpi.h" ] && echo "exists")
    ifeq ($(MPI_TEST), exists)
      $(info ✓ Cray MPICH found at $(MPICH_DIR))
      NVCC_INCLUDES += -I$(MPICH_DIR)/include
      NVCC_LDFLAGS += -L$(MPICH_DIR)/lib
      NVCC_LDFLAGS += -lmpi
      # Add GTL library if available (required for GPU-aware MPI)
      ifdef GTL_LIBRARY_PATH
        $(info ✓ GTL library found at $(GTL_LIBRARY_PATH))
        NVCC_LDFLAGS += -L$(GTL_LIBRARY_PATH)
        NVCC_LDFLAGS += -lmpi_gtl_cuda
      endif
      NVCC_FLAGS += -DUSE_MPI
    else
      $(warning ✗ MPICH_DIR set but mpi.h not found)
    endif
  # Check standard OpenMPI location
  else ifeq ($(shell [ -d $(OPENMPI_DIR)/lib ] && [ -d $(OPENMPI_DIR)/include ] && echo "exists"), exists)
    $(info ✓ MPI found at $(OPENMPI_DIR))
    NVCC_INCLUDES += -I$(OPENMPI_DIR)/include
    NVCC_LDFLAGS += -L$(OPENMPI_DIR)/lib
    NVCC_LDFLAGS += -lmpi
    NVCC_FLAGS += -DUSE_MPI
  else
    $(warning ✗ MPI not found, multi-GPU will be disabled!)
    $(warning   Set MPICH_DIR environment variable or install MPI)
  endif
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

# -------------------------------
# Run the training with Multi-GPU (16 GPUs across 4 nodes)
# -------------------------------
echo "=============================================="
echo "🚀 Starting Multi-Node Multi-GPU Training with cuDNN Flash Attention"
echo "Configuration: 4 nodes × 4 GPUs = 16 GPUs total"
echo "Attention: Flash Attention (cuDNN)"
echo "Start time: $(date)"
echo "=============================================="

# Use srun (SLURM's native launcher) - works best with Cray MPICH on Perlmutter
# srun integrates directly with SLURM's resource allocation
# CRITICAL: Don't use --gpu-bind, let the code handle GPU selection via local_device_idx
# MPI will initialize and coordinate across all processes
srun --ntasks=16 \
     --ntasks-per-node=4 \
     --gpus-per-node=4 \
     --cpus-per-task=32 \
     --cpu-bind=cores \
     bash -c '
       # Ensure GTL library is in LD_LIBRARY_PATH at runtime
       if [ -n "'$GTL_LIBRARY_PATH'" ]; then
         export LD_LIBRARY_PATH="'$GTL_LIBRARY_PATH':$LD_LIBRARY_PATH"
       fi
       
       # Enable GPU-aware MPI
       export MPICH_GPU_SUPPORT_ENABLED=1
       
       # Debug: Check GPU visibility
       NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
       echo "Rank $SLURM_PROCID on $(hostname) - MPI will coordinate, $NUM_GPUS GPUs visible"
       
       # Run training - Use MPI init method (default)
       # MPI_Init will be called inside train_gpt2cu to get rank and size
       # All 4 GPUs visible to each process, code selects based on MPI local rank
       ./train_gpt2cu \
         -i "dev/data/fineweb10B/fineweb_train_*.bin" \
         -j "dev/data/fineweb10B/fineweb_val_*.bin" \
         -o MultiGPU_16x_log774M_1024l_FlashAttention \
         -e "d36" \
         -b 2 \
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
         -h 1 \
         -pi mpi
     '

TRAIN_EXIT_CODE=$?

echo "=============================================="
echo "Training completed at $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "🎉 SUCCESS: Multi-Node Multi-GPU Training with cuDNN Flash Attention completed!"
    echo "Output directory:"
    ls -la MultiGPU_16x_log774M_1024l_FlashAttention/ 2>/dev/null || echo "Check output directory"
    echo ""
    echo "✅ Training used 16 GPUs (4 nodes × 4 GPUs) with NCCL and cuDNN Flash Attention"
else
    echo "❌ Multi-Node Multi-GPU Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo "=============================================="
echo "✨ Multi-Node Multi-GPU cuDNN Flash Attention Training Complete! ✨"
echo "=============================================="
