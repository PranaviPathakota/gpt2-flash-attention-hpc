#!/bin/bash
#SBATCH -A m4012                 # replace with your NERSC account
#SBATCH -C "gpu&hbm40g"         # Explicitly request 80GB GPU nodes (use quotes!)
#SBATCH -q debug              # queue: regular or debug
#SBATCH -t 00:10:00              # time limit (hh:mm:ss)
#SBATCH -J llm.c_MultiGPU_Train          # job name
#SBATCH -N 4                    # number of nodes
#SBATCH --gpus-per-node=4       # 4 GPUs for multi-GPU training
#SBATCH --ntasks-per-node=4     # MPI tasks (one per GPU)
#SBATCH --cpus-per-task=32      # OpenMP threads per MPI rank
#SBATCH -o llm.c_MultiGPU_16x_1k_Train.o%j     # stdout file
#SBATCH -e llm.c_MultiGPU_16x_1k_Train.e%j     # stderr file

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

# Print system information
echo "=============================================="
echo "🚀 Multi-GPU Training Script (16 GPUs across 4 nodes)"
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

# Set basic library paths
export LD_LIBRARY_PATH="$CUDA_MATH_LIBS/lib:$CUDA_HOME/lib64:$MPICH_DIR/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_MATH_LIBS/lib:$CUDA_HOME/lib64:$MPICH_DIR/lib:$LIBRARY_PATH"
export CPATH="$CUDA_HOME/include:$CUDA_MATH_LIBS/include:$MPICH_DIR/include:$CPATH"
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

# Create build directory
mkdir -p build

# Check if training data exists
echo "Checking for training data..."
if ls dev/data/fineweb10B/fineweb_train_*.bin 1> /dev/null 2>&1; then
    echo "✅ Training data found"
else
    echo "❌ Training data not found in dev/data/fineweb10B/"
    exit 1
fi

if ls dev/data/fineweb10B/fineweb_val_*.bin 1> /dev/null 2>&1; then
    echo "✅ Validation data found"
else
    echo "❌ Validation data not found in dev/data/fineweb10B/"
    exit 1
fi

# Replace Makefile with multi-GPU enabled version (NO cuDNN)
echo "=============================================="
echo "Creating Makefile with Multi-GPU support (no cuDNN)..."
if [ -f "Makefile" ]; then
    cp Makefile Makefile.backup
    echo "✅ Original Makefile backed up"
fi

# Create the Makefile with NCCL/MPI detection
cat > Makefile << 'EOF'
CC ?= clang
CFLAGS = -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes
LDFLAGS =
LDLIBS = -lm
INCLUDES =
CFLAGS_COND = -march=native

# Find nvcc
SHELL_UNAME = $(shell uname)
REMOVE_FILES = rm -f
OUTPUT_FILE = -o $@
CUDA_OUTPUT_FILE = -o $@

# Default O3 CPU optimization level for NVCC
FORCE_NVCC_O ?= 3

# NVCC flags
NVCC_FLAGS = --threads=0 -t=0 --use_fast_math -std=c++17 -O$(FORCE_NVCC_O)
NVCC_LDFLAGS = -lcublas -lcublasLt -lnvidia-ml
NVCC_INCLUDES =
NVCC_LDLIBS =

# Build directory
BUILD_DIR = build
$(shell mkdir -p $(BUILD_DIR))
REMOVE_BUILD_OBJECT_FILES := rm -f $(BUILD_DIR)/*.o

# NVCC location
NVCC := $(shell which nvcc 2>/dev/null)

# GPU compute capability detection
ifndef GPU_COMPUTE_CAPABILITY
  ifneq ($(shell which nvidia-smi 2>/dev/null),)
    GPU_COMPUTE_CAPABILITY=$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/\.//g' | sort -n | head -n 1)
    GPU_COMPUTE_CAPABILITY := $(strip $(GPU_COMPUTE_CAPABILITY))
  endif
endif

ifneq ($(GPU_COMPUTE_CAPABILITY),)
  NVCC_FLAGS += --generate-code arch=compute_$(GPU_COMPUTE_CAPABILITY),code=[compute_$(GPU_COMPUTE_CAPABILITY),sm_$(GPU_COMPUTE_CAPABILITY)]
  $(info ✓ GPU Compute Capability: $(GPU_COMPUTE_CAPABILITY))
endif

$(info ---------------------------------------------)

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
      NVCC_LDLIBS += -lnccl
    endif
  else
    # Fallback: Check using multiple methods for different Linux distributions
    NCCL_FOUND := $(shell (command -v rpm >/dev/null 2>&1 && rpm -qa 2>/dev/null | grep -q nccl) || (command -v dpkg >/dev/null 2>&1 && dpkg -l 2>/dev/null | grep -q nccl) || (find /usr -name "*libnccl*" -type f 2>/dev/null | head -1 | grep -q .) || (find $(CUDA_HOME)/lib64 -name "*libnccl*" 2>/dev/null | head -1 | grep -q .) && echo "exists")
    ifeq ($(NCCL_FOUND), exists)
      $(info ✓ NCCL found in system paths, multi-GPU enabled)
      NVCC_FLAGS += -DMULTI_GPU
      NVCC_LDLIBS += -lnccl
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
      NVCC_LDLIBS += -lmpi
      # Add GTL library if available (required for GPU-aware MPI)
      ifdef GTL_LIBRARY_PATH
        $(info ✓ GTL library found at $(GTL_LIBRARY_PATH))
        NVCC_LDFLAGS += -L$(GTL_LIBRARY_PATH)
        NVCC_LDLIBS += -lmpi_gtl_cuda
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
    NVCC_LDLIBS += -lmpi
    NVCC_FLAGS += -DUSE_MPI
  else
    $(warning ✗ MPI not found, multi-GPU will be disabled!)
    $(warning   Set MPICH_DIR environment variable or install MPI)
  endif
endif

# OpenMP detection
ifeq ($(NO_OMP), 1)
  $(info → OpenMP manually disabled)
else
  ifeq ($(shell echo | $(CC) -fopenmp -x c -E - > /dev/null 2>&1; echo $$?), 0)
    CFLAGS += -fopenmp -DOMP
    LDLIBS += -lgomp
    $(info ✓ OpenMP found)
  else
    $(info ✗ OpenMP not found)
  endif
endif

# Precision settings
PRECISION ?= BF16
VALID_PRECISIONS := FP32 FP16 BF16
ifeq ($(filter $(PRECISION),$(VALID_PRECISIONS)),)
  $(error Invalid precision $(PRECISION), valid precisions are $(VALID_PRECISIONS))
endif
ifeq ($(PRECISION), FP32)
  PFLAGS = -DENABLE_FP32
else ifeq ($(PRECISION), FP16)
  PFLAGS = -DENABLE_FP16
else
  PFLAGS = -DENABLE_BF16
endif

$(info cuDNN Flash Attention: DISABLED)
$(info ---------------------------------------------)

# Targets
.PHONY: all train_gpt2 test_gpt2 train_gpt2cu test_gpt2cu clean

TARGETS = train_gpt2 test_gpt2

ifeq ($(NVCC),)
    $(info ✗ nvcc not found, skipping GPU builds)
else
    $(info ✓ nvcc found, including GPU support)
    TARGETS += train_gpt2cu test_gpt2cu
endif

all: $(TARGETS)

train_gpt2: train_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) $(OUTPUT_FILE)

test_gpt2: test_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) $(OUTPUT_FILE)

train_gpt2cu: train_gpt2.cu
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

test_gpt2cu: test_gpt2.cu
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

clean:
	$(REMOVE_FILES) $(TARGETS)
	$(REMOVE_BUILD_OBJECT_FILES)
EOF

echo "✅ Makefile created with Multi-GPU support"

# Verify NCCL is available before building
echo "=============================================="
echo "Pre-build NCCL verification:"
if [ -n "$NCCL_HOME" ]; then
    echo "✅ NCCL_HOME is set: $NCCL_HOME"
    export NCCL_HOME  # Make sure it's exported for make
elif [ -n "$NCCL_DIR" ]; then
    echo "✅ NCCL_DIR is set: $NCCL_DIR"
    export NCCL_HOME=$NCCL_DIR
else
    echo "❌ WARNING: NCCL_HOME/NCCL_DIR not set!"
    echo "   Multi-GPU build will likely fail"
fi

# Clean any previous builds
echo "=============================================="
echo "Cleaning previous builds..."
make clean

# Build with multi-GPU support (no cuDNN)
echo "=============================================="
echo "Building train_gpt2cu with Multi-GPU support..."
echo "(Standard attention, no Flash Attention)"

make train_gpt2cu

BUILD_EXIT_CODE=$?

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo "✅ Build successful!"
else
    echo "❌ Build failed! Debug information:"
    echo "Available source files:"
    ls -la *.cu *.c 2>/dev/null || echo "No source files found"
    echo "Build directory:"
    ls -la build/ 2>/dev/null || echo "Build directory empty"
    exit 1
fi

# Verify executable
if [ -x "./train_gpt2cu" ]; then
    echo "✅ Executable created successfully"
    ls -la ./train_gpt2cu
    
    # Check NCCL linking
    echo ""
    echo "Checking NCCL integration..."
    if ldd ./train_gpt2cu | grep -q nccl; then
        echo "✅ NCCL is linked - Multi-GPU enabled!"
        ldd ./train_gpt2cu | grep nccl
    else
        echo "⚠️  NCCL not linked - training will run on single GPU only"
    fi
else
    echo "❌ Executable not found"
    exit 1
fi

# -------------------------------
# Run the training with Multi-GPU (16 GPUs across 4 nodes)
# -------------------------------
echo "=============================================="
echo "🚀 Starting Multi-Node Multi-GPU Training"
echo "Configuration: 4 nodes × 4 GPUs = 16 GPUs total"
echo "Attention: Standard (no Flash Attention)"
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
         -o MultiGPU_16x_log774M_1024l \
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
    echo "🎉 SUCCESS: Multi-Node Multi-GPU Training completed!"
    echo "Output directory:"
    ls -la MultiGPU_16x_log774M_1024l/ 2>/dev/null || echo "Check output directory"
    echo ""
    echo "✅ Training used 16 GPUs (4 nodes × 4 GPUs) with NCCL"
    echo "✅ Standard attention (no Flash Attention) was used"
else
    echo "❌ Multi-GPU Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo "=============================================="
echo "✨ Multi-Node Multi-GPU Training Complete! ✨"
echo "=============================================="