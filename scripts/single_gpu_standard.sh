#!/bin/bash
#SBATCH -A m4012                 # replace with your NERSC account
#SBATCH -C gpu                  # GPU nodes (40GB A100)
#SBATCH -q regular              # queue: regular or debug
#SBATCH -t 01:00:00              # time limit (hh:mm:ss)
#SBATCH -J llm.cGpt2-2_Train          # job name
#SBATCH -N 1                    # number of nodes
#SBATCH --gpus-per-node=1       # 1 GPU for single-GPU training
#SBATCH --ntasks-per-node=1     # 1 task (no MPI)
#SBATCH --cpus-per-task=32      # OpenMP threads
#SBATCH -o llm.c_SingleGPU.o%j               # stdout file
#SBATCH -e llm.c_SingleGPU.e%j               # stderr file

# Load required modules (NO MPI for single GPU)
module load gcc/12.2.0
module load python/3.9
module load cudatoolkit/12.4
module load PrgEnv-gnu

# Print system information
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME" 
echo "Start time: $(date)"
echo "Python version: $(python --version)"
echo "=============================================="

# Display GPU information
nvidia-smi

# Activate conda environment FIRST
echo "Activating conda environment..."
conda activate env-llm.c
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment env-llm.c"
    exit 1
fi
echo "✓ Conda environment activated: $CONDA_PREFIX"

# Set environment variables for optimal performance
export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Set CUDA paths
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4
export CUDA_MATH_LIBS=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/12.4/targets/x86_64-linux

# Set basic library paths
export LD_LIBRARY_PATH="$CUDA_MATH_LIBS/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_MATH_LIBS/lib:$CUDA_HOME/lib64:$LIBRARY_PATH"
export CPATH="$CUDA_HOME/include:$CUDA_MATH_LIBS/include:$CPATH"
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
echo "=============================================="

# Create build directory
mkdir -p build

# Check if training data exists
echo "Checking for training data..."
if ls dev/data/fineweb10B/fineweb_train_*.bin 1> /dev/null 2>&1; then
    echo "✓ Training data found"
else
    echo "✗ Training data not found in dev/data/fineweb10B/"
    exit 1
fi

if ls dev/data/fineweb10B/fineweb_val_*.bin 1> /dev/null 2>&1; then
    echo "✓ Validation data found"
else
    echo "✗ Validation data not found in dev/data/fineweb10B/"
    exit 1
fi

# Replace the problematic Makefile with our fixed version
echo "=============================================="
echo "Replacing Makefile with fixed version..."
if [ -f "Makefile" ]; then
    cp Makefile Makefile.backup
    echo "✓ Original Makefile backed up"
fi

# Create a simple single-GPU Makefile (NO MPI, NO NCCL)
cat > Makefile << 'EOF'
CC ?= clang
CFLAGS = -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes
LDFLAGS =
LDLIBS = -lm
INCLUDES =
CFLAGS_COND = -march=native

# Disable multi-GPU for single GPU training
NO_MULTI_GPU = 1
NO_USE_MPI = 1

# Find nvcc
SHELL_UNAME = $(shell uname)
REMOVE_FILES = rm -f
OUTPUT_FILE = -o $@
CUDA_OUTPUT_FILE = -o $@

# Default O3 CPU optimization level for NVCC (0 for fastest compile time)
FORCE_NVCC_O ?= 3

# NVCC flags
NVCC_FLAGS = --threads=0 -t=0 --use_fast_math -std=c++17 -O$(FORCE_NVCC_O)
NVCC_LDFLAGS = -lcublas -lcublasLt
NVCC_INCLUDES =
NVCC_LDLIBS =
NVCC_CUDNN =
USE_CUDNN ?= 0

# Build directory
BUILD_DIR = build
$(shell mkdir -p $(BUILD_DIR))
REMOVE_BUILD_OBJECT_FILES := rm -f $(BUILD_DIR)/*.o

# Function to check if a file exists in the PATH
define file_exists_in_path
  $(which $(1) 2>/dev/null)
endef

ifneq ($(CI),true)
  ifndef GPU_COMPUTE_CAPABILITY
    ifneq ($(call file_exists_in_path, nvidia-smi),)
      GPU_COMPUTE_CAPABILITY=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/\.//g' | sort -n | head -n 1)
      GPU_COMPUTE_CAPABILITY := $(strip $(GPU_COMPUTE_CAPABILITY))
    endif
  endif
endif

ifneq ($(GPU_COMPUTE_CAPABILITY),)
  NVCC_FLAGS += --generate-code arch=compute_$(GPU_COMPUTE_CAPABILITY),code=[compute_$(GPU_COMPUTE_CAPABILITY),sm_$(GPU_COMPUTE_CAPABILITY)]
endif

# Platform detection
$(info ---------------------------------------------)

NVCC := $(shell which nvcc 2>/dev/null)
NVCC_LDFLAGS += -lnvidia-ml

# Function to test if the compiler accepts a given flag
define check_and_add_flag
  $(eval FLAG_SUPPORTED := $(shell printf "int main() { return 0; }\n" | $(CC) $(1) -x c - -o /dev/null 2>/dev/null && echo 'yes'))
  ifeq ($(FLAG_SUPPORTED),yes)
      CFLAGS += $(1)
  endif
endef

# Check each flag and add it if supported
$(foreach flag,$(CFLAGS_COND),$(eval $(call check_and_add_flag,$(flag))))

# cuDNN support (simplified - disabled by default to avoid linking issues)
ifeq ($(USE_CUDNN), 1)
  $(info → Attempting to enable cuDNN support)
  ifneq ($(CUDNN_FRONTEND_PATH),)
    NVCC_INCLUDES += -I$(CUDNN_FRONTEND_PATH)
  endif
  ifneq ($(CUDNN_INCLUDE),)
    NVCC_INCLUDES += -I$(CUDNN_INCLUDE)
  endif
  ifneq ($(CUDNN_LIB),)
    NVCC_LDFLAGS += -L$(CUDNN_LIB)
  endif
  NVCC_LDFLAGS += -lcudnn
  NVCC_FLAGS += -DENABLE_CUDNN
  NVCC_CUDNN = $(BUILD_DIR)/cudnn_att.o
else
  $(info → cuDNN disabled (USE_CUDNN=0))
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

# NCCL detection (fixed for SUSE Linux)
ifeq ($(NO_MULTI_GPU), 1)
  $(info → Multi-GPU manually disabled)
else
  # Check using multiple methods for different Linux distributions
  NCCL_FOUND := $(shell (command -v rpm >/dev/null 2>&1 && rpm -qa 2>/dev/null | grep -q nccl) || (command -v dpkg >/dev/null 2>&1 && dpkg -l 2>/dev/null | grep -q nccl) || (find /usr -name "*libnccl*" -type f 2>/dev/null | head -1 | grep -q .) && echo "exists")
  ifeq ($(NCCL_FOUND), exists)
    $(info ✓ NCCL found, multi-GPU enabled)
    NVCC_FLAGS += -DMULTI_GPU
    NVCC_LDLIBS += -lnccl
  else
    $(info ✗ NCCL not found, multi-GPU disabled)
  endif
endif

# MPI detection (simplified)
OPENMPI_DIR ?= /usr/lib/x86_64-linux-gnu/openmpi
ifeq ($(NO_USE_MPI), 1)
  $(info → MPI manually disabled)
else ifeq ($(shell [ -d $(OPENMPI_DIR)/lib ] && [ -d $(OPENMPI_DIR)/include ] && echo "exists"), exists)
  $(info ✓ MPI found)
  NVCC_INCLUDES += -I$(OPENMPI_DIR)/include
  NVCC_LDFLAGS += -L$(OPENMPI_DIR)/lib
  NVCC_LDLIBS += -lmpi
  NVCC_FLAGS += -DUSE_MPI
else
  $(info ✗ MPI not found)
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

# Targets
.PHONY: all train_gpt2 test_gpt2 train_gpt2cu test_gpt2cu clean

TARGETS = train_gpt2 test_gpt2

ifeq ($(NVCC),)
    $(info ✗ nvcc not found, skipping GPU builds)
else
    $(info ✓ nvcc found, including GPU support)
    TARGETS += train_gpt2cu test_gpt2cu $(NVCC_CUDNN)
endif

$(info ---------------------------------------------)

all: $(TARGETS)

train_gpt2: train_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) $(OUTPUT_FILE)

test_gpt2: test_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) $(OUTPUT_FILE)

$(NVCC_CUDNN): llmc/cudnn_att.cpp
	$(NVCC) -c $(NVCC_FLAGS) $(PFLAGS) $^ $(NVCC_INCLUDES) -o $@

train_gpt2cu: train_gpt2.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

test_gpt2cu: test_gpt2.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

clean:
	$(REMOVE_FILES) $(TARGETS)
	$(REMOVE_BUILD_OBJECT_FILES)
EOF

echo "✓ Fixed Makefile created"

# Clean any previous builds
echo "=============================================="
echo "Cleaning previous builds..."
make clean

# Build without cuDNN first (most reliable)
echo "=============================================="
echo "Building train_gpt2cu (without cuDNN)..."

make train_gpt2cu 

BUILD_EXIT_CODE=$?

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo "✓ Build successful!"
else
    echo "✗ Build failed! Debug information:"
    echo "Available source files:"
    ls -la *.cu *.c 2>/dev/null || echo "No source files found"
    echo "Build directory:"
    ls -la build/ 2>/dev/null || echo "Build directory empty"
    exit 1
fi

# Verify executable
if [ -x "./train_gpt2cu" ]; then
    echo "✓ Executable created successfully"
    ls -la ./train_gpt2cu
else
    echo "✗ Executable not found"
    exit 1
fi

# -------------------------------
# Run the training
# -------------------------------
echo "=============================================="
echo "Starting training at $(date)"
echo "Using standard attention (cuDNN disabled)"
echo "=============================================="

./train_gpt2cu \
    -i "dev/data/fineweb10B/fineweb_train_*.bin" \
    -j "dev/data/fineweb10B/fineweb_val_*.bin" \
    -o SingleGPU_1x_log124M_1024l \
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
    echo "✓ Training completed successfully!"
    echo "Output directory:"
    ls -la SingleGPU_1x_log124M_1024l/ 2>/dev/null || echo "SingleGPU_1x_log124M_1024l directory not found"
else
    echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo "=============================================="