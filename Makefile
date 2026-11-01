# Compilers
CC       := gcc
NVCC     := nvcc

# Target GPU Architecture for RTX 4070 Ti (Compute Capability 8.9)
CUDA_ARCH := -arch=sm_89

# Flags
CFLAGS   := -O3 -Wall -fopenmp -Iinclude
NVFLAGS  := -O3 $(CUDA_ARCH) -Xcompiler -fopenmp -Iinclude

# Directories
SRC_DIR   := src
INC_DIR   := include
TEST_DIR  := tests
BENCH_DIR := benchmarks
BIN_DIR   := bin

# Default Target
all: $(BIN_DIR)/main

# Main Engine Binary
$(BIN_DIR)/main: main.cu $(SRC_DIR)/dataset.c $(SRC_DIR)/layers.cu
@mkdir -p $(BIN_DIR)
$(NVCC) $(NVFLAGS) $^ -o $@
@echo "✓ Built $@"

# Clean Build Artifacts
clean:
rm -rf $(BIN_DIR)/*
@echo "✓ Cleaned bin/"

.PHONY: all clean
