CC       := gcc
NVCC     := nvcc
CUDA_ARCH := -arch=sm_89

CFLAGS   := -O3 -Wall -fopenmp -Iinclude
NVFLAGS  := -O3 $(CUDA_ARCH) -Xcompiler -fopenmp -Iinclude

SRC_DIR       := src
TEST_DIR      := tests
BENCH_CPU_DIR := benchmarks/cpu_benchmarks
BENCH_GPU_DIR := benchmarks/gpu_benchmarks
BIN_DIR       := bin

ifeq ($(firstword $(MAKECMDGOALS)),$(filter $(firstword $(MAKECMDGOALS)),bench_dataset benchdataset))
  BENCH_CPU_ARG := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(BENCH_CPU_ARG):;@:)
endif
CPU_CSV ?= $(if $(BENCH_CPU_ARG),$(BENCH_CPU_ARG),benchmark_results.csv)

ifeq ($(firstword $(MAKECMDGOALS)),$(filter $(firstword $(MAKECMDGOALS)),bench_gpu benchgpu))
  BENCH_GPU_ARG := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(BENCH_GPU_ARG):;@:)
endif
GPU_CSV ?= $(if $(BENCH_GPU_ARG),$(BENCH_GPU_ARG),batch_scaling.csv)

ifeq ($(firstword $(MAKECMDGOALS)),predict)
  PREDICT_ARG := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(PREDICT_ARG):;@:)
endif

all: $(BIN_DIR)/main

run: $(BIN_DIR)/main
	./$(BIN_DIR)/main
	python3 scripts/plot_training.py benchmarks/training_log.csv

test_dataset: $(BIN_DIR)/test_dataset
	./$(BIN_DIR)/test_dataset

$(BIN_DIR)/test_dataset: $(TEST_DIR)/test_dataset.c $(SRC_DIR)/dataset.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@

test_forward: $(BIN_DIR)/test_forward
	./$(BIN_DIR)/test_forward

$(BIN_DIR)/test_forward: $(TEST_DIR)/test_forward_kernels.cu $(SRC_DIR)/layers.cu
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVFLAGS) $^ -o $@

test_backward: $(BIN_DIR)/test_backward
	./$(BIN_DIR)/test_backward

$(BIN_DIR)/test_backward: $(TEST_DIR)/test_backward_kernels.cu $(SRC_DIR)/layers.cu
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVFLAGS) $^ -o $@

test_kernels: test_forward test_backward

bench_dataset: $(BIN_DIR)/bench_dataset
	./$(BIN_DIR)/bench_dataset $(CPU_CSV)
	python3 scripts/plot_benchmark.py $(CPU_CSV)

benchdataset: bench_dataset

$(BIN_DIR)/bench_dataset: $(BENCH_CPU_DIR)/bench_dataset.c $(SRC_DIR)/dataset.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@

bench_gpu: $(BIN_DIR)/bench_batch_size
	./$(BIN_DIR)/bench_batch_size $(GPU_CSV)
	python3 scripts/plot_gpu_benchmark.py $(GPU_CSV)

benchgpu: bench_gpu

$(BIN_DIR)/bench_batch_size: $(BENCH_GPU_DIR)/bench_batch_size.cu $(SRC_DIR)/layers.cu
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVFLAGS) $^ -o $@

$(BIN_DIR)/main: main.cu $(SRC_DIR)/dataset.c $(SRC_DIR)/layers.cu
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVFLAGS) $^ -o $@

predict: $(BIN_DIR)/predict
	./$(BIN_DIR)/predict $(PREDICT_ARG)

$(BIN_DIR)/predict: predict.cu $(SRC_DIR)/layers.cu
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVFLAGS) $^ -o $@

clean:
	rm -rf $(BIN_DIR)/*

.PHONY: all run test_dataset test_forward test_backward test_kernels bench_dataset benchdataset bench_gpu benchgpu predict clean
