# CUDA Kernel Learning Makefile

# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O0 -arch=sm_75 -std=c++11

# Directories
KERNEL_DIR = kernels
COMMON_DIR = common
BIN_DIR = bin

# Common utilities
COMMON_SRC = $(COMMON_DIR)/utils.cu
COMMON_HDR = $(COMMON_DIR)/utils.h

# Kernel source files
KERNELS = vectoradd vectorsum matmul conv2d grayscale histogram prefixsum sort identity

# Targets
TARGETS = $(addprefix $(BIN_DIR)/, $(KERNELS))

# Default target: build all
all: $(BIN_DIR) $(TARGETS)

# Create bin directory
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Pattern rule for building kernels
$(BIN_DIR)/%: $(KERNEL_DIR)/%.cu $(COMMON_SRC) $(COMMON_HDR) | $(BIN_DIR)
	@echo "Building $*..."
	$(NVCC) $(NVCC_FLAGS) $< $(COMMON_SRC) -o $@

# Individual kernel targets
vectoradd: $(BIN_DIR)/vectoradd
vectorsum: $(BIN_DIR)/vectorsum
matmul: $(BIN_DIR)/matmul
conv2d: $(BIN_DIR)/conv2d
grayscale: $(BIN_DIR)/grayscale
histogram: $(BIN_DIR)/histogram
prefixsum: $(BIN_DIR)/prefixsum
sort: $(BIN_DIR)/sort
identity: $(BIN_DIR)/identity

# Run all kernels
run-all: all
	@echo ""
	@echo "Running all kernels..."
	@echo ""
	@for kernel in $(KERNELS); do \
		echo "Running $$kernel..."; \
		./$(BIN_DIR)/$$kernel; \
		echo ""; \
	done

# Run individual kernel
run-%: $(BIN_DIR)/%
	./$(BIN_DIR)/$*

# Clean
clean:
	rm -rf $(BIN_DIR)

# Help
help:
	@echo "CUDA Kernel Learning - Makefile Help"
	@echo ""
	@echo "Available targets:"
	@echo "  make all          - Build all kernels"
	@echo "  make <kernel>     - Build specific kernel (e.g., make vectoradd)"
	@echo "  make run-all      - Build and run all kernels"
	@echo "  make run-<kernel> - Build and run specific kernel (e.g., make run-vectoradd)"
	@echo "  make clean        - Remove all binaries"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Available kernels:"
	@echo "  vectoradd    - Vector addition"
	@echo "  vectorsum    - Vector reduction/sum"
	@echo "  matmul       - Matrix multiplication"
	@echo "  conv2d       - 2D convolution"
	@echo "  grayscale    - RGB to grayscale conversion"
	@echo "  histogram    - Histogram computation"
	@echo "  prefixsum    - Parallel prefix sum (scan)"
	@echo "  sort         - Parallel sorting"
	@echo "  identity     - Memory copy/identity"

.PHONY: all clean help run-all $(KERNELS)
