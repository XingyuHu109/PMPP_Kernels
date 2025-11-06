# CUDA Kernel Learning Scaffold

Clean scaffold for learning GPU programming with CUDA, based on the PMPP book.

## Structure

```
├── Makefile         # Build system
├── common/          # Utilities (timing, error checking, verification)
└── kernels/         # 9 kernel implementations with CPU baseline + CUDA stubs
    ├── vectoradd.cu    # Vector addition
    ├── vectorsum.cu    # Reduction
    ├── matmul.cu       # Matrix multiplication
    ├── conv2d.cu       # 2D convolution
    ├── grayscale.cu    # RGB to grayscale
    ├── histogram.cu    # Histogram
    ├── prefixsum.cu    # Parallel scan
    ├── sort.cu         # Bitonic sort
    └── identity.cu     # Memory bandwidth test
```

## Quick Start

```bash
# Build and run
make vectoradd
./bin/vectoradd

# Or use shortcuts
make run-vectoradd
make run-all
make clean
```

## How It Works

Each kernel file has:
1. CPU reference implementation
2. CUDA kernel stub (marked `// TODO:`) - **implement here**
3. Test harness with timing and verification

Run unimplemented kernels to see them fail, implement the `// TODO:` sections, then watch them pass with GPU speedups!

## Learning Order

1. **vectoradd** - Thread indexing basics
2. **identity** - Memory coalescing
3. **matmul** - Shared memory and tiling
4. **vectorsum** - Parallel reduction
5. **histogram** - Atomic operations
6. Continue with conv2d, grayscale, prefixsum, sort

## Kernel Overview

| Kernel | Concept | Key Learning | Size |
|--------|---------|--------------|------|
| vectoradd | Parallel computation | Thread indexing | 1M elements |
| vectorsum | Reduction | Shared memory, sync | 1M elements |
| matmul | 2D computation | Tiling, shared memory | 1024×1024 |
| conv2d | Stencil | Boundary handling, constant mem | 1920×1080, 5×5 |
| grayscale | Image processing | 2D indexing | 1920×1080 RGB |
| histogram | Scatter ops | Atomics, privatization | 1M, 256 bins |
| prefixsum | Scan | Hierarchical algorithms | 1M elements |
| sort | Parallel sort | Bitonic sort | 1M elements |
| identity | Memory copy | Coalesced access | 1M elements |

## Requirements

- CUDA Toolkit
- NVIDIA GPU (compute capability 7.0+)

## License

MIT License - See LICENSE file
