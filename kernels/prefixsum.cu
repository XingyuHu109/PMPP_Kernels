#include "../common/utils.h"

#define N 1048576  // 1M elements
#define BLOCK_SIZE 256

// CPU implementation (inclusive scan)
void prefixSumCPU(const float* input, float* output, int n) {
    output[0] = input[0];
    for (int i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i];
    }
}

// TODO: CUDA kernel for block-level scan (Kogge-Stone or Brent-Kung)
__global__ void prefixSumKernel(const float* input, float* output, float* blockSums, int n) {
    // TODO: Implement parallel prefix sum (scan) within blocks
    // Hint: Use shared memory for efficient communication
    // Store the final sum of each block in blockSums for hierarchical scan
    // Implement either Kogge-Stone (work-efficient) or Brent-Kung algorithm
}

// TODO: Helper kernel to add block offsets
__global__ void addBlockSumsKernel(float* output, const float* blockSums, int n) {
    // TODO: Add the scanned block sums to each element
    // This completes the hierarchical scan
}

int main() {
    printHeader("Prefix Sum (Parallel Scan)");

    // Allocate host memory
    float* h_input = (float*)malloc(N * sizeof(float));
    float* h_output = (float*)malloc(N * sizeof(float));
    float* h_output_ref = (float*)malloc(N * sizeof(float));

    // Initialize input data (small values to avoid overflow)
    initRandom(h_input, N, 0.0f, 1.0f);

    // CPU execution
    printf("Running CPU version... ");
    fflush(stdout);
    CpuTimer cpu_timer;
    cpu_timer.start();
    prefixSumCPU(h_input, h_output_ref, N);
    cpu_timer.stop();
    float cpu_time = cpu_timer.elapsed();
    printf("done.\n");

    // Allocate device memory
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float *d_input, *d_output, *d_blockSums, *d_scannedBlockSums;

    cudaCheck(cudaMalloc(&d_input, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_blockSums, gridSize * sizeof(float)));
    cudaCheck(cudaMalloc(&d_scannedBlockSums, gridSize * sizeof(float)));

    // Copy data to device
    cudaCheck(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // GPU execution (hierarchical scan)
    GpuTimer gpu_timer;
    gpu_timer.start();

    // Phase 1: Scan within blocks
    prefixSumKernel<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, d_blockSums, N);

    // Phase 2: Scan the block sums (recursive call or single block)
    if (gridSize > 1) {
        prefixSumKernel<<<1, BLOCK_SIZE>>>(d_blockSums, d_scannedBlockSums, nullptr, gridSize);

        // Phase 3: Add scanned block sums to each block
        addBlockSumsKernel<<<gridSize, BLOCK_SIZE>>>(d_output, d_scannedBlockSums, N);
    }

    gpu_timer.stop();
    float gpu_time = gpu_timer.elapsed();

    // Copy result back to host
    cudaCheck(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify result (with larger tolerance due to accumulation)
    bool passed = verifyResult(h_output, h_output_ref, N, 0.1f);

    printf("Size:         %d elements\n", N);
    printf("CPU time:     %.3f ms\n", cpu_time);
    printf("GPU time:     %.3f ms\n", gpu_time);
    if (gpu_time > 0) {
        printf("Speedup:      %.2fx\n", cpu_time / gpu_time);
    }
    printf("Verification: %s\n", passed ? "PASSED ✓" : "FAILED ✗");
    printf("================================================================================\n");

    // Cleanup
    free(h_input); free(h_output); free(h_output_ref);
    cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_blockSums); cudaFree(d_scannedBlockSums);

    return passed ? 0 : 1;
}
