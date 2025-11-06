#include "../common/utils.h"

#define N 1048576  // 1M elements
#define BLOCK_SIZE 256

// CPU implementation
float vectorSumCPU(const float* a, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
    return sum;
}

// TODO: CUDA kernel implementation for reduction
// This kernel performs parallel reduction within blocks
__global__ void vectorSumKernel(const float* input, float* output, int n) {
    // TODO: Implement parallel reduction kernel
    // Hint: Use shared memory for block-level reduction
    // Each block produces one partial sum
}

int main() {
    printHeader("Vector Sum (Reduction)");

    // Allocate host memory
    float* h_input = (float*)malloc(N * sizeof(float));

    // Initialize input data
    initRandom(h_input, N, 0.0f, 1.0f);

    // CPU execution
    CpuTimer cpu_timer;
    cpu_timer.start();
    float cpu_sum = vectorSumCPU(h_input, N);
    cpu_timer.stop();
    float cpu_time = cpu_timer.elapsed();

    // Allocate device memory
    float* d_input;
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float* d_partial_sums;
    cudaCheck(cudaMalloc(&d_input, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_partial_sums, gridSize * sizeof(float)));

    // Copy data to device
    cudaCheck(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // GPU execution (two-pass reduction)
    GpuTimer gpu_timer;
    gpu_timer.start();

    // First pass: reduce within blocks
    vectorSumKernel<<<gridSize, BLOCK_SIZE>>>(d_input, d_partial_sums, N);

    // Second pass: reduce partial sums (if needed)
    if (gridSize > 1) {
        vectorSumKernel<<<1, BLOCK_SIZE>>>(d_partial_sums, d_partial_sums, gridSize);
    }

    gpu_timer.stop();
    float gpu_time = gpu_timer.elapsed();

    // Copy result back to host
    float gpu_sum;
    cudaCheck(cudaMemcpy(&gpu_sum, d_partial_sums, sizeof(float), cudaMemcpyDeviceToHost));

    // Verify result (with tolerance for floating-point)
    bool passed = fabs(gpu_sum - cpu_sum) < 1.0f;  // Relaxed tolerance for sum

    printf("Size:         %d elements\n", N);
    printf("CPU time:     %.3f ms (Sum: %.6f)\n", cpu_time, cpu_sum);
    printf("GPU time:     %.3f ms (Sum: %.6f)\n", gpu_time, gpu_sum);
    if (gpu_time > 0) {
        printf("Speedup:      %.2fx\n", cpu_time / gpu_time);
    }
    printf("Verification: %s\n", passed ? "PASSED ✓" : "FAILED ✗");
    printf("================================================================================\n");

    // Cleanup
    free(h_input);
    cudaFree(d_input);
    cudaFree(d_partial_sums);

    return passed ? 0 : 1;
}
