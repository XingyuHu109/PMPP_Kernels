#include "../common/utils.h"
#include <algorithm>

#define N 1048576  // 1M elements
#define BLOCK_SIZE 256

// CPU implementation (using std::sort)
void sortCPU(int* data, int n) {
    std::sort(data, data + n);
}

// TODO: CUDA kernel for bitonic sort (block-level)
// Bitonic sort is a simple parallel sorting algorithm
__global__ void bitonicSortKernel(int* data, int n, int k, int j) {
    // TODO: Implement bitonic sort kernel
    // This kernel performs one step of the bitonic sort
    // Parameters k and j define the current stage of sorting
    // Hint: Each thread compares and swaps two elements based on k and j
}

// TODO: Alternative - Odd-Even merge sort kernel
__global__ void oddEvenMergeSortKernel(int* data, int n) {
    // TODO: Implement odd-even merge sort
    // Another parallel sorting algorithm option
}

int main() {
    printHeader("Parallel Sort (Bitonic Sort)");

    // Round N to nearest power of 2 for bitonic sort
    int n = N;
    int powerOf2 = 1;
    while (powerOf2 < n) powerOf2 <<= 1;
    n = powerOf2;

    // Allocate host memory
    int* h_data = (int*)malloc(n * sizeof(int));
    int* h_data_ref = (int*)malloc(n * sizeof(int));

    // Initialize random data
    initRandomInt(h_data, n, 0, 10000);
    memcpy(h_data_ref, h_data, n * sizeof(int));

    // CPU execution
    printf("Running CPU version... ");
    fflush(stdout);
    CpuTimer cpu_timer;
    cpu_timer.start();
    sortCPU(h_data_ref, n);
    cpu_timer.stop();
    float cpu_time = cpu_timer.elapsed();
    printf("done.\n");

    // Allocate device memory
    int* d_data;
    cudaCheck(cudaMalloc(&d_data, n * sizeof(int)));

    // Copy data to device
    cudaCheck(cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice));

    // GPU execution (bitonic sort)
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    GpuTimer gpu_timer;
    gpu_timer.start();

    // Bitonic sort iterations
    for (int k = 2; k <= n; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            bitonicSortKernel<<<gridSize, blockSize>>>(d_data, n, k, j);
            cudaCheck(cudaDeviceSynchronize());
        }
    }

    gpu_timer.stop();
    float gpu_time = gpu_timer.elapsed();

    // Copy result back to host
    cudaCheck(cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    bool passed = verifyResultInt(h_data, h_data_ref, n);

    printf("Size:         %d elements\n", n);
    printf("CPU time:     %.3f ms\n", cpu_time);
    printf("GPU time:     %.3f ms\n", gpu_time);
    if (gpu_time > 0) {
        printf("Speedup:      %.2fx\n", cpu_time / gpu_time);
    }
    printf("Verification: %s\n", passed ? "PASSED ✓" : "FAILED ✗");
    printf("================================================================================\n");

    // Cleanup
    free(h_data); free(h_data_ref);
    cudaFree(d_data);

    return passed ? 0 : 1;
}
