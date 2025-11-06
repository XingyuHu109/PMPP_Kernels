#include "../common/utils.h"
#include <random>

#define N 1048576  // 1M elements
#define NUM_BINS 256
#define BLOCK_SIZE 256

// CPU implementation
void histogramCPU(const unsigned char* input, int* histogram, int n) {
    // Initialize histogram
    for (int i = 0; i < NUM_BINS; i++) {
        histogram[i] = 0;
    }

    // Count occurrences
    for (int i = 0; i < n; i++) {
        histogram[input[i]]++;
    }
}

// TODO: CUDA kernel implementation (atomic version)
__global__ void histogramAtomicKernel(const unsigned char* input, int* histogram, int n) {
    // TODO: Implement histogram using atomic operations
    // Hint: Use atomicAdd to increment histogram bins
    // Be aware of potential performance issues with global atomics
}

// TODO: CUDA kernel with privatization (optimization)
// Each block maintains a private histogram, then merges to global
__global__ void histogramPrivatizedKernel(const unsigned char* input, int* histogram, int n) {
    // TODO: Implement histogram with privatization
    // Use shared memory for block-local histogram
    // Reduce contention on global memory atomics
}

int main() {
    printHeader("Histogram Computation");

    // Allocate host memory
    unsigned char* h_input = (unsigned char*)malloc(N * sizeof(unsigned char));
    int* h_histogram = (int*)malloc(NUM_BINS * sizeof(int));
    int* h_histogram_ref = (int*)malloc(NUM_BINS * sizeof(int));

    // Initialize random input data (values 0-255)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);
    for (int i = 0; i < N; i++) {
        h_input[i] = (unsigned char)dis(gen);
    }

    // CPU execution
    CpuTimer cpu_timer;
    cpu_timer.start();
    histogramCPU(h_input, h_histogram_ref, N);
    cpu_timer.stop();
    float cpu_time = cpu_timer.elapsed();

    // Allocate device memory
    unsigned char* d_input;
    int* d_histogram;
    cudaCheck(cudaMalloc(&d_input, N * sizeof(unsigned char)));
    cudaCheck(cudaMalloc(&d_histogram, NUM_BINS * sizeof(int)));

    // Copy data to device and initialize histogram to zero
    cudaCheck(cudaMemcpy(d_input, h_input, N * sizeof(unsigned char), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_histogram, 0, NUM_BINS * sizeof(int)));

    // GPU execution
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    GpuTimer gpu_timer;
    gpu_timer.start();
    histogramAtomicKernel<<<gridSize, BLOCK_SIZE>>>(d_input, d_histogram, N);
    gpu_timer.stop();
    float gpu_time = gpu_timer.elapsed();

    // Copy result back to host
    cudaCheck(cudaMemcpy(h_histogram, d_histogram, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    bool passed = verifyResultInt(h_histogram, h_histogram_ref, NUM_BINS);

    printf("Data size:    %d elements\n", N);
    printf("Bins:         %d\n", NUM_BINS);
    printf("CPU time:     %.3f ms\n", cpu_time);
    printf("GPU time:     %.3f ms\n", gpu_time);
    if (gpu_time > 0) {
        printf("Speedup:      %.2fx\n", cpu_time / gpu_time);
    }
    printf("Verification: %s\n", passed ? "PASSED ✓" : "FAILED ✗");
    printf("================================================================================\n");

    // Cleanup
    free(h_input); free(h_histogram); free(h_histogram_ref);
    cudaFree(d_input); cudaFree(d_histogram);

    return passed ? 0 : 1;
}
