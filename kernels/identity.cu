#include "../common/utils.h"
// #include <__clang_cuda_builtin_vars.h>

#define N 1048576  // 1M elements

// CPU implementation
void memoryCopyCPU(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = input[i];
    }
}

// TODO: CUDA kernel implementation (basic memory copy)
// This is a simple kernel to understand memory access patterns
__global__ void memoryCopyKernel(const float* input, float* output, int n) {
    // TODO: Implement basic memory copy kernel
    // Each thread copies one element
    // This helps understand global memory access patterns
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        output[idx] = input[idx];
    }
}

// TODO: CUDA kernel with coalesced access (optimization)
// Demonstrates importance of memory coalescing
__global__ void memoryCopyCoalescedKernel(const float* input, float* output, int n) {
    // TODO: Implement with coalesced memory access
    // Use vectorized loads/stores (float4) for better bandwidth
    
}

int main() {
    printHeader("Memory Copy / Identity");

    // Allocate host memory
    float* h_input = (float*)malloc(N * sizeof(float));
    float* h_output = (float*)malloc(N * sizeof(float));
    float* h_output_ref = (float*)malloc(N * sizeof(float));

    // Initialize input data
    initRandom(h_input, N, 0.0f, 10.0f);

    // CPU execution
    CpuTimer cpu_timer;
    cpu_timer.start();
    memoryCopyCPU(h_input, h_output_ref, N);
    cpu_timer.stop();
    float cpu_time = cpu_timer.elapsed();

    // Allocate device memory
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * sizeof(float)));

    // Copy data to device
    cudaCheck(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // GPU execution
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    GpuTimer gpu_timer;
    gpu_timer.start();
    memoryCopyKernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    gpu_timer.stop();
    float gpu_time = gpu_timer.elapsed();

    // Copy result back to host
    cudaCheck(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify result
    bool passed = verifyResult(h_output, h_output_ref, N);

    // Calculate bandwidth
    float bandwidth_cpu = (2.0f * N * sizeof(float)) / (cpu_time * 1e6);  // GB/s
    float bandwidth_gpu = (2.0f * N * sizeof(float)) / (gpu_time * 1e6);  // GB/s

    printf("Size:         %d elements\n", N);
    printf("CPU time:     %.3f ms (%.2f GB/s)\n", cpu_time, bandwidth_cpu);
    printf("GPU time:     %.3f ms (%.2f GB/s)\n", gpu_time, bandwidth_gpu);
    if (gpu_time > 0) {
        printf("Speedup:      %.2fx\n", cpu_time / gpu_time);
    }
    printf("Verification: %s\n", passed ? "PASSED ✓" : "FAILED ✗");
    printf("================================================================================\n");

    // Cleanup
    free(h_input); free(h_output); free(h_output_ref);
    cudaFree(d_input); cudaFree(d_output);

    return passed ? 0 : 1;
}
