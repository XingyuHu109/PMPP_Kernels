#include "../common/utils.h"

#define N 1048576  // 1M elements

// CPU implementation
void vectorAddCPU(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// TODO: CUDA kernel implementation
// Each thread computes one element of the result
__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    // TODO: Implement vector addition kernel
    // Hint: Calculate global thread index and add corresponding elements
}

int main() {
    printHeader("Vector Addition");

    // Allocate host memory
    float *h_a, *h_b, *h_c, *h_c_ref;
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_c = (float*)malloc(N * sizeof(float));
    h_c_ref = (float*)malloc(N * sizeof(float));

    // Initialize input data
    initRandom(h_a, N, 0.0f, 10.0f);
    initRandom(h_b, N, 0.0f, 10.0f);

    // CPU execution
    CpuTimer cpu_timer;
    cpu_timer.start();
    vectorAddCPU(h_a, h_b, h_c_ref, N);
    cpu_timer.stop();
    float cpu_time = cpu_timer.elapsed();

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaCheck(cudaMalloc(&d_a, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_b, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_c, N * sizeof(float)));

    // Copy data to device
    cudaCheck(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    // GPU execution
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    GpuTimer gpu_timer;
    gpu_timer.start();
    vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    gpu_timer.stop();
    float gpu_time = gpu_timer.elapsed();

    // Copy result back to host
    cudaCheck(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify result
    bool passed = verifyResult(h_c, h_c_ref, N);
    printResults("Vector Addition", N, cpu_time, gpu_time, passed);

    // Cleanup
    free(h_a); free(h_b); free(h_c); free(h_c_ref);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return passed ? 0 : 1;
}
