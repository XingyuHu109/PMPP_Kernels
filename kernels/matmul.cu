#include "../common/utils.h"

#define WIDTH 1024  // Matrix dimension (WIDTH x WIDTH)
#define TILE_SIZE 16  // Tile size for shared memory optimization

// CPU implementation
void matmulCPU(const float* A, const float* B, float* C, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += A[row * width + k] * B[k * width + col];
            }
            C[row * width + col] = sum;
        }
    }
}

// TODO: CUDA kernel implementation (basic version)
// Each thread computes one element of the result matrix
__global__ void matmulKernel(const float* A, const float* B, float* C, int width) {
    // TODO: Implement matrix multiplication kernel
    // Hint: Calculate row and col from thread indices
    // C[row][col] = sum of A[row][k] * B[k][col] for all k
}

// TODO: CUDA kernel with tiled optimization (advanced)
// Uses shared memory to improve memory access patterns
__global__ void matmulTiledKernel(const float* A, const float* B, float* C, int width) {
    // TODO: Implement tiled matrix multiplication with shared memory
    // Hint: Load tiles of A and B into shared memory
    // This reduces global memory accesses significantly
}

int main() {
    printHeader("Matrix Multiplication");

    int size = WIDTH * WIDTH;

    // Allocate host memory
    float *h_A, *h_B, *h_C, *h_C_ref;
    h_A = (float*)malloc(size * sizeof(float));
    h_B = (float*)malloc(size * sizeof(float));
    h_C = (float*)malloc(size * sizeof(float));
    h_C_ref = (float*)malloc(size * sizeof(float));

    // Initialize input data
    initRandom(h_A, size, 0.0f, 1.0f);
    initRandom(h_B, size, 0.0f, 1.0f);

    // CPU execution
    printf("Running CPU version... ");
    fflush(stdout);
    CpuTimer cpu_timer;
    cpu_timer.start();
    matmulCPU(h_A, h_B, h_C_ref, WIDTH);
    cpu_timer.stop();
    float cpu_time = cpu_timer.elapsed();
    printf("done.\n");

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaCheck(cudaMalloc(&d_A, size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_B, size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_C, size * sizeof(float)));

    // Copy data to device
    cudaCheck(cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice));

    // GPU execution (basic version)
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((WIDTH + TILE_SIZE - 1) / TILE_SIZE, (WIDTH + TILE_SIZE - 1) / TILE_SIZE);

    GpuTimer gpu_timer;
    gpu_timer.start();
    matmulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, WIDTH);
    gpu_timer.stop();
    float gpu_time = gpu_timer.elapsed();

    // Copy result back to host
    cudaCheck(cudaMemcpy(h_C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify result
    bool passed = verifyResult(h_C, h_C_ref, size, 0.01f);

    printf("Matrix size:  %d x %d\n", WIDTH, WIDTH);
    printf("CPU time:     %.3f ms\n", cpu_time);
    printf("GPU time:     %.3f ms\n", gpu_time);
    if (gpu_time > 0) {
        printf("Speedup:      %.2fx\n", cpu_time / gpu_time);
    }
    printf("Verification: %s\n", passed ? "PASSED ✓" : "FAILED ✗");
    printf("================================================================================\n");

    // Cleanup
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return passed ? 0 : 1;
}
