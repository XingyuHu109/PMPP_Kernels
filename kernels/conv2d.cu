#include "../common/utils.h"

#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1080
#define KERNEL_SIZE 5  // 5x5 convolution kernel

// CPU implementation
void conv2dCPU(const float* input, const float* kernel, float* output,
               int width, int height, int ksize) {
    int radius = ksize / 2;

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;

            for (int krow = 0; krow < ksize; krow++) {
                for (int kcol = 0; kcol < ksize; kcol++) {
                    int irow = row + krow - radius;
                    int icol = col + kcol - radius;

                    // Handle boundary conditions (zero padding)
                    if (irow >= 0 && irow < height && icol >= 0 && icol < width) {
                        sum += input[irow * width + icol] * kernel[krow * ksize + kcol];
                    }
                }
            }

            output[row * width + col] = sum;
        }
    }
}

// TODO: CUDA kernel implementation (basic version)
__global__ void conv2dKernel(const float* input, const float* kernel, float* output,
                              int width, int height, int ksize) {
    // TODO: Implement 2D convolution kernel
    // Hint: Each thread computes one output pixel
    // Handle boundary conditions with zero padding
}

// TODO: CUDA kernel with constant memory (optimization)
// Store the convolution kernel in constant memory for faster access
__constant__ float d_const_kernel[KERNEL_SIZE * KERNEL_SIZE];

__global__ void conv2dConstKernel(const float* input, float* output,
                                   int width, int height, int ksize) {
    // TODO: Implement 2D convolution using constant memory kernel
    // The kernel is already in d_const_kernel
}

int main() {
    printHeader("2D Convolution");

    int image_size = IMAGE_WIDTH * IMAGE_HEIGHT;
    int kernel_size = KERNEL_SIZE * KERNEL_SIZE;

    // Allocate host memory
    float *h_input, *h_kernel, *h_output, *h_output_ref;
    h_input = (float*)malloc(image_size * sizeof(float));
    h_kernel = (float*)malloc(kernel_size * sizeof(float));
    h_output = (float*)malloc(image_size * sizeof(float));
    h_output_ref = (float*)malloc(image_size * sizeof(float));

    // Initialize input data (random image)
    initRandom(h_input, image_size, 0.0f, 255.0f);

    // Initialize Gaussian blur kernel (normalized)
    float kernel_sum = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
        h_kernel[i] = 1.0f;  // Simple box filter
        kernel_sum += h_kernel[i];
    }
    for (int i = 0; i < kernel_size; i++) {
        h_kernel[i] /= kernel_sum;  // Normalize
    }

    // CPU execution
    printf("Running CPU version... ");
    fflush(stdout);
    CpuTimer cpu_timer;
    cpu_timer.start();
    conv2dCPU(h_input, h_kernel, h_output_ref, IMAGE_WIDTH, IMAGE_HEIGHT, KERNEL_SIZE);
    cpu_timer.stop();
    float cpu_time = cpu_timer.elapsed();
    printf("done.\n");

    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    cudaCheck(cudaMalloc(&d_input, image_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_kernel, kernel_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, image_size * sizeof(float)));

    // Copy data to device
    cudaCheck(cudaMemcpy(d_input, h_input, image_size * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    // GPU execution
    dim3 blockSize(16, 16);
    dim3 gridSize((IMAGE_WIDTH + blockSize.x - 1) / blockSize.x,
                  (IMAGE_HEIGHT + blockSize.y - 1) / blockSize.y);

    GpuTimer gpu_timer;
    gpu_timer.start();
    conv2dKernel<<<gridSize, blockSize>>>(d_input, d_kernel, d_output,
                                          IMAGE_WIDTH, IMAGE_HEIGHT, KERNEL_SIZE);
    gpu_timer.stop();
    float gpu_time = gpu_timer.elapsed();

    // Copy result back to host
    cudaCheck(cudaMemcpy(h_output, d_output, image_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify result
    bool passed = verifyResult(h_output, h_output_ref, image_size, 0.01f);

    printf("Image size:   %d x %d\n", IMAGE_WIDTH, IMAGE_HEIGHT);
    printf("Kernel size:  %d x %d\n", KERNEL_SIZE, KERNEL_SIZE);
    printf("CPU time:     %.3f ms\n", cpu_time);
    printf("GPU time:     %.3f ms\n", gpu_time);
    if (gpu_time > 0) {
        printf("Speedup:      %.2fx\n", cpu_time / gpu_time);
    }
    printf("Verification: %s\n", passed ? "PASSED ✓" : "FAILED ✗");
    printf("================================================================================\n");

    // Cleanup
    free(h_input); free(h_kernel); free(h_output); free(h_output_ref);
    cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);

    return passed ? 0 : 1;
}
