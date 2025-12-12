#include "../common/utils.h"
// #include <__clang_cuda_builtin_vars.h>
#include <random>

#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1080
#define CHANNELS 3  // RGB

// RGB to grayscale conversion weights (ITU-R BT.709)
#define R_WEIGHT 0.2126f
#define G_WEIGHT 0.7152f
#define B_WEIGHT 0.0722f

// CPU implementation
void rgbToGrayscaleCPU(const unsigned char* rgb, unsigned char* gray,
                       int width, int height) {
    int pixels = width * height;

    for (int i = 0; i < pixels; i++) {
        unsigned char r = rgb[i * CHANNELS + 0];
        unsigned char g = rgb[i * CHANNELS + 1];
        unsigned char b = rgb[i * CHANNELS + 2];

        gray[i] = (unsigned char)(R_WEIGHT * r + G_WEIGHT * g + B_WEIGHT * b);
    }
}

__global__ void rgbToGrayscaleKernel(const unsigned char* rgb, unsigned char* gray, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width){
        // Data given is in [H, W, C] format
        int startIndex = (row * width + col) * CHANNELS;
        
        unsigned char r = rgb[startIndex];
        unsigned char g = rgb[startIndex + 1];
        unsigned char b = rgb[startIndex + 2];

        gray[row * width + col] = (unsigned char)(r * R_WEIGHT + g * G_WEIGHT + b * B_WEIGHT);

    }
}

int main() {
    printHeader("RGB to Grayscale Conversion");

    int num_pixels = IMAGE_WIDTH * IMAGE_HEIGHT;
    int rgb_size = num_pixels * CHANNELS;
    int gray_size = num_pixels;

    // Allocate host memory
    unsigned char *h_rgb, *h_gray, *h_gray_ref;
    h_rgb = (unsigned char*)malloc(rgb_size * sizeof(unsigned char));
    h_gray = (unsigned char*)malloc(gray_size * sizeof(unsigned char));
    h_gray_ref = (unsigned char*)malloc(gray_size * sizeof(unsigned char));

    // Initialize random RGB image
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);
    for (int i = 0; i < rgb_size; i++) {
        h_rgb[i] = (unsigned char)dis(gen);
    }

    // CPU execution
    CpuTimer cpu_timer;
    cpu_timer.start();
    rgbToGrayscaleCPU(h_rgb, h_gray_ref, IMAGE_WIDTH, IMAGE_HEIGHT);
    cpu_timer.stop();
    float cpu_time = cpu_timer.elapsed();

    // Allocate device memory
    unsigned char *d_rgb, *d_gray;
    cudaCheck(cudaMalloc(&d_rgb, rgb_size * sizeof(unsigned char)));
    cudaCheck(cudaMalloc(&d_gray, gray_size * sizeof(unsigned char)));

    // Copy data to device
    cudaCheck(cudaMemcpy(d_rgb, h_rgb, rgb_size * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // GPU execution
    dim3 blockSize(16, 16);
    dim3 gridSize((IMAGE_WIDTH + blockSize.x - 1) / blockSize.x,
                  (IMAGE_HEIGHT + blockSize.y - 1) / blockSize.y);

    GpuTimer gpu_timer;
    gpu_timer.start();
    rgbToGrayscaleKernel<<<gridSize, blockSize>>>(d_rgb, d_gray, IMAGE_WIDTH, IMAGE_HEIGHT);
    gpu_timer.stop();
    float gpu_time = gpu_timer.elapsed();

    // Copy result back to host
    cudaCheck(cudaMemcpy(h_gray, d_gray, gray_size * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Verify result (allow ±1 difference due to rounding)
    bool passed = true;
    for (int i = 0; i < gray_size; i++) {
        if (abs((int)h_gray[i] - (int)h_gray_ref[i]) > 1) {
            printf("Mismatch at pixel %d: GPU=%d, CPU=%d\n", i, h_gray[i], h_gray_ref[i]);
            passed = false;
            break;
        }
    }

    printf("Image size:   %d x %d (%d pixels)\n", IMAGE_WIDTH, IMAGE_HEIGHT, num_pixels);
    printf("CPU time:     %.3f ms\n", cpu_time);
    printf("GPU time:     %.3f ms\n", gpu_time);
    if (gpu_time > 0) {
        printf("Speedup:      %.2fx\n", cpu_time / gpu_time);
    }
    printf("Verification: %s\n", passed ? "PASSED ✓" : "FAILED ✗");
    printf("================================================================================\n");

    // Cleanup
    free(h_rgb); free(h_gray); free(h_gray_ref);
    cudaFree(d_rgb); cudaFree(d_gray);

    return passed ? 0 : 1;
}
