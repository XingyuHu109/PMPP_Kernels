#include "utils.h"
#include <random>

void initRandom(float* data, int size, float min_val, float max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (int i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

void initSequential(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)i;
    }
}

void initRandomInt(int* data, int size, int min_val, int max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min_val, max_val);

    for (int i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

bool verifyResult(const float* result, const float* expected, int size, float epsilon) {
    for (int i = 0; i < size; i++) {
        if (fabs(result[i] - expected[i]) > epsilon) {
            printf("Mismatch at index %d: GPU=%.6f, CPU=%.6f\n", i, result[i], expected[i]);
            return false;
        }
    }
    return true;
}

bool verifyResultInt(const int* result, const int* expected, int size) {
    for (int i = 0; i < size; i++) {
        if (result[i] != expected[i]) {
            printf("Mismatch at index %d: GPU=%d, CPU=%d\n", i, result[i], expected[i]);
            return false;
        }
    }
    return true;
}

void printHeader(const char* kernel_name) {
    printf("\n");
    printf("================================================================================\n");
    printf("  %s\n", kernel_name);
    printf("================================================================================\n");
}

void printResults(const char* kernel_name, int size, float cpu_time, float gpu_time, bool passed) {
    printf("Size:         %d elements\n", size);
    printf("CPU time:     %.3f ms\n", cpu_time);
    printf("GPU time:     %.3f ms\n", gpu_time);

    if (gpu_time > 0) {
        float speedup = cpu_time / gpu_time;
        printf("Speedup:      %.2fx\n", speedup);
    }

    printf("Verification: %s\n", passed ? "PASSED ✓" : "FAILED ✗");
    printf("================================================================================\n");
}
