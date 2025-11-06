#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// CUDA error checking macro
#define cudaCheck(err) __cudaCheck(err, __FILE__, __LINE__)
inline void __cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// CPU Timer
class CpuTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;

public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
    }

    float elapsed() {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0f; // return milliseconds
    }
};

// GPU Timer using CUDA events
class GpuTimer {
private:
    cudaEvent_t start_event, stop_event;

public:
    GpuTimer() {
        cudaCheck(cudaEventCreate(&start_event));
        cudaCheck(cudaEventCreate(&stop_event));
    }

    ~GpuTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() {
        cudaCheck(cudaEventRecord(start_event, 0));
    }

    void stop() {
        cudaCheck(cudaEventRecord(stop_event, 0));
        cudaCheck(cudaEventSynchronize(stop_event));
    }

    float elapsed() {
        float ms;
        cudaCheck(cudaEventElapsedTime(&ms, start_event, stop_event));
        return ms;
    }
};

// Data initialization helpers
void initRandom(float* data, int size, float min_val = 0.0f, float max_val = 1.0f);
void initSequential(float* data, int size);
void initRandomInt(int* data, int size, int min_val, int max_val);

// Verification helpers
bool verifyResult(const float* result, const float* expected, int size, float epsilon = 1e-3f);
bool verifyResultInt(const int* result, const int* expected, int size);

// Print helpers
void printResults(const char* kernel_name, int size, float cpu_time, float gpu_time, bool passed);
void printHeader(const char* kernel_name);

#endif // UTILS_H
