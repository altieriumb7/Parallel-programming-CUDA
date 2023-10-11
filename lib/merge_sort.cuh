#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../lib/constants.cuh"
// Device function to merge two sorted arrays
__device__ void gpu_bottomUpMerge(int* source, int* dest, int start, int middle, int end) {
    // Function body remains the same
}

// Device function to calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    // Function body remains the same
}

// GPU kernel for mergesort
__global__ void gpu_mergesort(int* source, int* dest, int size, int width, int slices, dim3* threads, dim3* blocks) {
    // Function body remains the same
}

// Mergesort function
void mergesort(int* data, int size, dim3 threadsPerBlock, dim3 blocksPerGrid) {
    // Function body remains the same
}
