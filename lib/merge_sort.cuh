#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

void mergesort(float* data, dim3 threadsPerBlock, dim3 blocksPerGrid);
__global__ void gpu_mergesort(float* source, float* dest, long width, long slices, dim3* threads, dim3* blocks);
__device__ void gpu_bottomUpMerge(float* source, float* dest, long start, long middle, long end);