#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

__device__ float min(float a, float b);
void mergesort(float* data, dim3 threadsPerBlock, dim3 blocksPerGrid,unsigned long long size);
__global__ void gpu_mergesort(float* source, float* dest, long width, long slices, dim3* threads, dim3* blocks.unsigned long long size) ;
__device__ void gpu_bottomUpMerge(float* source, float* dest, long start, long middle, long end);