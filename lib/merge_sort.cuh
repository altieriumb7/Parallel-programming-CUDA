#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../lib/constants.cuh"
void mergesort(unsigned short *data, dim3 threadsPerBlock, dim3 blocksPerGrid, unsigned long long size);

__device__ unsigned int getIdx(dim3 threads, dim3 blocks);

__global__ void gpu_mergesort(unsigned short *source, unsigned short *dest, unsigned long long width, unsigned long long slices, dim3 threads, dim3 blocks, unsigned long long size);

__device__ void gpu_bottomUpMerge(unsigned short *source, unsigned short *dest, unsigned long long start, unsigned long long middle, unsigned long long end);
