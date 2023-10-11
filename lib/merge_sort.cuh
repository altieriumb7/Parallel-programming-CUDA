#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../lib/constants.cuh"
__device__ void gpuBottomUpMerge(int* src, int* dest, unsigned long long start, unsigned long long middle, unsigned long long end);
__global__ void gpuMergeSort(int* source, int* destination, unsigned long long size, unsigned long long width, unsigned long long slices, dim3* threads, dim3* blocks);
void mergeSort_p(int* data, unsigned long long size, dim3 threadsPerBlock, dim3 blocksPerGrid);

__device__ void gpuBottomUpMergeShared(int* source, int* dest, long long start, long long middle, long long end, int* sharedMem);
__global__ void gpuMergeSortShared(int* source, int* dest, unsigned long long size, unsigned long long width, unsigned long long slices, dim3* threads, dim3* blocks);
void mergeSortShared(int* data, unsigned long long size, dim3 threadsPerBlock, dim3 blocksPerGrid);


__device__ unsigned long long getThreadIndex(dim3* threads, dim3* blocks);
