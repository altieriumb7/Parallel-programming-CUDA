#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../lib/constants.cuh"

// GPU helper function for bottom-up merge
__device__ void gpuBottomUpMerge(int* src, int* dest, int start, int middle, int end);
// GPU mergesort kernel
__global__ void gpuMergeSort(int* source, int* destination, unsigned long long size, int width, int slices, dim3* threads, dim3* blocks);
// Mergesort function
void mergeSort( int* data, unsigned long long size, dim3 threadsPerBlock, dim3 blocksPerGrid);


// GPU helper function for bottom-up merge with shared memory
__device__ void gpuBottomUpMergeShared(int* source, int* dest, int start, int middle, int end, int* sharedMem);

// GPU mergesort kernel
__global__ void gpuMergeSortShared(int* source, int* dest, unsigned long long size, int width, int slices, dim3* threads, dim3* blocks);

// Mergesort function
void mergeSortShared(int* data, unsigned long long size, dim3 threadsPerBlock, dim3 blocksPerGrid);


__device__ unsigned int getThreadIndex(dim3* threads, dim3* blocks);
