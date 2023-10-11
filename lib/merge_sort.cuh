#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../lib/constants.cuh"

// GPU helper function for bottom-up merge
__device__ void gpuBottomUpMerge(unsigned int* src, unsigned int* dest, unsigned int start, unsigned int middle, unsigned int end);

// GPU mergesort kernel
__global__ void gpuMergeSort(unsigned int* source, unsigned int* destination, unsigned long long size, unsigned int width, unsigned int slices, dim3* threads, dim3* blocks);

// Mergesort function
void mergeSort(unsigned int* data, unsigned long long size, dim3 threadsPerBlock, dim3 blocksPerGrid);

// GPU helper function for bottom-up merge with shared memory
__device__ void gpuBottomUpMerge(unsigned int* source, unsigned int* dest, unsigned int start, unsigned int middle, unsigned int end, unsigned int* sharedMem);

// GPU mergesort kernel with shared memory
__global__ void gpuMergeSortShared(unsigned int* source, unsigned int* dest, unsigned long long size, unsigned int width, unsigned int slices, dim3* threads, dim3* blocks);

// Mergesort function with shared memory
void mergeSortShared(unsigned int* data, unsigned long long size, dim3 threadsPerBlock, dim3 blocksPerGrid);

__device__ unsigned int getThreadIndex(dim3* threads, dim3* blocks);
