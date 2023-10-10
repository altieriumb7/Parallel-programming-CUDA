#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// GPU helper function for bottom-up merge
__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end);

// GPU helper function to calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks);

// GPU mergesort kernel
__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks);
// Mergesort function
void mergesort(long* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid);
int isSorted(long* data, long size);



//------------
__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks);
__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end, long* shared_mem);
void mergesort_shared(long* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid);