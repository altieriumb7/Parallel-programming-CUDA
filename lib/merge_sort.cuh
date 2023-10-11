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

__host__ void merge_sort_seq(int *data, const unsigned long long left, const unsigned long long right);
__device__ unsigned long long getThreadIndex(dim3* threads, dim3* blocks);
