 
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include "../lib/constants.cuh"

// Kernel for performing radix sort
__global__ void radixSort(int *values);

// Device function for inclusive scan (prefix sum)
__device__ int inclusiveScan(int *x);

// Device function to partition values based on a specific bit
__device__ void partitionByBit(int *values, int bit);

// Kernel for performing radix sort using shared memory
__global__ void radixSortShared(int *values);

// Device function for inclusive scan (prefix sum) using shared memory
__device__ int inclusiveScanShared(int *x);

// Device function to partition values based on a specific bit using shared memory
__device__ void partitionByBitShared(int *values, int bit);
