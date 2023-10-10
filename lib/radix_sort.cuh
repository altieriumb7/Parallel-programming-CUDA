 
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include "../lib/constants.cuh"

// Partition an array of unsigned integers by a specific bit
__device__ void partition_by_bit(unsigned int *values, unsigned int bit);

// Perform radix sort on an array of unsigned integers
__global__ void radix_sort(unsigned int *values);

// Perform parallel prefix sum (scan) on an array of unsigned integers
__device__ int plus_scan(unsigned int *x);

#include "../lib/radix_sort.cuh"

// Define the size of the shared memory buffer
#define SHARED_MEM_SIZE 1024

__device__ void partition_by_bit_shared(unsigned int *values, unsigned int bit, unsigned int *shared_mem);

__global__ void radix_sort_shared(unsigned int *values);

__device__ int plus_scan(unsigned int *x, unsigned int *shared_mem);

__device__ void partition_by_bit(unsigned int *values, unsigned int bit, unsigned int *shared_mem);


