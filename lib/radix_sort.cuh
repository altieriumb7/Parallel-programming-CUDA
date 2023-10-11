 
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include "../lib/constants.cuh"


__global__ void radixSort(unsigned int *values);

__device__ int inclusiveScan(unsigned int *x);

__device__ void partitionByBit(unsigned int *values, unsigned int bit);

__global__ void radixSortShared(unsigned int *values);

__device__ int inclusiveScanShared(unsigned int *x);

__device__ void partitionByBitShared(unsigned int *values, unsigned int bit);


