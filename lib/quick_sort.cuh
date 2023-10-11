 
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include "../lib/constants.cuh"

__device__ unsigned int d_partitionSize;

__global__ void partition(unsigned int *d_data, unsigned int *d_low, unsigned int *d_high, unsigned int d_size);

void quick_sort_p(unsigned int d_array[], unsigned int d_start, unsigned int d_end);

__global__ void partition_shared(unsigned int *d_array, unsigned int *d_low, unsigned int *d_high, unsigned int d_size);

void quick_sort_p_shared(unsigned int d_array[], unsigned int d_start, unsigned int d_end);
