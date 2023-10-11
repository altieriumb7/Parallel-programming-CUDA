 
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include "../lib/constants.cuh"

__global__ void partition(int *d_data, int *d_low, int *d_high, int d_size);
void quick_sort_p(int d_array[], int d_start, int d_end, int numBlocks, int numThreads);

__global__ void partition_shared(int *d_array, int *d_low, int *d_high, int d_size);
void quick_sort_p_shared(int d_array[], int d_start, int d_end, int numBlocks, int numThreads);

