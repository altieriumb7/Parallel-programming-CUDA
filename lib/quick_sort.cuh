 
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include "../lib/constants.cuh"
__global__ void partition(unsigned short *d_data, unsigned short *d_low, unsigned short *d_high, unsigned short d_size);

void quick_sort_p(unsigned short d_array[], unsigned short d_start, unsigned short d_end, unsigned short numBlocks, unsigned short numThreads);

__global__ void partition_shared(unsigned short *d_array, unsigned short *d_low, unsigned short *d_high, unsigned short d_size);

void quick_sort_p_shared(unsigned short d_array[], unsigned short d_start, unsigned short d_end, unsigned short numBlocks, unsigned short numThreads);

