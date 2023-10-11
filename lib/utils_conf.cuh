#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../lib/utils.cuh"
#include "../lib/constants.cuh"
struct Config
{
    dim3 gridSize;
    dim3 blockSize;
    unsigned long long partition_size;
    unsigned long total_threads;
    unsigned long total_blocks;
    unsigned long threads_per_block;
    size_t required_shared_memory;
    int max_shared_memory_per_block;
};
Config determine_config(const unsigned long long N);
