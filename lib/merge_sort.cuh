#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../lib/constants.cuh"

void mergesort(float, dim3, dim3);
__global__ void gpu_mergesort(float, float, long, long, dim3, dim3);
__device__ void gpu_bottomUpMerge(float, float, long, long, long);