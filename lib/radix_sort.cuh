#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../lib/constants.cuh"
unsigned int ddata[WSIZE];

__global__ void parallelRadix();
__device__ unsigned int custom_popc(unsigned int value);