#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../lib/constants.cuh"
__global__ void parallelRadix(unsigned int *ddata);
__device__ unsigned int custom_popc(unsigned int value);