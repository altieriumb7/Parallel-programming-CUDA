#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../lib/constants.cuh"

__global__ void parallelRadix_shared(unsigned int* ddata);
__global__ void parallelRadix_glob(unsigned int* ddata);