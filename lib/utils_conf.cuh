#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../lib/utils.cuh"
#include "../lib/constants.cuh"
Config determine_config(const unsigned long long N);
