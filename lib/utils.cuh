#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../lib/constants.cuh"

#include "../lib/utils.cuh"


void print_array(int* array, unsigned long long size);
double time_now(void);
void zero_array(int *data, const unsigned long long N);

void fill_array(int *data, const unsigned long long N);

bool is_sorted(int* result, const unsigned long long N);

bool is_power_of_two(const unsigned long x);