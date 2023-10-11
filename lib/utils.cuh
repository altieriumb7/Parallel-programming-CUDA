#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../lib/constants.cuh"

#include "../lib/utils.cuh"


void print_array(unsigned short* array, unsigned long long size);
double time_now(void);
void zero_array(unsigned short *data, const unsigned long long N);

void fill_array(unsigned short *data, const unsigned long long N);

bool is_sorted(unsigned short* result, const unsigned long long N);

bool is_power_of_two(const unsigned long x);