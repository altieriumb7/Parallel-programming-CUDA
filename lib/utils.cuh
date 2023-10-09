#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../lib/constants.cuh"

double get_time(void);


void init_array(unsigned long *data, const unsigned long long N);


__host__ void print_array(const unsigned long *data, const unsigned long long N);


bool isSorted(const unsigned int *arr, int size);
bool is_power_of_two(const unsigned long x);

__host__ __device__ void get_max(unsigned long *data, const unsigned long long N, unsigned long *max);


__device__ void power(unsigned base, unsigned exp, unsigned *result);

void print_table(int n_algorithms, char algorithms[][100], char machine[][4], unsigned long threads[], bool used_shared[], bool correctness[], double elapsed_time[]);


void write_statistics_csv(int n, char algorithms[][100], double elapsed_times[]);