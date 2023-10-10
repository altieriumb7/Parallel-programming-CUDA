 
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include "../lib/constants.cuh"

void printArr( int arr[], int n );
__global__ void partition (int *arr, int *arr_l, int *arr_h, int n);
void quickSortIterative (int arr[], int l, int h);

__global__ void partition(int *arr, int *arr_l, int *arr_h, int n,int dsize);
__global__ void partition_shared(int *arr, int *arr_l, int *arr_h, int n);