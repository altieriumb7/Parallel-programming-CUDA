#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <cuda.h>
#include <assert.h>
#include "../lib/merge_sort.cuh"
#include "../lib/utils.cuh"
#include "../lib/quick_sort.cuh"
#include "../lib/radix_sort.cuh"
#include "../lib/utilsParallelSort.cuh"
#include <cuda_runtime.h>

#define size 10000

int main() {
    // Your existing code for sorting 'arr' goes here
    ParallelSortConfig sort_config = determine_config(5000);

    sort_config.blockSize = dim3(sort_config.threads_per_block);
    sort_config.gridSize = dim3(sort_config.total_blocks);
    int arr[5000];
    srand(time(NULL));
    for (int i = 0; i < 5000; i++) {
        arr[i] = rand() % 10000;
    }
    int n = sizeof(arr) / sizeof(*arr);
    quickSortIterative(arr, 0, n - 1);

    int sorted = 1; // Assume it's sorted
    for (int i = 1; i < n; i++) {
        if (arr[i - 1] > arr[i]) {
            sorted = 0; // Array is not sorted
            break;
        }
    }

    if (sorted) {
        printf("Array 'arr' is sorted.\n");
    } else {
        printf("Array 'arr' is not sorted.\n");
    }

    // Add the provided code for radix sort here

    unsigned int a[1000];
    int size_a = 1000;
    srand(time(NULL));
    for (int i = 0; i < 1000; i++) {
        a[i] = rand() % 1000;
    }

    unsigned int *dev_a;
    cudaMalloc(&dev_a, size_a * sizeof(unsigned int));
    cudaMemcpy(dev_a, a, size_a * sizeof(unsigned int), cudaMemcpyHostToDevice);
    radix_sort<<<1, size_a>>>(dev_a);
    cudaMemcpy(a, dev_a, size_a * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    sorted = 1; // Assume it's sorted
    for (int i = 1; i < size_a; i++) {
        if (a[i - 1] > a[i]) {
            sorted = 0; // Array is not sorted
            break;
        }
    }

    if (sorted) {
        printf("Array 'a' is sorted.\n");
    } else {
        printf("Array 'a' is not sorted.\n");
    }

    // Your provided mergesort code with checks
    //-------------------------------------------------------------------------------------------------------------------------
    // Your existing code for sorting 'arr' goes here
    
    // ... (your existing code)

    // Create an array of numbers (you can replace this with your input)
    long data[5000];
    long size_data = sizeof(data) / sizeof(data[0]);

    // Print unsorted data
    for (int i = 0; i < 5000; i++) {
        data[i] = rand() % 100000;
    }

    // Sort the data using mergesort
    mergesort(data, size_data, sort_config.threads_per_block, sort_config.total_blocks);

    // Check if the array is sorted
    if (isSorted(data, size_data)) {
        printf("Array is sorted.\n");
    } else {
        printf("Array is not sorted.\n");
    }

    // Your provided mergesort code with checks
    //-------------------------------------------------------------------------------------------------------------------------
    
    // Testing the shared radix sort
    unsigned int host_array[1000];
    srand(time(NULL));
    for (int i = 0; i < 1000; i++) {
        host_array[i] = rand() % 1000;
    }

    unsigned int *device_array;
    cudaMalloc(&device_array, 1000 * sizeof(unsigned int));
    cudaMemcpy(device_array, host_array, 1000 * sizeof(unsigned int), cudaMemcpyHostToDevice);

    radix_sort_shared<<<1,1000 >>>(device_array);

    cudaMemcpy(host_array, device_array, 1000 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    bool sorted_shared = true;
    for (int i = 1; i < 1000; i++) {
        if (host_array[i - 1] > host_array[i]) {
            sorted_shared = false;
            break;
        }
    }

    if (sorted_shared) {
        printf("Array 'host_array' is sorted using shared radix sort.\n");
    } else {
        printf("Array 'host_array' is not sorted using shared radix sort.\n");
    }

    // Cleanup
    cudaFree(device_array);

    return 0;
}
