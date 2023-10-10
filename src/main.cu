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


#include <cuda_runtime.h>

#define size 10000

int main() {
    // Your existing code for sorting 'arr' goes here

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

    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 8;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    // Create an array of numbers (you can replace this with your input)
    long data[5000];
    long size_data = sizeof(data) / sizeof(data[0]);

    // Print unsorted data
    for (int i = 0; i < 5000; i++) {
        data[i] = rand() % 100000;
    }

    // Sort the data using mergesort
    mergesort(data, size_data, threadsPerBlock, blocksPerGrid);

    

    // Check if the array is sorted
    if (isSorted(data, size_data)) {
        printf("Array is sorted.\n");
    } else {
        printf("Array is not sorted.\n");
    }
    //
        // Your provided mergesort code with checks
    
    long data2[5000];
    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 8;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    // Create an array of numbers (you can replace this with your input)
    size_data = sizeof(data2) / sizeof(data2[0]);

    // Print unsorted data
    for (int i = 0; i < 5000; i++) {
        data2[i] = rand() % 100000;
    }

    // Sort the data using mergesort
    mergesort_shared(data2, size_data, threadsPerBlock, blocksPerGrid);

    

    // Check if the array is sorted
    if (isSorted(data2, size_data)) {
        printf("Array is sorted.\n");
    } else {
        printf("Array is not sorted.\n");
    }
    printf("Sorted data: ");
    for (int i = 0; i < size_data; i++) {
        printf("%ld ", data2[i]);
    }
    printf("\n");

}
