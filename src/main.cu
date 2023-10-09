#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <cuda.h>
#include <assert.h>
#include "../lib/radix_sort.cuh"
#include "../lib/utils.cuh"
#include "../lib/constants.cuh"
 
int main() {
    const int arraySize = 60;
    unsigned int hdata[arraySize];
    unsigned int ddata_glob[arraySize];

    unsigned int ddata_shared[arraySize];


    float totalTime_glob = 0;
    float totalTime_shared = 0;

    for (int lcount = 0; lcount < LOOPS; lcount++) {
        // Array elements have values in the range of 1024
        unsigned int range = 1 << UPPER_BIT;

        // Fill the array with random elements
        for (int i = 0; i < arraySize; i++) {
            hdata[i] = rand() % range;
        }

        cudaMalloc((void**)&ddata_glob, arraySize * sizeof(unsigned int));
        cudaMalloc((void**)&ddata_shared, arraySize * sizeof(unsigned int));

        // Copy data from host to device for global memory kernel
        cudaMemcpy(ddata_glob, hdata, arraySize * sizeof(unsigned int), cudaMemcpyHostToDevice);

        // Copy data from host to device for shared memory kernel
        cudaMemcpy(ddata_shared, hdata, arraySize * sizeof(unsigned int), cudaMemcpyHostToDevice);

        // Execution time measurement for global memory kernel
        cudaEvent_t start_glob, stop_glob;
        cudaEventCreate(&start_glob);
        cudaEventCreate(&stop_glob);
        cudaEventRecord(start_glob);

        parallelRadix_glob<<<1, WSIZE>>>(ddata_glob);
        cudaDeviceSynchronize();

        cudaEventRecord(stop_glob);
        cudaEventSynchronize(stop_glob);
        
        float elapsedTime_glob;
        cudaEventElapsedTime(&elapsedTime_glob, start_glob, stop_glob);
        totalTime_glob += elapsedTime_glob;
        cudaMemcpy(hdata, ddata_glob, arraySize * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        if (isSorted(hdata, arraySize)) {
            printf("Global memory kernel: Array is sorted correctly.\n");
        } else {
            printf("Global memory kernel: Array is NOT sorted correctly.\n");
        }

        // Execution time measurement for shared memory kernel
        cudaEvent_t start_shared, stop_shared;
        cudaEventCreate(&start_shared);
        cudaEventCreate(&stop_shared);
        cudaEventRecord(start_shared);

        parallelRadix_shared<<<1, WSIZE>>>(ddata_shared);
        cudaDeviceSynchronize();

        cudaEventRecord(stop_shared);
        cudaEventSynchronize(stop_shared);

        float elapsedTime_shared;
        cudaEventElapsedTime(&elapsedTime_shared, start_shared, stop_shared);
        totalTime_shared += elapsedTime_shared;

        // Copy data from device to host for global memory kernel
        cudaMemcpy(hdata, ddata_shared, arraySize * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (isSorted(ddata_shared, arraySize)) {
            printf("Shared memory kernel: Array is sorted correctly.\n");
        } else {
            printf("Shared memory kernel: Array is NOT sorted correctly.\n");
        }
        // Free device memory
        cudaFree(ddata_glob);
        cudaFree(ddata_shared);
    }
    printf("Parallel Radix Sort using Global Memory:\n");
    printf("Array size = %d\n", arraySize);
    printf("Time elapsed = %f milliseconds\n", totalTime_glob);

    printf("\nParallel Radix Sort using Shared Memory:\n");
    printf("Array size = %d\n", arraySize);
    printf("Time elapsed = %f milliseconds\n", totalTime_shared);


    return 0;
}