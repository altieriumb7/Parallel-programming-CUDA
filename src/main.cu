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
    const int arraySize = 10000;
    unsigned int hdata[arraySize];
    float totalTime_glob = 0;
    float totalTime_shared = 0;

    for (int lcount = 0; lcount < LOOPS; lcount++) {
        // Array elements have values in the range of 1024
        unsigned int range = 1 << UPPER_BIT;

        // Fill the array with random elements
        for (int i = 0; i < arraySize; i++) {
            hdata[i] = rand() % range;
        }

        // Copy data from host to device for global memory kernel
        cudaMemcpyToSymbol(ddata_glob, hdata, arraySize * sizeof(unsigned int));
        // Copy data from host to device for shared memory kernel
        cudaMemcpyToSymbol(ddata_shared, hdata, arraySize * sizeof(unsigned int));

        // Execution time measurement for global memory kernel
        auto t1_glob = std::chrono::high_resolution_clock::now();
        parallelRadix_glob<<<1, WSIZE>>>();
        cudaDeviceSynchronize();
        auto t2_glob = std::chrono::high_resolution_clock::now();
        auto duration_glob = std::chrono::duration_cast<std::chrono::milliseconds>(t2_glob - t1_glob).count();
        totalTime_glob += duration_glob;

        // Execution time measurement for shared memory kernel
        auto t1_shared = std::chrono::high_resolution_clock::now();
        parallelRadix_shared<<<1, WSIZE>>>();
        cudaDeviceSynchronize();
        auto t2_shared = std::chrono::high_resolution_clock::now();
        auto duration_shared = std::chrono::duration_cast<std::chrono::milliseconds>(t2_shared - t1_shared).count();
        totalTime_shared += duration_shared;

        // Copy data from device to host for global memory kernel
        cudaMemcpyFromSymbol(hdata, ddata_glob, arraySize * sizeof(unsigned int));
    }

    printf("Parallel Radix Sort using Global Memory:\n");
    printf("Array size = %d\n", arraySize);
    printf("Time elapsed = %g milliseconds\n", totalTime_glob);

    printf("\nParallel Radix Sort using Shared Memory:\n");
    printf("Array size = %d\n", arraySize);
    printf("Time elapsed = %g milliseconds\n", totalTime_shared);

    return 0;
}