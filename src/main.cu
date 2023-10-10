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
    const int arraySize = WSIZE * LOOPS;
    unsigned int hdata[arraySize],*ddata;
    const size_t size_array = arraySize* sizeof(unsigned int);
    double t_start = 0, t_stop = 0,
    cudaMalloc((void **)&ddata, size_array);
    float totalTime = 0;

    srand(time(NULL));

    for (int lcount = 0; lcount < LOOPS; lcount++) {
        // Array elements have values in the range of 1024
        unsigned int range = 1U << UPPER_BIT;

        // Fill the array with random elements
        for (int i = 0; i < arraySize; i++) {
            hdata[i] = rand() % range;
        }

        cudaMemcpyToSymbol(ddata, hdata, arraySize * sizeof(unsigned int));

        // Execution time measurement: start the clock
        t_start = get_time();

        parallelRadix<<<1, WSIZE>>>(ddata);
        cudaDeviceSynchronize();

        // Execution time measurement: stop the clock
        t_stop = get_time();

        // Calculate the execution time
        long long duration = t_start-t_stop;
        
        totalTime += duration;

        // Copy data from device to host
        cudaMemcpyFromSymbol(hdata, ddata, arraySize * sizeof(unsigned int));
    }

    if (isSorted(hdata, arraySize)) {
        printf("Shared memory kernel: Array is sorted correctly.\n");
    } else {
        printf("Shared memory kernel: Array is NOT sorted correctly.\n");
    }

    printf("Parallel Radix Sort:\n");
    printf("Array size = %d\n", arraySize);
    printf("Time elapsed = %g milliseconds\n", totalTime);

    return 0;
}





