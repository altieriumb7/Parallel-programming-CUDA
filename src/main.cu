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
    unsigned int hdata[arraySize];
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
        struct timeval t1, t2;
        gettimeofday(&t1, NULL);

        parallelRadix<<<1, WSIZE>>>();
        cudaDeviceSynchronize();

        // Execution time measurement: stop the clock
        gettimeofday(&t2, NULL);

        // Calculate the execution time
        long long duration = (t2.tv_sec - t1.tv_sec) * 1000000LL + (t2.tv_usec - t1.tv_usec);
        duration /= 1000; // Convert to milliseconds
        // Summation of each loop's execution time
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





