#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h> // Include for strcmp
#include <cuda.h>
#include <assert.h>
#include "../lib/merge_sort.cuh"
#include "../lib/utils.cuh"
#include "../lib/utilsParallelSort.cuh"



#include <cuda_runtime.h>

using namespace std;

#define size 10000

int main(int argc, char** argv) 
{
    clock_t start, end;
    double cput;

    start = clock();

    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 8;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    int inputLength;
    float* data = nullptr;

    // Initialize data here or load it from a file

    mergesort(data, threadsPerBlock, blocksPerGrid);

    // Print the sorted array or save it to a file

    

    end = clock();
    cput = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Execution time: %lf\n", cput);
    if (is_sorted(data,size)){
        printf("array sorted");
    }else{
        printf("array not sorted");
    }

    return 0;
}

