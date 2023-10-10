#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h> // Include for strcmp
#include <cuda.h>
#include <assert.h>
#include "../lib/merge_sort.cuh"
#include "../lib/utils.cuh"
#include "../lib/utilsParallelSort.cuh"

bool verbose;

int main(int argc, char** argv) {
    bool write_output = false;
    double start = 0, end = 0;
    double cput;
    unsigned short *data, *dev_data;
    unsigned long long N = 512;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-w") == 0) {
            write_output = true;
        } else {
            N = atoi(argv[i]);
        }
    }

    const size_t size_array = N * sizeof(unsigned short);
    data = (unsigned short *)malloc(size_array);
    cudaMalloc((void **)&dev_data, size_array);
    init_array(data, N);
    ParallelSortConfig sort_config = determine_config(N);

    sort_config.blockSize = dim3(sort_config.threads_per_block);
    sort_config.gridSize = dim3(sort_config.total_blocks);
    cudaMemcpy(dev_data, data, size_array, cudaMemcpyHostToDevice);

    start = get_time();

    mergesort(data, sort_config.blockSize, sort_config.gridSize,N);
    end = get_time();
    cudaPeekAtLastError();
    cudaMemcpy(data, dev_data, size_array, cudaMemcpyDeviceToHost);

    cput = ((double)(end - start)) / 1000;
    printf("\nRunning time = %g s\n", cput);

    if (is_sorted(dev_data, N)) {
        printf("Array sorted properly.\n");
    } else {
        printf("Array sorted improperly.\n");
    }

    return 0;
}
