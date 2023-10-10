#include <iostream>
#include <stdlib.h>
#include "helper_cuda.h"
#include <sys/time.h>
#include "wb.h"
using namespace std;
#include "../lib/utilsParallelSort.cuh"




bool verbose;
int main(int argc, char** argv) 
{
    double start=0,end=0;
    double cput;
    unsigned short *data, *dev_data;
    unsigned long long N = 512;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-w") == 0)
        {
            write_output = true;
        }
        else
        {
            N = atoi(argv[i]);
        }
    }
    const size_t size_array = N * sizeof(unsigned short);
    data = (unsigned short *)malloc(size_array);
    cudaHandleError(cudaMalloc((void **)&dev_data, size_array));
    init_array(data, N);
    sort_config = determine_config(N);

    sort_config.blockSize = dim3(sort_config.threads_per_block);
    sort_config.gridSize = dim3(sort_config.total_blocks);
    cudaHandleError(cudaMemcpy(dev_data, data, size_array, cudaMemcpyHostToDevice));

    
    start = get_time();

    mergesort(data, sort_config.blockSize, sort_config.gridSize);
    end = get_time();
    cudaHandleError(cudaPeekAtLastError());
    cudaHandleError(cudaMemcpy(data, dev_data, size_array, cudaMemcpyDeviceToHost));

    double cput = ((double)(end - start)) / 1000;
    printf("\nRunning time = %f s\n", cput);
    if(is_sorted(data,N)){
        printf('Array sorted propertly');
    }else{
        printf('Array sorted improperly');
    }
    return 0;
}
