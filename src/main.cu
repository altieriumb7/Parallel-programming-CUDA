#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "../lib/merge_sort.cuh"
#include "../lib/utils.cuh"
#include "../lib/quick_sort.cuh"
#include "../lib/radix_sort.cuh"
#include "../lib/constants.cuh"
#include "../lib/utils_conf.cuh"

int main(int argc, char *argv[]) {

    bool sorted[6];
    unsigned long long N = 1024;
    int *data, *dev_data;
    bool stream_inout = false;
    double sorting_time[6],t_start=0,t_stop=0;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-w") == 0)
        {
            stream_inout = true;
        }
        else
        {
            N = atoi(argv[i]);
        }
    }
    
    Config config = determine_config(N);
    config.blockSize = dim3(config.threads_per_block);
    config.gridSize = dim3(config.total_blocks);
    const size_t size_array = N * sizeof(int);

    data = (int *)malloc(size_array);
    cudaMalloc((void **)&dev_data, size_array);
    //.------------------------------------------------------------------- radix sort parallel global memory ---------------------------------------------------------------
    fill_array(data, N);
    cudaMemcpy(dev_data, data, size_array, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    t_start = time_now();
    radixSort<<<1, N>>>(dev_data);

    t_stop = time_now();
    cudaPeekAtLastError();
    cudaMemcpy(data, dev_data, size_array, cudaMemcpyDeviceToHost);
    sorted[3]=is_sorted(data,N);
    sorting_time[3] = t_stop - t_start;
    
    if (sorted[3]){
        printf("Sorted properly using Radix Sorting Parallel global mem.\n");
        printf("Time for sorting: %lf s\n", sorting_time[3]);

    }else{
        printf("Error in sorting radix sort global mem");
    }
    zero_array(data, N);
    //-------------------------------------------------------------------radix sort parallel shared memery---------------------------------------------------------------------
    fill_array(data, N);
    cudaMemcpy(dev_data, data, size_array, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    t_start = time_now();
    radixSortShared<<<1, N>>>(dev_data);

    t_stop = time_now();
    cudaPeekAtLastError();
    cudaMemcpy(data, dev_data, size_array, cudaMemcpyDeviceToHost);
    sorted[4]=is_sorted(data,N);
    sorting_time[4] = t_stop - t_start;
    
    if (sorted[4]){
        printf("Sorted properly using Radix Sorting Parallel shared mem.\n");
        printf("Time for sorting: %lf s\n", sorting_time[4]);

    }else{
        printf("Error in sorting radix sort shared mem");
    }
    zero_array(data, N);
    //----------------------------------------------------------------------------quick sort parallel shared memory ------------------------------------------------
    fill_array(data, N);
    cudaMemcpy(dev_data, data, size_array, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    t_start = time_now();
    quick_sort_p(dev_data, 0, N - 1,config.blockSize.x,config.gridSize.x);
    t_stop = time_now();
    cudaPeekAtLastError();
    cudaMemcpy(data, dev_data, size_array, cudaMemcpyDeviceToHost);
    sorted[0]=is_sorted(data,N);
    sorting_time[0] = t_stop - t_start;
    if (sorted[0]){
        printf("Sorted properly using Quick Sorting Parallel shared mem.\n");
        printf("Time for sorting: %lf s\n", sorting_time[0]);
    }else{
        printf("Error in sorting quick sort shared mem");
    }

   
    
    zero_array(data, N);
    
    //----------------------------------------------------------------------------quick sort parallel global memory ------------------------------------------------
    fill_array(data, N);
    cudaMemcpy(dev_data, data, size_array, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    t_start = time_now();
    quick_sort_p(dev_data, 0, N - 1,config.blockSize.x,config.gridSize.x);
    t_stop = time_now();
    cudaPeekAtLastError();
    cudaMemcpy(data, dev_data, size_array, cudaMemcpyDeviceToHost);
    sorted[1]=is_sorted(data,N);
    sorting_time[1] = t_stop - t_start;
    
    if (sorted[1]){
        printf("Sorted properly using Quich Sorting Parallel global mem.\n");
        printf("Time for sorting: %lf s\n", sorting_time[1]);
    }else{
        printf("Error in sorting quick sort global mem");
    }
    zero_array(data, N);
    //----------------------------------------------------------------------------merge sort parallel global memory ------------------------------------------------

    fill_array(data, N);
    cudaMemcpy(dev_data, data, size_array, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    t_start = time_now();
    mergeSort_p(dev_data,N, config.threads_per_block, config.total_blocks);
    t_stop = time_now();
    cudaPeekAtLastError();
    cudaMemcpy(data, dev_data, size_array, cudaMemcpyDeviceToHost);
    sorted[2]=is_sorted(data,N);
    sorting_time[2] = t_stop - t_start;
    
    if (sorted[2]){
        printf("Sorted properly using Merge Sorting Parallel global mem.\n");
        printf("Time for sorting: %lf s\n", sorting_time[2]);
    }else{
        printf("Error in sorting merge sort shared mem");
    }
    zero_array(data, N);
    return 0;
    
    

    
}
