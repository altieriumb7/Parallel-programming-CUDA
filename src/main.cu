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
    double elapsed_time[6];// 6 algorithms to test
    unsigned long long N = 1024;
    unsigned int *data, *dev_data;
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
    const size_t size_array = N * sizeof(unsigned int);

    data = (unsigned int *)malloc(size_array);
    cudaMalloc((void **)&dev_data, size_array);
    
    //----------------------------------------------------------------------------quick sort parallel global memory ------------------------------------------------
    fill_array(data, N);
    cudaMemcpy(dev_data, data, size_array, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    t_start = time_now();
    quick_sort_p_shared(dev_data, 0, N - 1);
    t_stop = time_now();
    cudaPeekAtLastError();
    cudaMemcpy(data, dev_data, size_array, cudaMemcpyDeviceToHost);
    sorted[0]=is_sorted(data,N);
    sorting_time[0] = t_stop - t_start;
    bzero(data, size_array);

    //----------------------------------------------------------------------------quick sort parallel shared memory ------------------------------------------------
    fill_array(data, N);
    cudaMemcpy(dev_data, data, size_array, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    t_start = time_now();
    quick_sort_p(dev_data, 0, N - 1);
    t_stop = time_now();
    cudaPeekAtLastError();
    cudaMemcpy(data, dev_data, size_array, cudaMemcpyDeviceToHost);
    sorted[1]=is_sorted(data,N);
    sorting_time[1] = t_stop - t_start;
    bzero(data, size_array);

    printf("Sorted Quick Sorting Parallel glob. mem.: %d\n", sorted[0]);
    printf("Time for sorting: %lf\n", sorting_time[0]);

    printf("Sorted Quick Sorting Parallel shared mem.: %d\n", sorted[1]);
    printf("Time for sorting: %lf\n", sorting_time[1]);


   
    //.------------------------------------------------------------------- 

    /*
    int *device_array2;
    cudaMalloc(&device_array2, 1000 * sizeof(int)); // Change the type here
    cudaMemcpy(device_array2, host_array2, 1000 * sizeof(int), cudaMemcpyHostToDevice);
    quickSortIterative_shared(device_array2, 0, 1000 - 1); 


    cudaMemcpy(host_array2, device_array2, 1000 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //-----
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
    mergesort(data, size_data, config.threads_per_block, config.total_blocks);

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
    
    //-----------------------------------------------------------------------------------------------------------------------------------------------------
    // Testing the shared radix sort
    int host_array2[1000];
    for (int i = 0; i < 1000; i++) {
        host_array2[i] = rand() % 10000;
    }

    int *device_array2;
    cudaMalloc(&device_array2, 1000 * sizeof(int)); // Change the type here
    cudaMemcpy(device_array2, host_array2, 1000 * sizeof(int), cudaMemcpyHostToDevice);
    quickSortIterative_shared(device_array2, 0, 1000 - 1); 


    cudaMemcpy(host_array2, device_array2, 1000 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    sorted_shared = true;
    for (int i = 1; i < 1000; i++) {

        if (host_array2[i - 1] > host_array2[i]) {
            sorted_shared = false;
            break;
        }
    }

    if (sorted_shared) {
        printf("Array 'host_array2' is sorted using shared quick sort.\n");
    } else {
        printf("Array 'host_array2' is not sorted using shared quick sort.\n");
    }

    // Cleanup
    cudaFree(device_array2);
    */
    return 0;
}
