#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h> // Include for strcmp
#include <cuda.h>
#include <assert.h>
#include "../lib/merge_sort.cuh"
#include "../lib/utils.cuh"
#include "../lib/quick_sort.cuh"
#include "../lib/radix_sort.cuh"



#include <cuda_runtime.h>

using namespace std;

#define size 10000
int main()
{
    int arr[5000];
    srand(time(NULL));
    for (int i = 0; i<5000; i++)
       {
         arr[i] = rand ()%10000;
       }
    int n = sizeof( arr ) / sizeof( *arr );
    quickSortIterative( arr, 0, n - 1 );
    int sorted = 1; // Assume it's sorted
    for (int i = 1; i < n; i++) {
        if (arr[i - 1] > arr[i]) {
            sorted = 0; // Array is not sorted
            break;
        }
    }

    if (sorted) {
        printf("Array is sorted.\n");
    } else {
        printf("Array is not sorted.\n");
    }

    unsigned int a[5000];
    
    srand(time(NULL));
    for (int i = 0; i < 5000; i++)
    {
    a[i] = rand ()%10000;
    }
    unsigned int *dev_a;
    cudaMalloc(&dev_a, size * sizeof(unsigned int));
    cudaMemcpy( dev_a, a, size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    radix_sort<<<1,size>>>(dev_a);
    cudaMemcpy( a, dev_a, size * sizeof(unsigned int), cudaMemcpyDeviceToHost );
    sorted = 0;
    for (int i = 1; i < n; i++) {
        if (a[i - 1] > a[i]) {
            sorted = 0; // Array is not sorted
            break;
        }
    }

    if (sorted) {
        printf("Array is sorted.\n");
    } else {
        printf("Array is not sorted.\n");
    }
    return 0;
}