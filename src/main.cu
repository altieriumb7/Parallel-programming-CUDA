#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h> // Include for strcmp
#include <cuda.h>
#include <assert.h>
#include "../lib/merge_sort.cuh"
#include "../lib/utils.cuh"
#include "../lib/quick_sort.cuh"



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

    return 0;
}