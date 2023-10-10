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
    printArr( arr, n );
    return 0;
}