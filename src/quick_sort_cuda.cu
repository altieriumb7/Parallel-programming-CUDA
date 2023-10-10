
#include "../lib/quick_sort.cuh"

 void printArr( int arr[], int n )
{
    int i;
    for ( i = 0; i < n; ++i )
        printf( "%d ", arr[i] );
}
__device__ int d_size;

__global__ void partition (int *arr, int *arr_l, int *arr_h, int n)
{
    int z = blockIdx.x*blockDim.x+threadIdx.x;
    d_size = 0;
    __syncthreads();
    if (z<n)
      {
        int h = arr_h[z];
        int l = arr_l[z];
        int x = arr[h];
        int i = (l - 1);
        int temp;
        for (int j = l; j <= h- 1; j++)
          {
            if (arr[j] <= x)
              {
                i++;
                temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
              }
          }
        temp = arr[i+1];
        arr[i+1] = arr[h];
        arr[h] = temp;
        int p = (i + 1);
        if (p-1 > l)
          {
            int ind = atomicAdd(&d_size, 1);
            arr_l[ind] = l;
            arr_h[ind] = p-1;  
          }
        if ( p+1 < h )
          {
            int ind = atomicAdd(&d_size, 1);
            arr_l[ind] = p+1;
            arr_h[ind] = h; 
          }
      }
}
 
void quickSortIterative (int arr[], int l, int h)
{
    int lstack[ h - l + 1 ], hstack[ h - l + 1];
 
    int top = -1, *d_d, *d_l, *d_h;
 
    lstack[ ++top ] = l;
    hstack[ top ] = h;

    cudaMalloc(&d_d, (h-l+1)*sizeof(int));
    cudaMemcpy(d_d, arr,(h-l+1)*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc(&d_l, (h-l+1)*sizeof(int));
    cudaMemcpy(d_l, lstack,(h-l+1)*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc(&d_h, (h-l+1)*sizeof(int));
    cudaMemcpy(d_h, hstack,(h-l+1)*sizeof(int),cudaMemcpyHostToDevice);
    int n_t = 1;
    int n_b = 1;
    int n_i = 1; 
    while ( n_i > 0 )
    {
        partition<<<n_b,n_t>>>( d_d, d_l, d_h, n_i);
        int answer;
        cudaMemcpyFromSymbol(&answer, d_size, sizeof(int), 0, cudaMemcpyDeviceToHost); 
        if (answer < 1024)
          {
            n_t = answer;
          }
        else
          {
            n_t = 1024;
            n_b = answer/n_t + (answer%n_t==0?0:1);
          }
        n_i = answer;
        cudaMemcpy(arr, d_d,(h-l+1)*sizeof(int),cudaMemcpyDeviceToHost);
    }
}
 
//-----------------------------------------------------------------


__device__ int d_size_shared; // Device variable to store the size of the partition

__global__ void partition_shared(int* arr, int* arr_l, int* arr_h, int n) {
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int sharedArr[SHARED_MEM_SIZE]; // Shared memory for local storage
    int l = arr_l[z];
    int h = arr_h[z];
    int x = arr[h];
    int i = l - 1;
    
    // Copy data from global memory to shared memory
    for (int j = l; j <= h; j++) {
        sharedArr[j - l] = arr[j];
    }
    
    __syncthreads();
    
    // Partition data in shared memory
    for (int j = 0; j < h - l; j++) {
        if (sharedArr[j] <= x) {
            i++;
            int temp = sharedArr[i - l];
            sharedArr[i - l] = sharedArr[j];
            sharedArr[j] = temp;
        }
    }
    
    // Copy data from shared memory back to global memory
    for (int j = l; j <= h; j++) {
        arr[j] = sharedArr[j - l];
    }
    
    int p = i + 1;
    if (p - 1 > l) {
        int ind = atomicAdd(&d_size_shared, 1);
        arr_l[ind] = l;
        arr_h[ind] = p - 1;
    }
    if (p + 1 < h) {
        int ind = atomicAdd(&d_size_shared, 1);
        arr_l[ind] = p + 1;
        arr_h[ind] = h;
    }
}

void quickSortIterative_shared(int arr[], int l, int h, const int block_size) {
    int* d_d, *d_l, *d_h, *d_size;

    // Allocate device memory and transfer data
    cudaMalloc(&d_d, (h - l + 1) * sizeof(int));
    cudaMemcpy(d_d, arr + l, (h - l + 1) * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_l, (h - l + 1) * sizeof(int));
    cudaMemcpy(d_l, &l, (h - l + 1) * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_h, (h - l + 1) * sizeof(int));
    cudaMemcpy(d_h, &h, (h - l + 1) * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_size, sizeof(int));
    int initial_size = 1;
    cudaMemcpy(d_size, &initial_size, sizeof(int), cudaMemcpyHostToDevice);
    
    int n_t = block_size;
    int n_b = (h - l + 1) / n_t + ((h - l + 1) % n_t == 0 ? 0 : 1);
    int n_i = h - l + 1;
    
    while (n_i > 0) {
        partition_shared<<<n_b, n_t>>>(d_d, d_l, d_h, n_i);
        int answer;
        cudaMemcpyFromSymbol(&answer, d_size, sizeof(int), 0, cudaMemcpyDeviceToHost);
        
        if (answer < SHARED_MEM_SIZE) {
            n_t = answer;
        } else {
            n_t = SHARED_MEM_SIZE;
            n_b = answer / n_t + (answer % n_t == 0 ? 0 : 1);
        }
        n_i = answer;
    }
    
    cudaMemcpy(arr + l, d_d, (h - l + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_d);
    cudaFree(d_l);
    cudaFree(d_h);
    cudaFree(d_size);
}
