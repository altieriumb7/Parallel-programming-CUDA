
#include "../lib/quick_sort.cuh"
__device__ int d_size;

 void printArr( int arr[], int n )
{
    int i;
    for ( i = 0; i < n; ++i )
        printf( "%d ", arr[i] );
}

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

__shared__ int sharedArr[SHARED_MEM_SIZE]; 


__global__ void partition_shared(int *arr, int *arr_l, int *arr_h, int n)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    d_size = 0;
    __syncthreads();

    if (z < n)
    {
        // Load data into shared memory
        sharedArr[threadIdx.x] = arr[z];
        __syncthreads();

        int h = arr_h[z];
        int l = arr_l[z];
        int x = sharedArr[threadIdx.x]; // Use shared memory
        int i = (l - 1);
        int temp;
        
        for (int j = l; j <= h - 1; j++)
        {
            if (sharedArr[j - l] <= x) // Adjust for shared memory
            {
                i++;
                temp = sharedArr[i - l]; // Adjust for shared memory
                sharedArr[i - l] = sharedArr[j - l]; // Adjust for shared memory
                sharedArr[j - l] = temp; // Adjust for shared memory
            }
        }

        temp = sharedArr[i + 1 - l]; // Adjust for shared memory
        sharedArr[i + 1 - l] = sharedArr[h - l]; // Adjust for shared memory
        sharedArr[h - l] = temp; // Adjust for shared memory
        int p = (i + 1);

        if (p - 1 > l)
        {
            int ind = atomicAdd(&d_size, 1);
            arr_l[ind] = l;
            arr_h[ind] = p - 1;
        }
        if (p + 1 < h)
        {
            int ind = atomicAdd(&d_size, 1);
            arr_l[ind] = p + 1;
            arr_h[ind] = h;
        }

        // Store the results back to global memory
        arr[z] = sharedArr[threadIdx.x];
    }
}

void quickSortIterative_shared(int arr[], int l, int h)
{
    int lstack[h - l + 1], hstack[h - l + 1];

    int top = -1, *d_d, *d_l, *d_h;

    lstack[++top] = l;
    hstack[top] = h;

    cudaMalloc(&d_d, (h - l + 1) * sizeof(int));
    cudaMemcpy(d_d, arr, (h - l + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_l, (h - l + 1) * sizeof(int));
    cudaMemcpy(d_l, lstack, (h - l + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_h, (h - l + 1) * sizeof(int));
    cudaMemcpy(d_h, hstack, (h - l + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int n_t = 1;
    int n_b = 1;
    int n_i = 1;
    while (n_i > 0)
    {
        partition<<<n_b, n_t>>>(d_d, d_l, d_h, n_i);
        int answer;
        cudaMemcpyFromSymbol(&answer, d_size, sizeof(int), 0, cudaMemcpyDeviceToHost);
        if (answer < 1024)
        {
            n_t = answer;
        }
        else
        {
            n_t = 1024;
            n_b = answer / n_t + (answer % n_t == 0 ? 0 : 1);
        }
        n_i = answer;
        cudaMemcpy(arr, d_d, (h - l + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    }
}


