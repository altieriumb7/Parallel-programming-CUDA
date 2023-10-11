#include "../lib/quick_sort.cuh"

// Device variable to store the partition size
__device__ int d_partitionSize;

// GPU kernel for partitioning data
__global__ void partition(int *d_data, int *d_low, int *d_high, int d_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_partitionSize = 0;  // Initialize partition size to 0
    __syncthreads();

    if (idx < d_size)
    {
        int high = d_high[idx];
        int low = d_low[idx];
        int pivot = d_data[high];
        int i = (low - 1);
        int temp;

        for (int j = low; j <= high - 1; j++)
        {
            if (d_data[j] <= pivot)
            {
                i++;
                temp = d_data[i];
                d_data[i] = d_data[j];
                d_data[j] = temp;
            }
        }

        temp = d_data[i + 1];
        d_data[i + 1] = d_data[high];
        d_data[high] = temp;
        int partitionPoint = (i + 1);

        if (partitionPoint - 1 > low)
        {
            int ind = atomicAdd(&d_partitionSize, 1);
            d_low[ind] = low;
            d_high[ind] = partitionPoint - 1;
        }
        if (partitionPoint + 1 < high)
        {
            int ind = atomicAdd(&d_partitionSize, 1);
            d_low[ind] = partitionPoint + 1;
            d_high[ind] = high;
        }
    }
}

// Host function for iterative quicksort on the GPU
void quick_sort_p(int d_array[], int d_start, int d_end, int numBlocks, int numThreads)
{
    int lowStack[d_end - d_start + 1], highStack[d_end - d_start + 1];

    int top = -1;
    int *d_d, *d_low, *d_high;

    lowStack[++top] = d_start;
    highStack[top] = d_end;

    cudaMalloc(&d_d, (d_end - d_start + 1) * sizeof(int));
    cudaMemcpy(d_d, d_array, (d_end - d_start + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_low, (d_end - d_start + 1) * sizeof(int));
    cudaMemcpy(d_low, lowStack, (d_end - d_start + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_high, (d_end - d_start + 1) * sizeof(int));
    cudaMemcpy(d_high, highStack, (d_end - d_start + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int numIterations = 1;

    while (numIterations > 0)
    {
        partition<<<numBlocks, numThreads>>>(d_d, d_low, d_high, numIterations);
        int partitionSize;
        cudaMemcpyFromSymbol(&partitionSize, d_partitionSize, sizeof(int), 0, cudaMemcpyDeviceToHost);

        if (partitionSize < 1024)
        {
            numThreads = partitionSize;
        }
        else
        {
            numThreads = 1024;
            numBlocks = partitionSize / numThreads + (partitionSize % numThreads == 0 ? 0 : 1);
        }
        numIterations = partitionSize;
        cudaMemcpy(d_array, d_d, (d_end - d_start + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    }
}

// Shared memory for the partition
__shared__ int sharedArr[SHARED_MEM_SIZE];

// GPU kernel for partitioning data using shared memory
__global__ void partition_shared(int *d_array, int *d_low, int *d_high, int d_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_partitionSize = 0;
    __syncthreads();

    if (idx < d_size)
    {
        // Load data into shared memory
        sharedArr[threadIdx.x] = d_array[idx];
        __syncthreads();

        int high = d_high[idx];
        int low = d_low[idx];
        int pivot = sharedArr[threadIdx.x]; // Use shared memory
        int i = (low - 1);
        int temp;

        for (int j = low; j <= high - 1; j++)
        {
            if (sharedArr[j - low] <= pivot) // Adjust for shared memory
            {
                i++;
                temp = sharedArr[i - low]; // Adjust for shared memory
                sharedArr[i - low] = sharedArr[j - low]; // Adjust for shared memory
                sharedArr[j - low] = temp; // Adjust for shared memory
            }
        }

        temp = sharedArr[i + 1 - low]; // Adjust for shared memory
        sharedArr[i + 1 - low] = sharedArr[high - low]; // Adjust for shared memory
        sharedArr[high - low] = temp; // Adjust for shared memory
        int partitionPoint = (i + 1);

        if (partitionPoint - 1 > low)
        {
            int ind = atomicAdd(&d_partitionSize, 1);
            d_low[ind] = low;
            d_high[ind] = partitionPoint - 1;
        }
        if (partitionPoint + 1 < high)
        {
            int ind = atomicAdd(&d_partitionSize, 1);
            d_low[ind] = partitionPoint + 1;
            d_high[ind] = high;
        }

        // Store the results back to global memory
        d_array[idx] = sharedArr[threadIdx.x];
    }
}

// Host function for iterative quicksort on the GPU using shared memory
void quick_sort_p_shared(int d_array[], int d_start, int d_end, int numBlocks, int numThreads)
{
    int lowStack[d_end - d_start + 1], highStack[d_end - d_start + 1];

    int top = -1;
    int *d_d, *d_low, *d_high;

    lowStack[++top] = d_start;
    highStack[top] = d_end;

    cudaMalloc(&d_d, (d_end - d_start + 1) * sizeof(int));
    cudaMemcpy(d_d, d_array, (d_end - d_start + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_low, (d_end - d_start + 1) * sizeof(int));
    cudaMemcpy(d_low, lowStack, (d_end - d_start + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_high, (d_end - d_start + 1) * sizeof(int));
    cudaMemcpy(d_high, highStack, (d_end - d_start + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int numIterations = 1;

    while (numIterations > 0)
    {
        partition_shared<<<numBlocks, numThreads>>>(d_d, d_low, d_high, numIterations);
        int partitionSize;
        cudaMemcpyFromSymbol(&partitionSize, d_partitionSize, sizeof(int), 0, cudaMemcpyDeviceToHost);

        if (partitionSize < 1024)
        {
            numThreads = partitionSize;
        }
        else
        {
            numBlocks = partitionSize / numThreads + (partitionSize % numThreads == 0 ? 0 : 1);
        }
        numIterations = partitionSize;
        cudaMemcpy(d_array, d_d, (d_end - d_start + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    }
}
