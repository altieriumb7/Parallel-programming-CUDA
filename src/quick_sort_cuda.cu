#include "../lib/quick_sort.cuh"

// Device variable to store the partition size
__device__ unsigned short d_partitionSize;

// GPU kernel for partitioning data
__global__ void partition(unsigned short *d_data, unsigned short *d_low, unsigned short *d_high, unsigned short d_size)
{
    unsigned short idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_partitionSize = 0;  // Initialize partition size to 0
    __syncthreads();

    if (idx < d_size)
    {
        unsigned short high = d_high[idx];
        unsigned short low = d_low[idx];
        unsigned short pivot = d_data[high];
        unsigned short i = (low - 1);
        unsigned short temp;

        for (unsigned short j = low; j <= high - 1; j++)
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
        unsigned short partitionPoint = (i + 1);

        if (partitionPoint - 1 > low)
        {
            unsigned short ind = atomicAdd(&d_partitionSize, 1);
            d_low[ind] = low;
            d_high[ind] = partitionPoint - 1;
        }
        if (partitionPoint + 1 < high)
        {
            unsigned short ind = atomicAdd(&d_partitionSize, 1);
            d_low[ind] = partitionPoint + 1;
            d_high[ind] = high;
        }
    }
}

// Host function for iterative quicksort on the GPU
void quick_sort_p(unsigned short d_array[], unsigned short d_start, unsigned short d_end,unsigned short numBlocks, unsigned short numThreads)
{
    unsigned short lowStack[d_end - d_start + 1], highStack[d_end - d_start + 1];

    int top = -1;
    unsigned short *d_d, *d_low, *d_high;

    lowStack[++top] = d_start;
    highStack[top] = d_end;

    cudaMalloc(&d_d, (d_end - d_start + 1) * sizeof(unsigned short));
    cudaMemcpy(d_d, d_array, (d_end - d_start + 1) * sizeof(unsigned short), cudaMemcpyHostToDevice);

    cudaMalloc(&d_low, (d_end - d_start + 1) * sizeof(unsigned short));
    cudaMemcpy(d_low, lowStack, (d_end - d_start + 1) * sizeof(unsigned short), cudaMemcpyHostToDevice);

    cudaMalloc(&d_high, (d_end - d_start + 1) * sizeof(unsigned short));
    cudaMemcpy(d_high, highStack, (d_end - d_start + 1) * sizeof(unsigned short), cudaMemcpyHostToDevice);

    numThreads = 1;
    numBlocks = 1;
    unsigned short numIterations = 1;

    while (numIterations > 0)
    {
        partition<<<numBlocks, numThreads>>>(d_d, d_low, d_high, numIterations);
        unsigned short partitionSize;
        cudaMemcpyFromSymbol(&partitionSize, d_partitionSize, sizeof(unsigned short), 0, cudaMemcpyDeviceToHost);

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
        cudaMemcpy(d_array, d_d, (d_end - d_start + 1) * sizeof(unsigned short), cudaMemcpyDeviceToHost);
    }
}



// Shared memory for the partition
__shared__ unsigned short sharedArr[SHARED_MEM_SIZE];

// GPU kernel for partitioning data using shared memory
__global__ void partition_shared(unsigned short *d_array, unsigned short *d_low, unsigned short *d_high, unsigned short d_size)
{
    unsigned short idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_partitionSize = 0;
    __syncthreads();

    if (idx < d_size)
    {
        // Load data into shared memory
        sharedArr[threadIdx.x] = d_array[idx];
        __syncthreads();

        unsigned short high = d_high[idx];
        unsigned short low = d_low[idx];
        unsigned short pivot = sharedArr[threadIdx.x]; // Use shared memory
        unsigned short i = (low - 1);
        unsigned short temp;

        for (unsigned short j = low; j <= high - 1; j++)
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
        unsigned short partitionPoint = (i + 1);

        if (partitionPoint - 1 > low)
        {
            unsigned short ind = atomicAdd(&d_partitionSize, 1);
            d_low[ind] = low;
            d_high[ind] = partitionPoint - 1;
        }
        if (partitionPoint + 1 < high)
        {
            unsigned short ind = atomicAdd(&d_partitionSize, 1);
            d_low[ind] = partitionPoint + 1;
            d_high[ind] = high;
        }

        // Store the results back to global memory
        d_array[idx] = sharedArr[threadIdx.x];
    }
}

// Host function for iterative quicksort on the GPU using shared memory
void quick_sort_p_shared(unsigned short d_array[], unsigned short d_start, unsigned short d_end, unsigned short numBlocks, unsigned short numThreads)
{
    unsigned short lowStack[d_end - d_start + 1], highStack[d_end - d_start + 1];

    int top = -1;
    unsigned short *d_d, *d_low, *d_high;

    lowStack[++top] = d_start;
    highStack[top] = d_end;

    cudaMalloc(&d_d, (d_end - d_start + 1) * sizeof(unsigned short));
    cudaMemcpy(d_d, d_array, (d_end - d_start + 1) * sizeof(unsigned short), cudaMemcpyHostToDevice);

    cudaMalloc(&d_low, (d_end - d_start + 1) * sizeof(unsigned short));
    cudaMemcpy(d_low, lowStack, (d_end - d_start + 1) * sizeof(unsigned short), cudaMemcpyHostToDevice);

    cudaMalloc(&d_high, (d_end - d_start + 1) * sizeof(unsigned short));
    cudaMemcpy(d_high, highStack, (d_end - d_start + 1) * sizeof(unsigned short), cudaMemcpyHostToDevice);

    numThreads = 1;
    numBlocks = 1;
    unsigned short numIterations = 1;

    while (numIterations > 0)
    {
        partition_shared<<<numBlocks, numThreads>>>(d_d, d_low, d_high, numIterations);
        unsigned short partitionSize;
        cudaMemcpyFromSymbol(&partitionSize, d_partitionSize, sizeof(unsigned short), 0, cudaMemcpyDeviceToHost);

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
        cudaMemcpy(d_array, d_d, (d_end - d_start + 1) * sizeof(unsigned short), cudaMemcpyDeviceToHost);
    }
}
