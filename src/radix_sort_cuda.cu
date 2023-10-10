
#include "../lib/radix_sort.cuh"




__global__ void radix_sort(unsigned int *values)
{
    int  bit;
    for( bit = 0; bit < 32; ++bit )
    {
        partition_by_bit(values, bit);
        __syncthreads();
    }
}

__device__ int plus_scan(unsigned int *x)
{
    unsigned int i = threadIdx.x; // id of thread executing this instance
    unsigned int n = blockDim.x;  // total number of threads in this block
    unsigned int offset;          // distance between elements to be added

    for( offset = 1; offset < n; offset *= 2) {
        unsigned int t;

        if ( i >= offset ) 
            t = x[i-offset];
        
        __syncthreads();

        if ( i >= offset ) 
            x[i] = t + x[i];      // i.e., x[i] = x[i] + x[i-1]

        __syncthreads();
    }
    return x[i];
}
__device__ void partition_by_bit(unsigned int *values, unsigned int bit)
{
    unsigned int i = threadIdx.x;
    unsigned int size = blockDim.x;
    unsigned int x_i = values[i];          // value of integer at position i
    unsigned int p_i = (x_i >> bit) & 1;   // value of bit at position bit
    values[i] = p_i;  
    __syncthreads();
    unsigned int T_before = plus_scan(values);
    unsigned int T_total  = values[size-1];
    unsigned int F_total  = size - T_total;
    __syncthreads();
    if ( p_i )
        values[T_before-1 + F_total] = x_i;
    else
        values[i - T_before] = x_i;
}

//-------------------------------------

__global__ void radix_sort_shared(unsigned int *values)
{
    int bit;
    unsigned int i = threadIdx.x;
    unsigned int size = blockDim.x;
    extern __shared__ unsigned int shared_values[];

    // Load data into shared memory
    shared_values[i] = values[i];
    __syncthreads();

    // Perform radix sort on each bit position
    for (bit = 0; bit < 32; ++bit)
    {
        unsigned int p_i = (shared_values[i] >> bit) & 1;

        // Perform a parallel prefix sum (scan) on the shared memory array
        for (unsigned int offset = 1; offset < size; offset *= 2)
        {
            unsigned int t;

            if (i >= offset)
                t = shared_values[i - offset];

            __syncthreads();

            if (i >= offset)
                shared_values[i] += t;

            __syncthreads();
        }

        // Calculate the total number of 1s (T_total) and 0s (F_total)
        unsigned int T_total = shared_values[size - 1];
        unsigned int F_total = size - T_total;

        // Calculate the position for the current element in the sorted order
        unsigned int pos = p_i ? T_total - 1 + F_total : i - T_total;

        // Use shared memory for a more efficient swap
        unsigned int temp = shared_values[i];
        __syncthreads();

        // Update the values array based on the partitioning
        shared_values[pos] = temp;
        __syncthreads();
    }

    // Store the sorted data back to global memory
    values[i] = shared_values[i];
}