
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




__device__ void partition_by_bit_shared(unsigned int *values, unsigned int bit, unsigned int *shared_mem,const unsigned long long N)
{
    unsigned int i = threadIdx.x;
    unsigned int size = blockDim.x;
    unsigned int x_i = values[i];          // Value of integer at position i
    unsigned int p_i = (x_i >> bit) & 1;   // Value of bit at position bit

    // Store the value of p_i in shared memory
    shared_mem[i] = p_i;
    
    __syncthreads();

    // Perform a parallel prefix sum (scan) on the shared memory array
    for (unsigned int offset = 1; offset < size; offset *= 2) {
        unsigned int t;
        if (i >= offset) {
            t = shared_mem[i - offset];
        }
        __syncthreads();
        if (i >= offset) {
            shared_mem[i] += t;
        }
        __syncthreads();
    }

    // Calculate the total number of 1s (T_total) and 0s (F_total)
    unsigned int T_total = shared_mem[size - 1];
    unsigned int F_total = size - T_total;

    // Calculate the position for the current element in the sorted order
    unsigned int pos = p_i ? T_total - 1 + F_total : i - T_total;

    // Use shared memory for a more efficient swap
    __shared__ unsigned int shared_values[N]; // Define an appropriate size here

    // Ensure all threads have finished their scan before proceeding
    __syncthreads();

    // Copy values to shared memory
    shared_values[shared_mem[i]] = x_i;
    
    // Ensure all threads have finished copying
    __syncthreads();

    // Copy values back to the original array
    values[i] = shared_values[pos];
}

__global__ void radix_sort_shared(unsigned int *values,const unsigned long long N)
{
    int bit;
    for (bit = 0; bit < 32; ++bit)
    {
        // Allocate shared memory for each block
        __shared__ unsigned int shared_mem[N]; // Define an appropriate size here

        partition_by_bit_shared(values, bit, shared_mem,N);
        __syncthreads();
    }
}

// The rest of your code remains the same...


__device__ int plus_scan(unsigned int *x, unsigned int *shared_mem)
{
    unsigned int i = threadIdx.x;
    unsigned int n = blockDim.x;
    unsigned int offset;

    for (offset = 1; offset < n; offset *= 2)
    {
        unsigned int t;

        if (i >= offset)
            t = x[i - offset];

        __syncthreads();

        if (i >= offset)
            x[i] = t + x[i];

        __syncthreads();
    }

    if (shared_mem)
    {
        // Copy the inclusive scan results to shared memory
        shared_mem[i] = x[i];
        __syncthreads();

        // Perform a block-level exclusive scan on shared memory
        for (offset = 1; offset < n; offset *= 2)
        {
            unsigned int t;

            if (i >= offset)
                t = shared_mem[i - offset];

            __syncthreads();

            if (i >= offset)
                shared_mem[i] = t + shared_mem[i];

            __syncthreads();
        }

        // Calculate the total sum from the last element of shared memory
        int total_sum = shared_mem[n - 1];

        // Add the total sum to each element in the block
        x[i] += total_sum;
    }

    return x[i];
}


