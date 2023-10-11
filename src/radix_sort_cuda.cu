#include "../lib/radix_sort.cuh"

// Kernel for performing radix sort
__global__ void radixSort(unsigned int *values)
{
    int bit;
    // Iterate through each bit (0 to 31)
    for (bit = 0; bit < 32; ++bit)
    {
        // Perform the partition by the current bit
        partitionByBit(values, bit);
        __syncthreads();
    }
}

// Device function for inclusive scan (prefix sum)
__device__ int inclusiveScan(unsigned int *x)
{
    unsigned int i = threadIdx.x;
    unsigned int n = blockDim.x;
    unsigned int offset;
    
    // Perform a parallel reduction to compute the inclusive scan
    for (offset = 1; offset < n; offset *= 2)
    {
        unsigned int t;
        if (i >= offset) 
            t = x[i - offset];
        
        __syncthreads();
        
        if (i >= offset) 
            x[i] += t;
        
        __syncthreads();
    }
    return x[i];
}

// Device function to partition values based on a specific bit
__device__ void partitionByBit(unsigned int *values, unsigned int bit)
{
    unsigned int i = threadIdx.x;
    unsigned int size = blockDim.x;
    unsigned int x_i = values[i];
    unsigned int p_i = (x_i >> bit) & 1; // Extract the bit at the given position
    values[i] = p_i;
    __syncthreads();
    
    // Perform inclusive scan on the values array
    unsigned int T_before = inclusiveScan(values);
    unsigned int T_total = values[size - 1];
    unsigned int F_total = size - T_total;
    __syncthreads();
    
    // Rearrange values based on the bit
    if (p_i)
        values[T_before - 1 + F_total] = x_i;
    else
        values[i - T_before] = x_i;
}

// Kernel for performing radix sort using shared memory
__global__ void radixSortShared(unsigned int *values)
{
    int bit;
    __shared__ unsigned int sharedValues[SHARED_MEM_SIZE];
    unsigned int* sValues = sharedValues;

    // Iterate through each bit (0 to 31)
    for (bit = 0; bit < 32; ++bit)
    {
        // Perform the partition using shared memory
        partitionByBitShared(sValues, bit);
        __syncthreads();

        // Copy the results back to global memory
        values[blockIdx.x * blockDim.x + threadIdx.x] = sValues[threadIdx.x];
        __syncthreads();
    }
}

// Device function for inclusive scan (prefix sum) using shared memory
__device__ int inclusiveScanShared(unsigned int *x)
{
    unsigned int i = threadIdx.x;
    unsigned int n = blockDim.x;
    unsigned int offset;

    // Perform a parallel reduction to compute the inclusive scan using shared memory
    for (offset = 1; offset < n; offset *= 2)
    {
        unsigned int t;
        if (i >= offset)
            t = x[i - offset];

        __syncthreads();

        if (i >= offset)
            x[i] += t;
        
        __syncthreads();
    }
    return x[i];
}

// Device function to partition values based on a specific bit using shared memory
__device__ void partitionByBitShared(unsigned int *values, unsigned int bit)
{
    unsigned int i = threadIdx.x;
    unsigned int size = blockDim.x;
    unsigned int x_i = values[i];
    unsigned int p_i = (x_i >> bit) & 1; // Extract the bit at the given position
    values[i] = p_i;
    __syncthreads();
    
    // Perform inclusive scan on the values array using shared memory
    unsigned int T_before = inclusiveScanShared(values);
    unsigned int T_total = values[size - 1];
    unsigned int F_total = size - T_total;
    __syncthreads();
    
    // Rearrange values based on the bit
    if (p_i)
        values[T_before - 1 + F_total] = x_i;
    else
        values[i - T_before] = x_i;
}
