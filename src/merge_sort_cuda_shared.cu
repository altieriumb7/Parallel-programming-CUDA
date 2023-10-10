#include <cuda_runtime.h>
#include "../lib/constants.cuh"
#include "../lib/merge_sort.cuh"

// Define the size of shared memory buffer (adjust as needed)

// GPU helper function for bottom-up merge using shared memory
__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end, long* shared_mem) {
    long i = start;
    long j = middle;
    for (long k = start + threadIdx.x; k < end; k += blockDim.x) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            shared_mem[threadIdx.x] = source[i];
            i++;
        } else {
            shared_mem[threadIdx.x] = source[j];
            j++;
        }
        __syncthreads(); // Ensure all threads have written to shared memory

        // Copy data from shared memory back to destination array
        dest[k] = shared_mem[threadIdx.x];
    }
}

// GPU mergesort kernel
__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
    long start = width * idx * slices;
    long middle, end;

    // Define shared memory buffer
    __shared__ long shared_mem[SHARED_MEM_SIZE];

    for (long slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = start + (width >> 1);
        if (middle > size)
            middle = size;

        end = start + width;
        if (end > size)
            end = size;

        gpu_bottomUpMerge(source, dest, start, middle, end, shared_mem);
        start += width;
    }
}

// Mergesort function
void mergesort_shared(long* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {
    long* D_data;
    long* D_swp;
    dim3* D_threads;
    dim3* D_blocks;

    // Allocate GPU memory
    cudaMalloc((void**)&D_data, size * sizeof(long));
    cudaMalloc((void**)&D_swp, size * sizeof(long));
    cudaMalloc((void**)&D_threads, sizeof(dim3));
    cudaMalloc((void**)&D_blocks, sizeof(dim3));

    // Copy data to GPU
    cudaMemcpy(D_data, data, size * sizeof(long), cudaMemcpyHostToDevice);

    // Copy thread and block information to GPU
    cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);

    long* A = D_data;
    long* B = D_swp;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    for (long width = 2; width < (size << 1); width <<= 1) {
        long slices = size / (nThreads * width) + 1;

        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);

        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }

    // Copy sorted data back to CPU
    cudaMemcpy(data, A, size * sizeof(long), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(D_threads);
    cudaFree(D_blocks);
}