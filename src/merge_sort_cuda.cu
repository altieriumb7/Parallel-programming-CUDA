#include <cuda_runtime.h>
#include "../lib/constants.cuh"
#include "../lib/merge_sort.cuh"

// GPU helper function for bottom-up merge
__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}

// GPU helper function to calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}

// GPU mergesort kernel
__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
    long start = width * idx * slices;
    long middle, end;

    for (long slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = start + (width >> 1);
        if (middle > size)
            middle = size;

        end = start + width;
        if (end > size)
            end = size;

        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

// Mergesort function
void mergesort(long* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {
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

int isSorted(long* data, long size) {
    for (int i = 1; i < size; i++) {
        if (data[i - 1] > data[i]) {
            return 0; // Not sorted
        }
    }
    return 1; // Sorted
}

//--------------------------



// ...

// GPU helper function for bottom-up merge
__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end, long* shared_mem) {
    long i = start;
    long j = middle;
    long k = start + threadIdx.x;  // Compute the starting index for this thread in shared memory
    long limit = start + blockDim.x; // Calculate the limit for the loop

    while (k < end) {
        if (i < middle && (j >= end || source[i] <= source[j])) {
            shared_mem[threadIdx.x] = source[i];
            i++;
        } else {
            shared_mem[threadIdx.x] = source[j];
            j++;
        }
        k += blockDim.x; // Increment k by the number of threads in the block
        __syncthreads(); // Ensure all threads have written to shared memory

        // Copy data from shared memory back to destination array
        if (k < limit) {
            dest[k] = shared_mem[threadIdx.x];
        }
        __syncthreads(); // Ensure all threads have copied data back
    }
}

// GPU mergesort kernel
__global__ void gpu_mergesort_shared(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
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

        gpu_mergesort_shared<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);

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