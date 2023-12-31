#include "../lib/merge_sort.cuh"

// GPU helper function for bottom-up merge
__device__ void gpuBottomUpMerge(int* src, int* dest, unsigned long long  start, unsigned long long  middle, unsigned long long  end) {
    unsigned long long  i = start;
    unsigned long long  j = middle;
    for (unsigned long long  k = start; k < end; k++) {
        if (i < middle && (j >= end || src[i] < src[j])) {
            dest[k] = src[i];
            i++;
        } else {
            dest[k] = src[j];
            j++;
        }
    }
}

__device__ unsigned long long getThreadIndex(dim3* threads, dim3* blocks) {
    unsigned long long threadIndex = threadIdx.x;
    unsigned long long threadMultiplier = threads->x;
    unsigned long long blockMultiplier = threadMultiplier * threads->y;

    return threadIndex +
           threadIdx.y * threadMultiplier +
           threadIdx.z * (threadMultiplier *= threads->y) +
           blockIdx.x  * (threadMultiplier *= threads->z) +
           blockIdx.y  * (threadMultiplier *= blocks->z) +
           blockIdx.z  * (threadMultiplier * blocks->y);
}

// GPU mergesort kernel
__global__ void gpuMergeSort(int* source, int* destination, unsigned long long size, unsigned long long  width, unsigned long long  slices, dim3* threads, dim3* blocks) {
    unsigned long long  idx = getThreadIndex(threads, blocks);
    unsigned long long  start = width * idx * slices;
    unsigned long long  middle, end;

    for (int slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = start + (width >> 1);
        if (middle > size)
            middle = size;

        end = start + width;
        if (end > size)
            end = size;

        gpuBottomUpMerge(source, destination, start, middle, end);
        start += width;
    }
}

// Mergesort function
void mergeSort_p(int* data, unsigned long long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {
    int* deviceData;
    int* deviceSwap;
    dim3* deviceThreads;
    dim3* deviceBlocks;

    // Allocate GPU memory
    cudaMalloc((void**)&deviceData, size * sizeof(int));
    cudaMalloc((void**)&deviceSwap, size * sizeof(int));
    cudaMalloc((void**)&deviceThreads, sizeof(dim3));
    cudaMalloc((void**)&deviceBlocks, sizeof(dim3));

    // Copy data to GPU
    cudaMemcpy(deviceData, data, size * sizeof(int), cudaMemcpyHostToDevice);

    // Copy thread and block information to GPU
    cudaMemcpy(deviceThreads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBlocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);

    int* A = deviceData;
    int* B = deviceSwap;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    for (long width = 2; width < (size << 1); width <<= 1) {
        long slices = size / (nThreads * width) + 1;

        gpuMergeSort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, deviceThreads, deviceBlocks);

        // Swap pointers A and B
        int* temp = A;
        A = B;
        B = temp;
    }

    // Copy sorted data back to CPU
    cudaMemcpy(data, A, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(deviceThreads);
    cudaFree(deviceBlocks);
}
