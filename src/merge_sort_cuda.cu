
#include "../lib/merge_sort.cuh"

// GPU helper function for bottom-up merge
__device__ void gpuBottomUpMerge(unsigned int* src, unsigned int* dest, unsigned int start, unsigned int middle, unsigned int end) {
    unsigned int i = start;
    unsigned int j = middle;
    for (unsigned int k = start; k < end; k++) {
        if (i < middle && (j >= end || src[i] < src[j])) {
            dest[k] = src[i];
            i++;
        } else {
            dest[k] = src[j];
            j++;
        }
    }
}

__device__ unsigned int getThreadIndex(dim3* threads, dim3* blocks) {
    int threadIndex = threadIdx.x;
    int threadMultiplier = threads->x;
    int blockMultiplier = threadMultiplier * threads->y;

    return threadIndex +
           threadIdx.y * threadMultiplier +
           threadIdx.z * (threadMultiplier *= threads->y) +
           blockIdx.x  * (threadMultiplier *= threads->z) +
           blockIdx.y  * (threadMultiplier *= blocks->z) +
           blockIdx.z  * (threadMultiplier * blocks->y);
}

// GPU mergesort kernel
__global__ void gpuMergeSort(unsigned int* source, unsigned int* destination, unsigned long long size, unsigned int width, unsigned int slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getThreadIndex(threads, blocks);
    unsigned int start = width * idx * slices;
    unsigned int middle, end;

    for (unsigned int slice = 0; slice < slices; slice++) {
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
void mergeSort(unsigned int* data, unsigned long long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {
    unsigned int* deviceData;
    unsigned int* deviceSwap;
    dim3* deviceThreads;
    dim3* deviceBlocks;

    // Allocate GPU memory
    cudaMalloc((void**)&deviceData, size * sizeof(unsigned int));
    cudaMalloc((void**)&deviceSwap, size * sizeof(unsigned int));
    cudaMalloc((void**)&deviceThreads, sizeof(dim3));
    cudaMalloc((void**)&deviceBlocks, sizeof(dim3));

    // Copy data to GPU
    cudaMemcpy(deviceData, data, size * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Copy thread and block information to GPU
    cudaMemcpy(deviceThreads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBlocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);

    unsigned int* A = deviceData;
    unsigned int* B = deviceSwap;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    for (long width = 2; width < (size << 1); width <<= 1) {
        long slices = size / (nThreads * width) + 1;

        gpuMergeSort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, deviceThreads, deviceBlocks);

        // Swap pointers A and B
        unsigned int* temp = A;
        A = B;
        B = temp;
    }

    // Copy sorted data back to CPU
    cudaMemcpy(data, A, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(deviceThreads);
    cudaFree(deviceBlocks);
}
// GPU helper function for bottom-up merge
__device__ void gpuBottomUpMerge(unsigned int* source, unsigned int* dest, unsigned int start, unsigned int middle, unsigned int end, unsigned int* sharedMem) {
    unsigned int i = start;
    unsigned int j = middle;
    unsigned int k = start + threadIdx.x;  // Calculate the starting index for this thread in shared memory
    unsigned int limit = start + blockDim.x; // Calculate the limit for the loop

    while (k < end) {
        if (i < middle && (j >= end || source[i] <= source[j])) {
            sharedMem[threadIdx.x] = source[i];
            i++;
        } else {
            sharedMem[threadIdx.x] = source[j];
            j++;
        }
        k += blockDim.x; // Increment k by the number of threads in the block
        __syncthreads(); // Ensure all threads have written to shared memory

        // Copy data from shared memory back to the destination array
        if (k < limit) {
            dest[k] = sharedMem[threadIdx.x];
        }
        __syncthreads(); // Ensure all threads have copied data back
    }
}

// GPU mergesort kernel
__global__ void gpuMergeSortShared(unsigned int* source, unsigned int* dest, unsigned long long size, unsigned int width, unsigned int slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getThreadIndex(threads, blocks);
    unsigned int start = width * idx * slices;
    unsigned int middle, end;

    // Define shared memory buffer
    __shared__ unsigned int sharedMem[SHARED_MEM_SIZE];

    for (unsigned int slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = start + (width >> 1);
        if (middle > size)
            middle = size;

        end = start + width;
        if (end > size)
            end = size;

        gpuBottomUpMerge(source, dest, start, middle, end, sharedMem);
        start += width;
    }
}

// Mergesort function
void mergeSortShared(unsigned int* data, unsigned long long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {
    unsigned int* deviceData;
    unsigned int* deviceSwap;
    dim3* deviceThreads;
    dim3* deviceBlocks;

    // Allocate GPU memory
    cudaMalloc((void**)&deviceData, size * sizeof(unsigned int));
    cudaMalloc((void**)&deviceSwap, size * sizeof(unsigned int));
    cudaMalloc((void**)&deviceThreads, sizeof(dim3));
    cudaMalloc((void**)&deviceBlocks, sizeof(dim3));

    // Copy data to GPU
    cudaMemcpy(deviceData, data, size * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Copy thread and block information to GPU
    cudaMemcpy(deviceThreads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBlocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);

    unsigned int* A = deviceData;
    unsigned int* B = deviceSwap;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    for (long width = 2; width < (size << 1); width <<= 1) {
        long slices = size / (nThreads * width) + 1;

        gpuMergeSortShared<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, deviceThreads, deviceBlocks);

        // Swap pointers A and B
        unsigned int* temp = A;
        A = B;
        B = temp;
    }

    // Copy sorted data back to CPU
    cudaMemcpy(data, A, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(deviceThreads);
    cudaFree(deviceBlocks);
}
