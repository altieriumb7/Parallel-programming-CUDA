#include "mergesort.cuh"
#include <cuda_runtime.h>
#include "../lib/constants.cuh"
#include "../lib/merge_sort.cuh"


void mergesort(float* data, dim3 threadsPerBlock, dim3 blocksPerGrid) 
{
    float* D_data;
    float* D_swp;
    dim3* D_threads;
    dim3* D_blocks;

    cudaMalloc((void**)&D_data, size * sizeof(float));
    cudaMalloc((void**)&D_swp, size * sizeof(float));

    cudaMemcpy(D_data, data, size * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&D_threads, sizeof(dim3));
    cudaMalloc((void**)&D_blocks, sizeof(dim3));

    cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);

    float* A = D_data;
    float* B = D_swp;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    for (int width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads) * width) + 1;

        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, width, slices, D_threads, D_blocks);

        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }

    cudaMemcpy(data, A, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A);
    cudaFree(B);
}

__device__ unsigned int getIdx(dim3* threads, dim3* blocks) 
{
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}

__global__ void gpu_mergesort(float* source, float* dest, long width, long slices, dim3* threads, dim3* blocks) 
{
    unsigned int idx = getIdx(threads, blocks);
    long start = width * idx * slices, middle, end;

    for (long slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

__device__ void gpu_bottomUpMerge(float* source, float* dest, long start, long middle, long end) 
{
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
