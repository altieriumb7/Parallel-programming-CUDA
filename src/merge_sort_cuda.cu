#include "../lib/merge_sort.cuh"

void mergesort(unsigned short *data, dim3 threadsPerBlock, dim3 blocksPerGrid, unsigned long long size)
{
    unsigned short *D_data;
    unsigned short *D_swp;
    dim3 D_threads;
    dim3 D_blocks;

    cudaMalloc((void **)&D_data, size * sizeof(unsigned short));
    cudaMalloc((void **)&D_swp, size * sizeof(unsigned short));

    cudaMemcpy(D_data, data, size * sizeof(unsigned short), cudaMemcpyHostToDevice);

    D_threads = threadsPerBlock;
    D_blocks = blocksPerGrid;

    unsigned short *A = D_data;
    unsigned short *B = D_swp;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    for (unsigned long long width = 2; width < (size << 1); width <<= 1)
    {
        unsigned long long slices = size / (nThreads * width) + 1;

        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, width, slices, D_threads, D_blocks, size);

        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }

    cudaMemcpy(data, A, size * sizeof(unsigned short), cudaMemcpyDeviceToHost);

    cudaFree(D_data);
    cudaFree(D_swp);
}

__device__ unsigned int getIdx(dim3 threads, dim3 blocks)
{
    int x;
    return threadIdx.x +
           threadIdx.y * (x = threads.x) +
           threadIdx.z * (x *= threads.y) +
           blockIdx.x * (x *= threads.z) +
           blockIdx.y * (x *= blocks.z) +
           blockIdx.z * (x *= blocks.y);
}

__global__ void gpu_mergesort(unsigned short *source, unsigned short *dest, unsigned long long width, unsigned long long slices, dim3 threads, dim3 blocks, unsigned long long size)
{
    unsigned int idx = getIdx(threads, blocks);
    unsigned long long start = width * idx * slices,
                       middle,
                       end;

    for (unsigned long long slice = 0; slice < slices; slice++)
    {
        if (start >= size)
            break;

        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

__device__ void gpu_bottomUpMerge(unsigned short *source, unsigned short *dest, unsigned long long start, unsigned long long middle, unsigned long long end)
{
    unsigned long i = start;
    unsigned long j = middle;
    for (unsigned long long k = start; k < end; k++)
    {
        if (i < middle && (j >= end || source[i] < source[j]))
        {
            dest[k] = source[i];
            i++;
        }
        else
        {
            dest[k] = source[j];
            j++;
        }
    }
}
