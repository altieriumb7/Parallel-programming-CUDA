
#include "../lib/merge_sort.cuh"


void mergesort(unsigned short * data, dim3 threadsPerBlock, dim3 blocksPerGrid) 
{
    unsigned short *D_data;
    unsigned short *D_swp;
    dim3 *D_threads;
    dim3 *D_blocks;
    
    checkCudaErrors(cudaMalloc((void**) &D_data, size * sizeof(long)));
    checkCudaErrors(cudaMalloc((void**) &D_swp, size * sizeof(long)));

    checkCudaErrors(cudaMemcpy(D_data, data, size * sizeof(long), cudaMemcpyHostToDevice));
 
    checkCudaErrors(cudaMalloc((void**) &D_threads, sizeof(dim3)));
    checkCudaErrors(cudaMalloc((void**) &D_blocks, sizeof(dim3)));

    checkCudaErrors(cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice));

    unsigned short *A = D_data;
    unsigned short *B = D_swp;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    for (int width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads) * width) + 1;

        

        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, width, slices, D_threads, D_blocks);

        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }

    checkCudaErrors(cudaMemcpy(data, A, size * sizeof(long), cudaMemcpyDeviceToHost));
    
    checkCudaErrors(cudaFree(A));
    checkCudaErrors(cudaFree(B));
}

__device__ unsigned int getIdx(dim3 *threads, dim3 *blocks) 
{
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}


__global__ void gpu_mergesort(unsigned short *source, unsigned short *dest,unsigned long long width, unsigned long long slices, dim3 *threads, dim3 *blocks) 
{
    unsigned int idx = getIdx(threads, blocks);
    unsigned long long start = width*idx*slices, 
         middle, 
         end;

    for (long slice = 0; slice < slices; slice++) 
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
    for (long k = start; k < end; k++) {
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