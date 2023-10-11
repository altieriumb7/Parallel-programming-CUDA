// Device function to merge two sorted arrays
__device__ void gpu_bottomUpMerge(int* source, int* dest, int start, int middle, int end) {
    int i = start;
    int j = middle;
    for (int k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}

// Device function to calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x = threads->x * threads->y * threads->z;
    return threadIdx.x +
           threadIdx.y * x +
           blockIdx.x * x * threads->z +
           blockIdx.y * x * threads->z * blocks->z +
           blockIdx.z * x * threads->z * blocks->z * blocks->y;
}

// GPU kernel for mergesort
__global__ void gpu_mergesort(int* source, int* dest, int size, int width, int slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
    int start = width * idx * slices;
    int middle, end;

    for (int slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = start + (width >> 1);
        middle = min(middle, size);

        end = start + width;
        end = min(end, size);

        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

// Mergesort function
void mergesort(int* data, int size, dim3 threadsPerBlock, dim3 blocksPerGrid) {
    int* D_data;
    int* D_swp;
    dim3* D_threads;
    dim3* D_blocks;

    // Allocate GPU memory
    cudaMalloc((void**)&D_data, size * sizeof(int));
    cudaMalloc((void**)&D_swp, size * sizeof(int));
    cudaMalloc((void**)&D_threads, sizeof(dim3));
    cudaMalloc((void**)&D_blocks, sizeof(dim3));

    // Copy data to GPU
    cudaMemcpy(D_data, data, size * sizeof(int), cudaMemcpyHostToDevice);

    // Copy thread and block information to GPU
    cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);

    int* A = D_data;
    int* B = D_swp;

    int nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                   blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    for (int width = 2; width < (size << 1); width <<= 1) {
        int slices = (size + (nThreads * width) - 1) / (nThreads * width); // Calculate the number of slices

        // Launch the GPU kernel
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);

        // Swap source and destination arrays
        int* temp = A;
        A = B;
        B = temp;
    }

    // Copy sorted data back to CPU
    cudaMemcpy(data, A, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(D_data);
    cudaFree(D_swp);
    cudaFree(D_threads);
    cudaFree(D_blocks);
}
