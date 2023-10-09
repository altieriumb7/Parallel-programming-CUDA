#include "../lib/radix_sort.cuh"

// Manually implement population count (popc) function
__device__ unsigned int manual_popc(unsigned int x) {
    unsigned int count = 0;
    while (x) {
        count += (x & 1);
        x >>= 1;
    }
    return count;
}

__global__ void parallelRadix_shared(unsigned int* ddata) {
    __shared__ volatile unsigned int sdata[WSIZE * 2];

    // Load data from global memory into shared memory
    sdata[threadIdx.x] = ddata[threadIdx.x];

    unsigned int bitmask = 1 << LOWER_BIT;
    unsigned int offset = 0;
    unsigned int thrmask = 0xFFFFFFFFU << threadIdx.x;
    unsigned int mypos;

    for (int i = LOWER_BIT; i <= UPPER_BIT; i++) {
        unsigned int mydata = sdata[((WSIZE - 1) - threadIdx.x) + offset];
        unsigned int mybit = mydata & bitmask;

        // Get population of ones and zeroes manually
        unsigned int ones = 0;
        unsigned int zeroes = 0;

        for (int j = 0; j < WSIZE; j++) {
            unsigned int bit = (mydata >> j) & 1;
            ones += (bit == 1);
            zeroes += (bit == 0);
        }

        offset ^= WSIZE;

        // Determine my position in ping-pong buffer
        if (!mybit) {
            mypos = (zeroes & thrmask);
        } else { // Threads with a one bit
            mypos = ((zeroes + ones) & thrmask);
        }

        // Move data to buffer
        sdata[mypos - 1 + offset] = mydata;

        // Repeat for the next bit
        bitmask <<= 1;
    }

    // Put results back into global memory
    ddata[threadIdx.x] = sdata[threadIdx.x + offset];
}

__global__ void parallelRadix_glob(unsigned int* ddata) {
    unsigned int bitmask = 1 << LOWER_BIT;
    unsigned int offset = 0;
    unsigned int mypos;

    for (int i = LOWER_BIT; i <= UPPER_BIT; i++) {
        unsigned int mydata = ddata[threadIdx.x + offset];
        unsigned int mybit = mydata & bitmask;

        // Get population of ones and zeroes manually
        unsigned int ones = 0, zeroes = 0;
        for (int j = 0; j < WSIZE; j++) {
            unsigned int bit = (mydata >> j) & 1;
            ones += (bit & mybit);
            zeroes += (bit & ~mybit);
        }

        offset ^= WSIZE; // Switch ping-pong buffers

        // Do zeroes, then ones
        if (!mybit) {
            mypos = manual_popc(zeroes & ((1 << WSIZE) - 1));
        } else 
        { 
            mypos = manual_popc(zeroes) + manual_popc(ones & ((1 << WSIZE) - 1));
        }

        ddata[threadIdx.x + offset + mypos - 1] = mydata;

        // Repeat for the next bit
        bitmask <<= 1;
    }
}
