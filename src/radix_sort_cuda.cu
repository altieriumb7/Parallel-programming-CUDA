
__device__ unsigned int ddata[WSIZE];

__global__ void parallelRadix() {
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

        // Get population of ones and zeroes using ballot()
        unsigned int ones = __ballot(mybit);
        unsigned int zeroes = ~ones;
        offset ^= WSIZE;

        // Determine my position in ping-pong buffer
        if (!mybit) {
            mypos = __popc(zeroes & thrmask);
        } else { // Threads with a one bit
            mypos = __popc(zeroes) + __popc(ones & thrmask);
        }

        // Move data to buffer
        sdata[mypos - 1 + offset] = mydata;

        // Repeat for next bit
        bitmask <<= 1;
    }

    // Put results back into global memory
    ddata[threadIdx.x] = sdata[threadIdx.x + offset];
}