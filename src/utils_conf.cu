#include "../lib/utils_conf.cuh"

// Function to determine the configuration based on N
Config determine_config(const unsigned long long N)
{
    ParallelSortConfig config;

    config.partition_size = PARTITION_SIZE;

    // Initialize with default values
    config.total_threads = min(N, MAXTHREADSPERBLOCK);
    config.total_blocks = 1;

    // Find the largest power of two <= total_threads
    for (unsigned long long i = config.total_threads; i >= 2; i--)
    {
        if (is_power_of_two(i))
        {
            // Adjust configuration based on power of two
            config.total_threads = i;
            config.partition_size = ceil(N / float(config.total_threads));
            config.threads_per_block = config.total_threads;
            break;
        }
    }

    // If total_threads is less than WARPSIZE, adjust the values
    if (config.total_threads < WARPSIZE)
    {
        config.threads_per_block = WARPSIZE;
        config.total_threads = WARPSIZE;
        config.total_blocks = 1;
        config.partition_size = ceil(N / float(config.total_threads));
    }

    // If N is greater than the starting partition size
    if (N > config.partition_size)
    {
        config.total_threads = ceil(N / float(config.partition_size));

        // If only one block is needed
        if (config.total_threads <= MAXTHREADSPERBLOCK)
        {
            config.total_blocks = 1;
            if (config.total_threads < WARPSIZE)
            {
                config.total_threads = WARPSIZE;
                config.threads_per_block = WARPSIZE;
            }
            else
            {
                config.threads_per_block = config.total_threads;
            }

            // Find the largest power of two <= total_threads
            for (unsigned long i = config.total_threads; i >= 2; i--)
            {
                if (is_power_of_two(i))
                {
                    // Adjust configuration based on power of two
                    config.total_threads = i;
                    config.partition_size = ceil(N / float(config.total_threads));
                    config.threads_per_block = config.total_threads;
                    break;
                }
            }
        }
        // If more than one block is needed
        else
        {
            config.threads_per_block = MAXTHREADSPERBLOCK;
            config.total_blocks = min(ceil(config.total_threads / float(config.threads_per_block)), MAXBLOCKS);
            config.total_threads = config.total_blocks * config.threads_per_block;

            // Find the largest power of two <= total_threads
            for (unsigned long i = config.total_threads; i >= 2; i--)
            {
                config.total_blocks = min(ceil(i / float(MAXTHREADSPERBLOCK)), MAXBLOCKS);
                config.total_threads = config.total_blocks * config.threads_per_block;

                if (is_power_of_two(config.total_threads))
                {
                    // Adjust configuration based on power of two
                    config.partition_size = ceil(N / float(config.total_threads));
                    break;
                }
            }
        }
    }

    // Calculate the required shared memory and maximum shared memory per block
    config.required_shared_memory = N * sizeof(unsigned short) / config.total_blocks;
    cudaDeviceGetAttribute(&config.max_shared_memory_per_block, cudaDevAttrMaxSharedMemoryPerBlock, 0);

    return config;
}
