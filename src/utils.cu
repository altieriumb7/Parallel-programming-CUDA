#include "../lib/utils.cuh"

// Function to get the current time
double time_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    // Return the time in seconds with nanosecond precision
    return (ts.tv_sec + (double)ts.tv_nsec / 1e9);
}
void zero_array(unsigned short *data, const unsigned long long N){
    for (unsigned long long i = 0; i < N; i++) {
        data[i] = 0;
    }
}

void print_array(unsigned short* array, unsigned long long size) {
    for (unsigned long long i = 0; i < size; i++) {
        printf("%u ", array[i]); // Assuming 'unsigned short' elements
    }
    printf("\n");
}

// Function to initialize an array with random values within a given range
void fill_array(unsigned short *data, const unsigned long long N)
{
    srand(42); // Initialize the random number generator with a seed for determinism

    for (unsigned long long i = 0; i < N; i++)
    {
        // Generate a random value within the specified range and store it in the array
        data[i] = rand() % (MAX_VALUE - MIN_VALUE + 1) + MIN_VALUE;
    }
}

// Function to check if an array is sorted in non-decreasing order
bool is_sorted(unsigned short* result, const unsigned long long N)
{
    for (unsigned short i = 0; i < N - 1; i++)
    {
        if (result[i] > result[i + 1])
        {
            // If any element is greater than the next element, the array is not sorted
            return false;
        }
    }
    // If the loop completes without finding out-of-order elements, the array is sorted
    return true;
}

// Function to check if a number is a power of two
bool is_power_of_two(const unsigned long x)
{
    // Check if the number is a power of two by examining its binary representation
    // A power of two has only one bit set, and when you subtract 1, all lower bits become set
    return (x & (x - 1)) == 0;
}
