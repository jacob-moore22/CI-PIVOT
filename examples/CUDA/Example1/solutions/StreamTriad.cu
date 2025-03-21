/**
 * Stream Triad Benchmark Implementation using CUDA
 * This example demonstrates:
 * 1. Basic CUDA kernel definition and launch
 * 2. Memory management (host and device)
 * 3. Data transfer between host and device
 * 4. Thread indexing and block organization
 * 5. Error handling in CUDA
 * 6. Performance measurement
 */

// CUDA kernel version of Stream Triad
// __global__ indicates that this is a kernel function that will be executed on the device
// This is a special CUDA keyword that tells the compiler this function will be called from the host
// and executed on the GPU device
__global__ void StreamTriad(
                const int n,           // Total number of elements to process
                const double scalar,   // Scalar multiplier for the triad operation
                const double *a,       // Input array a
                const double *b,       // Input array b
                      double *c)       // Output array c
{
    // Get the global thread index to determine which element to process
    // blockIdx.x: Current block index in the grid
    // blockDim.x: Number of threads per block
    // threadIdx.x: Current thread index within the block
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    // Protect from going out-of-bounds by checking if i is greater than or equal to n
    // This is important because the grid size might be padded to ensure even block sizes
    if (i >= n) return;

    // Perform the triad operation: c[i] = a[i] + scalar*b[i]
    // This is the core computation that each thread performs
    c[i] = a[i] + scalar*b[i];
}

#include <stdio.h>
#include <sys/time.h>
#include "timer.h"

// Number of times to run the benchmark for averaging
#define NTIMES 16

int main(int argc, char *argv[]){
    // Variables for timing measurements
    struct timespec tkernel, ttotal;  // Structure to hold timing information
    // initializing data and arrays
    int stream_array_size = 80000000;  // Size of arrays (80 million elements)
    double scalar = 3.0, tkernel_sum = 0.0, ttotal_sum = 0.0;  // Scalar value and timing accumulators

    // allocate host memory and initialize
    // Host memory is the CPU's main memory
    double *a = (double *)malloc(stream_array_size*sizeof(double));
    double *b = (double *)malloc(stream_array_size*sizeof(double));
    double *c = (double *)malloc(stream_array_size*sizeof(double));

    // Initialize arrays with values
    // This is done on the CPU before transferring to GPU
    for (int i=0; i<stream_array_size; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;  // Initialize c array to 0
    }

    // allocate device memory
    // Device memory is the GPU's memory
    double *a_d, *b_d, *c_d;  // Device pointers (d suffix for device)
    cudaError_t err;  // Variable to store CUDA error codes
    
    // Allocate memory on the GPU for each array
    // cudaMalloc allocates memory on the GPU device
    err = cudaMalloc(&a_d, stream_array_size*sizeof(double));
    if (err != cudaSuccess) {
        printf("Error allocating a_d: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaMalloc(&b_d, stream_array_size*sizeof(double));
    if (err != cudaSuccess) {
        printf("Error allocating b_d: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaMalloc(&c_d, stream_array_size*sizeof(double));
    if (err != cudaSuccess) {
        printf("Error allocating c_d: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // setting block size and padding total grid size to get even block sizes
    // Block size is the number of threads per block (must be a multiple of 32)
    int blocksize = 512;  // Common block size for modern GPUs
    // Calculate grid size to cover all elements, rounding up to ensure full coverage
    int gridsize = (stream_array_size + blocksize - 1)/blocksize;

    // Main benchmark loop
    for (int k = 0; k < NTIMES; k++){
        cpu_timer_start(&ttotal);  // Start total time measurement
        
        // copying array data from host to device
        // cudaMemcpyHostToDevice indicates transfer direction
        err = cudaMemcpy(a_d, a, stream_array_size*sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Error copying a to device: %s\n", cudaGetErrorString(err));
            return -1;
        }
        
        err = cudaMemcpy(b_d, b, stream_array_size*sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Error copying b to device: %s\n", cudaGetErrorString(err));
            return -1;
        }
        
        // Synchronize to ensure data transfer is complete
        // This ensures all data is on the GPU before kernel launch
        cudaDeviceSynchronize();

        cpu_timer_start(&tkernel);  // Start kernel time measurement
        
        // launch stream triad kernel
        // <<<gridsize, blocksize>>> is CUDA's special syntax for kernel launch configuration
        // This determines how many blocks and threads per block to use
        StreamTriad<<<gridsize, blocksize>>>(stream_array_size, scalar, a_d, b_d, c_d);
        
        // Check for kernel launch errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error launching kernel: %s\n", cudaGetErrorString(err));
            return -1;
        }
        
        // Synchronize to ensure kernel completion
        // This ensures the kernel has finished before we try to copy results back
        cudaDeviceSynchronize();
        tkernel_sum += cpu_timer_stop(tkernel);  // Record kernel execution time

        // Copy results back from device to host
        // cudaMemcpyDeviceToHost indicates transfer direction
        // This operation is blocking, so no need for additional synchronization
        err = cudaMemcpy(c, c_d, stream_array_size*sizeof(double), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("Error copying c from device: %s\n", cudaGetErrorString(err));
            return -1;
        }
        
        ttotal_sum += cpu_timer_stop(ttotal);  // Record total time
        
        // check results and print errors if found. limit to only 10 errors per iteration
        // This verifies the computation was correct
        for (int i=0, icount=0; i<stream_array_size && icount < 10; i++){
            if (fabs(c[i] - (1.0 + 3.0*2.0)) > 1e-10) {  // Use small epsilon for floating point comparison
                printf("Error with result c[%d]=%lf on iter %d\n",i,c[i],k);
                icount++;
            }
        }
    }
    // Print average execution times
    printf("Average runtime is %lf msecs data transfer is %lf msecs\n",
           tkernel_sum/NTIMES, (ttotal_sum - tkernel_sum)/NTIMES);

    // free device memory
    // Always free GPU memory when done
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    // free host memory
    // Free CPU memory
    free(a);
    free(b);
    free(c);
    
    return 0;
}