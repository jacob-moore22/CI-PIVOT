/** -*- C++ -*-
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
//
// __global__ indicates that this is a kernel function that will be
// executed on the device. This is a special CUDA keyword that tells
// the compiler this function will be called from the host and
// executed on the GPU device. (NOTE: this code must be compiled with
// a CUDA aware compiler such as nvcc or nvc++.)
__global__ void StreamTriad(
                            const int n,           // Total number of elements to process
                            const double scalar,   // Scalar multiplier for the triad operation
                            const double *a,       // Input array a
                            const double *b,       // Input array b
                            double *c)       // Output array c
{
  // Get the global thread index to determine which element to process
  // gridDim:   {x,y,z} Number of blocks in the compute grid
  // blockIdx:  {x,y,z} Current block index within the compute grid
  // blockDim:  {x,y,z} Number of threads within each block
  // threadIdx: {x,y,z} Current thread index within the current block
  //
  // NOTE: In this example, we are using a 1D arrays. So, we only need
  // to use the x values
  const int i = blockIdx.x*blockDim.x + threadIdx.x;

  // NOTE: This kernel can be called with any sized grid, with any
  // sized blocks. Often times, the array size is not evenly divisible
  // by the block size. In that case, there may be some allocated
  // threads for which there are no valid array elements. So, it is
  // good practice to add an array bounds check. If the current thread
  // is beyond the array bounds, then do nothing.
  if (i >= n) return;

  // TODO: For all remaining threads, perform a single triad operation:
  // e.g.  c[i] = a[i] + scalar*b[i]
  
  
  
  // BONUS: What would happen if this kernel were called with too few
  // blocks and threads? How could you modify this code to handle this
  // condition?
  

  
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
  const int stream_array_size = 80000000;  // Size of arrays (80 million elements)
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
  
  // Variable to store CUDA error codes
  cudaError_t err;
  
  // TODO: Allocate memory on the GPU for each array. Use cudaMalloc
  // to allocate memory on the GPU device.  
  double *a_d = nullptr;
  double *b_d = nullptr; // (_d indicates device pointers)
  double *c_d = nullptr;
  
  // USAGE: err = cudaMalloc(&a_d, stream_array_size*sizeof(double));


  
  // TIP: Check err to ensure that CUDA API calls return cudaSuccess.
  // if (err != cudaSuccess) {
  //   printf("Error allocating array on device: %s\n", cudaGetErrorString(err));
  //   return -1; // Exit with -1 on error
  // }
  
  

  // Set the grid and block sizes so that there are at least one
  // thread for each element in the array.
  
  // Block size is the number of threads per block (must be a multiple of 32)
  const int blocksize = 512;  // Common block size for modern GPUs
  // Calculate grid size to cover all elements, rounding up to ensure full coverage
  const int gridsize = (stream_array_size + blocksize - 1)/blocksize;

  // Main benchmark loop
  for (int k = 0; k < NTIMES; k++){
    cpu_timer_start(&ttotal);  // Start total time measurement
        
    // TODO: Copy input array data from host to device. Use cudaMemcpy
    // with transfer direction specified as cudaMemcpyHostToDevice.
    
    // USAGE: err = cudaMemcpy(a_d, a, stream_array_size*sizeof(double), cudaMemcpyHostToDevice);
    
    
    
    // Synchronize to ensure data transfer is complete.
    // This ensures all data is on the GPU before kernel launch
    cudaDeviceSynchronize();

    cpu_timer_start(&tkernel);  // Start kernel time measurement
        
    // TODO: Launch stream triad kernel.
    
    // <<<gridsize, blocksize>>> is CUDA's special syntax for kernel
    // launch configuration. These arguments specify the total number
    // of blocks and the number of threads per block to allocate on
    // the GPU for this kernel.
    
    // USAGE: StreamTriad<<<gridsize, blocksize>>>(stream_array_size, scalar, a_d, b_d, c_d);

    
        
    // TIP: Check for kernel launch errors
    // err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //   printf("Error launching kernel: %s\n", cudaGetErrorString(err));
    //   return -1;
    // }

    
        
    // Synchronize to ensure kernel completion. This ensures the
    // kernel has finished before we try to copy results back to the
    // host.
    cudaDeviceSynchronize();
    
    tkernel_sum += cpu_timer_stop(tkernel);  // Record kernel execution time

    // TODO: Copy results back from device to host.  Use
    // cudaMemcpyDeviceToHost to indicate transfer direction.
    // NOTE: This operation is blocking, so no need for additional
    // synchronization.
    // USAGE: err = cudaMemcpy(c, c_d, stream_array_size*sizeof(double), cudaMemcpyDeviceToHost);


    
    // TIP: Check err code for cudaSuccess.
    // if (err != cudaSuccess) {
    //   printf("Error copying c from device: %s\n", cudaGetErrorString(err));
    //   return -1;
    // }
        
    ttotal_sum += cpu_timer_stop(ttotal);  // Record total time
        
    // Check results and print errors if found. limit to only 10 errors per iteration
    // This verifies the computation was correct
    for (int i=0, icount=0; i<stream_array_size && icount < 10; i++){
      // When comparing floating point values, allow for a very small
      // epislon difference.
      if (fabs(c[i] - (a[i] + scalar*b[i])) > 1e-10) {
        printf("Error with result c[%d]=%lf on iter %d\n",i,c[i],k);
        icount++;
      }
    }
  }
  
  // Print average execution times
  printf("Average runtime is %lf msecs data transfer is %lf msecs\n",
         tkernel_sum/NTIMES, (ttotal_sum - tkernel_sum)/NTIMES);

  // Free device memory.
  // Always free GPU memory when done!
  if (a_d) cudaFree(a_d);
  if (b_d) cudaFree(b_d);
  if (c_d) cudaFree(c_d);

  // Free host memory.
  // Always free CPU memory when done!
  if (a) free(a);
  if (b) free(b);
  if (c) free(c);
  
  return 0; // Exit with 0 on successful completion.
}