// CUDA kernel version of Stream Triad
// __global__ indicates that this is a kernel function that will be executed on the device
__global__ void StreamTriad(
                const int n,
                const double scalar,
                const double *a,
                const double *b,
                      double *c)
{
    // Get the global thread index to determine which element to process
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    // Protect from going out-of-bounds by checking if i is greater than or equal to n
    if (i >= n) return;

    // Perform the triad operation: c[i] = a[i] + scalar*b[i]
    c[i] = a[i] + scalar*b[i];
}

#include <stdio.h>
#include <sys/time.h>
#include "timer.h"

#define NTIMES 16

int main(int argc, char *argv[]){
    struct timespec tkernel, ttotal;
    // initializing data and arrays
    int stream_array_size = 80000000;
    double scalar = 3.0, tkernel_sum = 0.0, ttotal_sum = 0.0;

    // allocate host memory and initialize
    double *a = (double *)malloc(stream_array_size*sizeof(double));
    double *b = (double *)malloc(stream_array_size*sizeof(double));
    double *c = (double *)malloc(stream_array_size*sizeof(double));

    // Initialize arrays with values
    for (int i=0; i<stream_array_size; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;  // Initialize c array to 0
    }

    // allocate device memory
    double *a_d, *b_d, *c_d;
    cudaError_t err;
    
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
    int blocksize = 512;
    int gridsize = (stream_array_size + blocksize - 1)/blocksize;

    for (int k = 0; k < NTIMES; k++){
        cpu_timer_start(&ttotal);
        
        // copying array data from host to device
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
        cudaDeviceSynchronize();

        cpu_timer_start(&tkernel);
        
        // launch stream triad kernel
        // <<<gridsize, blocksize>>> is the number of blocks and threads per block
        StreamTriad<<<gridsize, blocksize>>>(stream_array_size, scalar, a_d, b_d, c_d);
        
        // Check for kernel launch errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error launching kernel: %s\n", cudaGetErrorString(err));
            return -1;
        }
        
        // Synchronize to ensure kernel completion
        cudaDeviceSynchronize();
        tkernel_sum += cpu_timer_stop(tkernel);

        // Copy results back from device to host
        // cuda memcpy from device to host blocks for completion so no need for synchronize
        err = cudaMemcpy(c, c_d, stream_array_size*sizeof(double), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("Error copying c from device: %s\n", cudaGetErrorString(err));
            return -1;
        }
        
        ttotal_sum += cpu_timer_stop(ttotal);
        
        // check results and print errors if found. limit to only 10 errors per iteration
        for (int i=0, icount=0; i<stream_array_size && icount < 10; i++){
            if (fabs(c[i] - (1.0 + 3.0*2.0)) > 1e-10) {  // Use small epsilon for floating point comparison
                printf("Error with result c[%d]=%lf on iter %d\n",i,c[i],k);
                icount++;
            }
        }
    }
    printf("Average runtime is %lf msecs data transfer is %lf msecs\n",
           tkernel_sum/NTIMES, (ttotal_sum - tkernel_sum)/NTIMES);

    // free device memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    // free host memory
    free(a);
    free(b);
    free(c);
    
    return 0;
}