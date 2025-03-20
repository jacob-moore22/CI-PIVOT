// nvcc -o matrix_multiply matrix_multiply.cu
// ./matrix_multiply

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro
#define cudaCheckError() {                                                              \
    cudaError_t e = cudaGetLastError();                                                 \
    if (e != cudaSuccess) {                                                             \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(EXIT_FAILURE);                                                             \
    }                                                                                   \
}

// Matrix dimensions
#define MATRIX_SIZE 1024
#define BLOCK_SIZE 32

// CUDA kernel for matrix multiplication
__global__ void matrixMultiply(float *A, float *B, float *C, int size) {
    // Calculate global row and column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if we're within matrix bounds
    if (row < size && col < size) {
        float sum = 0.0f;
        
        // Compute the dot product of row of A and column of B
        for (int i = 0; i < size; i++) {
            sum += A[row * size + i] * B[i * size + col];
        }
        
        // Store the result
        C[row * size + col] = sum;
    }
}

int main() {
    // Set matrix dimensions
    const int size = MATRIX_SIZE;
    const int bytes = size * size * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    
    // Initialize matrices with random values
    for (int i = 0; i < size * size; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);
    cudaCheckError();
    
    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaCheckError();
    
    // Set up execution configuration
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((size + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                        (size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch the kernel
    printf("Performing matrix multiplication...\n");
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);
    cudaCheckError();
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();
    cudaCheckError();
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaCheckError();
    
    // Verify result by checking a few elements
    printf("Verification (showing first 3x3 elements of result):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float expected = 0.0f;
            for (int k = 0; k < size; k++) {
                expected += h_A[i * size + k] * h_B[k * size + j];
            }
            printf("C[%d][%d] = %.2f (expected: %.2f)\n", i, j, h_C[i * size + j], expected);
        }
    }
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("Matrix multiplication completed successfully.\n");
    return 0;
}