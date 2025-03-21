#include <iostream>
#include <vector>
#include <mpi.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

/**
 * Parallel vector-matrix multiplication using MPI
 * This example demonstrates:
 * 1. Basic MPI initialization and finalization
 * 2. Process rank and size concepts
 * 3. Point-to-point communication (Send/Recv)
 * 4. Collective communication (Broadcast)
 * 5. Data distribution and gathering
 * 6. Local computation with distributed data
 */
int main(int argc, char** argv) {
    // Initialize the MPI environment
    // This must be called before any other MPI function
    MPI_Init(&argc, &argv);
    
    // Get the rank (process ID) and size (total number of processes)
    // rank: unique identifier for each process (0 to size-1)
    // size: total number of processes running the program
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Start total program timer
    double total_start = MPI_Wtime();

    // Problem size: N x N matrix and N-element vector
    // This determines the size of our computation
    const size_t N = 10000; // Adjust as needed

    // Calculate how many rows each process will handle
    // This divides the work evenly among processes
    size_t rows_per_proc = N / size;
    size_t local_start_row = rank * rows_per_proc;
    size_t local_end_row = (rank == size - 1) ? N : local_start_row + rows_per_proc;
    size_t local_rows = local_end_row - local_start_row;

    // Start initialization timer
    double init_start = MPI_Wtime();

    // Allocate memory for our data structures
    // matrix: full matrix (only used by rank 0)
    // vector: input vector (will be broadcast to all processes)
    // local_result: each process's portion of the result
    // result: final result (only used by rank 0)
    std::vector<double> matrix;
    std::vector<double> vector(N);
    std::vector<double> local_result(local_rows, 0.0);
    std::vector<double> result;

    // Only rank 0 (master process) initializes the full data
    if (rank == 0) {
        // Allocate space for the full matrix and result vector
        matrix.resize(N * N);
        result.resize(N);

        // Initialize with sample data
        // In a real application, this would be your actual data
        for (size_t i = 0; i < N; i++) {
            vector[i] = i + 1.0;
            for (size_t j = 0; j < N; j++) {
                matrix[i * N + j] = (i + j) / 2.0;
            }
        }

        std::cout << "Matrix and vector initialized." << std::endl;
    }

    double init_time = MPI_Wtime() - init_start;
    if (rank == 0) {
        std::cout << "Initialization time: " << init_time << " seconds" << std::endl;
    }

    // Start distribution timer
    double dist_start = MPI_Wtime();

    // Broadcast the vector to all processes
    // MPI_Bcast: one-to-all communication
    // Parameters: data, count, datatype, root (rank of process broadcasting data), communicator
    MPI_Bcast(vector.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Allocate memory for this process's portion of the matrix
    std::vector<double> local_matrix(local_rows * N);

    // Distribute matrix rows to all processes
    if (rank == 0) {
        // Master process sends parts of the matrix to other processes
        for (int dest = 1; dest < size; dest++) {
            size_t dest_start_row = dest * rows_per_proc;
            size_t dest_rows = (dest == size - 1) ? (N - dest_start_row) : rows_per_proc;
            // MPI_Send: point-to-point communication
            MPI_Send(&matrix[dest_start_row * N], dest_rows * N, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
        }

        // Copy local part for rank 0
        for (size_t i = 0; i < local_rows * N; i++) {
            local_matrix[i] = matrix[i];
        }
    } 
    else {
        // Other processes receive their portion of the matrix
        // MPI_Recv: point-to-point communication
        MPI_Recv(local_matrix.data(), local_rows * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double dist_time = MPI_Wtime() - dist_start;
    if (rank == 0) {
        std::cout << "Distribution time: " << dist_time << " seconds" << std::endl;
    }

    // Start computation timer
    double comp_start = MPI_Wtime();

    // Each process performs its portion of the matrix-vector multiplication
    // This is the actual computation part
    for (size_t i = 0; i < local_rows; i++) {
        local_result[i] = 0.0;
        for (size_t j = 0; j < N; j++) {
            local_result[i] += local_matrix[i * N + j] * vector[j];
        }
    }

    double comp_time = MPI_Wtime() - comp_start;
    if (rank == 0) {
        std::cout << "Computation time: " << comp_time << " seconds" << std::endl;
    }

    // Start gather timer
    double gather_start = MPI_Wtime();

    // Gather results from all processes back to rank 0
    if (rank == 0) {
        // Copy local results to the final result vector
        for (size_t i = 0; i < local_rows; i++) {
            result[i] = local_result[i];
        }

        // Receive results from other processes
        for (int source = 1; source < size; source++) {
            size_t source_start_row = source * rows_per_proc;
            size_t source_rows = (source == size - 1) ? (N - source_start_row) : rows_per_proc;
            MPI_Recv(&result[source_start_row], source_rows, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        std::cout << "Result vector computed successfully." << std::endl;
        
        // Print first few elements for verification
        std::cout << "First 5 elements of result: ";
        for (size_t i = 0; i < std::min(5UL, N); i++) {
            std::cout << result[i] << " ";
        }
        std::cout << std::endl;
    } 
    else {
        // Other processes send their results to rank 0
        MPI_Send(local_result.data(), local_rows, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    double gather_time = MPI_Wtime() - gather_start;
    if (rank == 0) {
        std::cout << "Gather time: " << gather_time << " seconds" << std::endl;
    }

    // Calculate and print total time
    double total_time = MPI_Wtime() - total_start;
    if (rank == 0) {
        std::cout << "\nTiming Summary:" << std::endl;
        std::cout << "Initialization: " << init_time << " seconds" << std::endl;
        std::cout << "Distribution: " << dist_time << " seconds" << std::endl;
        std::cout << "Computation: " << comp_time << " seconds" << std::endl;
        std::cout << "Gather: " << gather_time << " seconds" << std::endl;
        std::cout << "Total time: " << total_time << " seconds" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    // Clean up MPI environment
    MPI_Finalize();
    return 0;
}