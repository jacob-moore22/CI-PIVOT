#include <iostream>
#include <vector>
#include <mpi.h>
#include <cmath>

/**
 * Parallel vector-matrix multiplication using MPI
 * Multiplies an N-element vector by an N×N matrix
 */
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Problem size
    const int N = 1000; // Adjust as needed

    // Determine local work size
    int rows_per_proc = N / size;
    int local_start_row = rank * rows_per_proc;
    int local_end_row = (rank == size - 1) ? N : local_start_row + rows_per_proc;
    int local_rows = local_end_row - local_start_row;

    // Allocate memory for the matrix, vector, and result
    std::vector<double> matrix;
    std::vector<double> vector(N);
    std::vector<double> local_result(local_rows, 0.0);
    std::vector<double> result;

    // Master process initializes data and distributes
    if (rank == 0) {
        // Initialize the matrix and vector
        matrix.resize(N * N);
        result.resize(N);

        // Fill with sample data
        for (int i = 0; i < N; i++) {
            vector[i] = i + 1.0;
            for (int j = 0; j < N; j++) {
                matrix[i * N + j] = (i + j) / 2.0;
            }
        }

        std::cout << "Matrix and vector initialized." << std::endl;
    }

    // Broadcast the vector to all processes
    MPI_Bcast(vector.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Allocate memory for local part of the matrix
    std::vector<double> local_matrix(local_rows * N);

    // Distribute the matrix rows to all processes
    if (rank == 0) {
        // Send matrix parts to other processes
        for (int dest = 1; dest < size; dest++) {
            int dest_start_row = dest * rows_per_proc;
            int dest_rows = (dest == size - 1) ? (N - dest_start_row) : rows_per_proc;
            MPI_Send(&matrix[dest_start_row * N], dest_rows * N, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
        }
        
        // Copy local part for rank 0
        for (int i = 0; i < local_rows * N; i++) {
            local_matrix[i] = matrix[i];
        }
    } 
    else {
        // Receive matrix part from the master process
        MPI_Recv(local_matrix.data(), local_rows * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Perform local matrix-vector multiplication
    for (int i = 0; i < local_rows; i++) {
        local_result[i] = 0.0;
        for (int j = 0; j < N; j++) {
            local_result[i] += local_matrix[i * N + j] * vector[j];
        }
    }

    // Gather results from all processes
    if (rank == 0) {
        // Copy local results to the appropriate position in the final result
        for (int i = 0; i < local_rows; i++) {
            result[i] = local_result[i];
        }

        // Receive results from other processes
        for (int source = 1; source < size; source++) {
            int source_start_row = source * rows_per_proc;
            int source_rows = (source == size - 1) ? (N - source_start_row) : rows_per_proc;
            MPI_Recv(&result[source_start_row], source_rows, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        std::cout << "Result vector computed successfully." << std::endl;
        
        // Print first few elements for verification
        std::cout << "First 5 elements of result: ";
        for (int i = 0; i < std::min(5, N); i++) {
            std::cout << result[i] << " ";
        }
        std::cout << std::endl;
    } 
    else {
        // Send local results to the master process
        MPI_Send(local_result.data(), local_rows, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}