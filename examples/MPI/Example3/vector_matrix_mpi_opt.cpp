#include <iostream>
#include <vector>
#include <mpi.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

// Aligned allocator for better cache performance
template<typename T, size_t Alignment>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template<typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept {}
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    pointer allocate(size_type n) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept {
        free(p);
    }
};

/**
 * Optimized parallel vector-matrix multiplication using MPI
 * This version includes:
 * 1. MPI_Scatter/Gather for efficient communication
 * 2. Memory alignment
 * 3. Process affinity
 */
int main(int argc, char** argv) {
    // Initialize MPI and get rank and size
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set process affinity
    #ifdef _GNU_SOURCE
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(rank, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    #endif

    // Start total program timer
    double total_start = MPI_Wtime();

    // Problem size
    const size_t N = 100000; // Adjust as needed

    // Calculate local work size
    size_t rows_per_proc = N / size;
    size_t local_start_row = rank * rows_per_proc;
    size_t local_end_row = (rank == size - 1) ? N : local_start_row + rows_per_proc;
    size_t local_rows = local_end_row - local_start_row;

    // Start initialization timer
    double init_start = MPI_Wtime();

    // Allocate aligned memory for data structures
    std::vector<double, AlignedAllocator<double, 64>> matrix;
    std::vector<double, AlignedAllocator<double, 64>> vector(N);
    std::vector<double, AlignedAllocator<double, 64>> local_result(local_rows, 0.0);
    std::vector<double, AlignedAllocator<double, 64>> result;

    // Initialize data on root process
    if (rank == 0) {
        matrix.resize(N * N);
        result.resize(N);

        // Initialize with sample data
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                matrix[i * N + j] = (i + j) / 2.0; // Row-major order for MPI_Scatter
                if (i == 0) vector[j] = j + 1.0;
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
    MPI_Bcast(vector.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Allocate memory for local matrix portion
    std::vector<double, AlignedAllocator<double, 64>> local_matrix(local_rows * N);

    // Scatter matrix rows to processes
    MPI_Scatter(matrix.data(), rows_per_proc * N, MPI_DOUBLE,
                local_matrix.data(), rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double dist_time = MPI_Wtime() - dist_start;
    if (rank == 0) {
        std::cout << "Distribution time: " << dist_time << " seconds" << std::endl;
    }

    // Start computation timer
    double comp_start = MPI_Wtime();

    // Perform local matrix-vector multiplication
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

    // Gather results using blocking communication
    if (rank == 0) {
        // Copy local results
        std::copy(local_result.begin(), local_result.end(), result.begin());

        // Receive results from other processes
        for (int source = 1; source < size; source++) {
            size_t source_start_row = source * rows_per_proc;
            size_t source_rows = (source == size - 1) ? (N - source_start_row) : rows_per_proc;
            MPI_Recv(&result[source_start_row], source_rows, MPI_DOUBLE,
                    source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
        // Send results to root process
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