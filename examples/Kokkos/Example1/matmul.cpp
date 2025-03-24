// g++ -std=c++14 -O3 kokkos_matrix_multiply.cpp -o kokkos_matrix_multiply \
//   -I${KOKKOS_PATH}/include -L${KOKKOS_PATH}/lib -lkokkos -DKOKKOS_ENABLE_CUDA

// Include the core Kokkos header which provides the basic Kokkos functionality
#include <Kokkos_Core.hpp>
// Include Kokkos random number generation utilities
#include <Kokkos_Random.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>

// Helper function to print matrix portions - demonstrates Kokkos View mirroring
template<typename ViewType>
void print_matrix_portion(const ViewType& matrix, int size, int print_size, const std::string& name) {
  // Create a host-side mirror view of the matrix and copy the data
  // This is necessary because Kokkos Views are device-side by default
  auto h_matrix = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrix);
  
  std::cout << name << " (showing " << print_size << "x" << print_size << " elements):" << std::endl;
  for (int i = 0; i < print_size; ++i) {
    for (int j = 0; j < print_size; ++j) {
      std::cout << std::fixed << std::setprecision(2) << h_matrix(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main(int argc, char* argv[]) {
  // Initialize Kokkos - this must be called before using any Kokkos functionality
  // It sets up the execution space (CPU, GPU, etc.) based on configuration
  Kokkos::initialize(argc, argv);
  
  {
    // Matrix dimensions
    const int N = 1024;
    const int print_size = 3;  // Print only the first 3x3 elements
    
    // Create Kokkos Views for matrices A, B, and C
    // Views are Kokkos' primary data structure for managing device memory
    // The ** indicates a 2D array layout
    Kokkos::View<double**> A("A", N, N);
    Kokkos::View<double**> B("B", N, N);
    Kokkos::View<double**> C("C", N, N);
    
    // Initialize matrices A and B with random values using Kokkos parallel_for
    // This demonstrates Kokkos' parallel execution model
    Kokkos::Random_XorShift64_Pool<> random_pool(12345);  // Create a random number pool
    Kokkos::parallel_for("init_matrices", N * N, KOKKOS_LAMBDA(const int idx) {
      const int i = idx / N;
      const int j = idx % N;
      
      // Get a random number generator from the pool
      auto rand_gen = random_pool.get_state();
      A(i, j) = rand_gen.drand(0.0, 1.0);
      B(i, j) = rand_gen.drand(0.0, 1.0);
      // Return the generator to the pool
      random_pool.free_state(rand_gen);
    });
    
    // Initialize matrix C with zeros using Kokkos deep_copy
    // deep_copy is used to copy data between host and device memory
    Kokkos::deep_copy(C, 0.0);
    
    // Print execution space information
    // This shows which backend (CPU, GPU, etc.) Kokkos is using
    std::cout << "Kokkos execution space: " 
              << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
    
    // Print small portion of matrices A and B for verification
    print_matrix_portion(A, N, print_size, "Matrix A");
    print_matrix_portion(B, N, print_size, "Matrix B");
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform matrix multiplication using Kokkos MDRangePolicy
    // MDRangePolicy allows for nested parallel loops with better performance
    // The Rank<2> indicates we're using 2D iteration space
    Kokkos::parallel_for("matrix_multiply", 
                         Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, N}),
                         KOKKOS_LAMBDA(const int i, const int j) {
      double sum = 0.0;
      for (int k = 0; k < N; ++k) {
        sum += A(i, k) * B(k, j);
      }
      C(i, j) = sum;
    });
    
    // Ensure all operations are complete before proceeding
    // This is important for timing and correctness
    Kokkos::fence();
    
    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Print result portion for verification
    print_matrix_portion(C, N, print_size, "Result Matrix C");
    
    // Print performance information
    std::cout << "Matrix multiplication completed in " << duration << " ms" << std::endl;
    
    // Verify result by computing a small portion on CPU
    // This demonstrates how to copy data back to host for verification
    auto h_A = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
    auto h_B = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), B);
    auto h_C = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), C);
    
    bool correct = true;
    for (int i = 0; i < print_size && correct; ++i) {
      for (int j = 0; j < print_size && correct; ++j) {
        double expected = 0.0;
        for (int k = 0; k < N; ++k) {
          expected += h_A(i, k) * h_B(k, j);
        }
        double relative_error = std::abs((h_C(i, j) - expected) / expected);
        if (relative_error > 1e-5) {
          std::cout << "Verification failed at (" << i << "," << j << "): "
                    << "Expected " << expected << ", got " << h_C(i, j) << std::endl;
          correct = false;
        }
      }
    }
    
    if (correct) {
      std::cout << "Verification passed for sampled elements." << std::endl;
    }
    
    // Calculate and print performance metrics
    double gflops = 2.0 * N * N * N / (duration * 1e6);
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
  }
  
  // Finalize Kokkos - this must be called after all Kokkos operations are complete
  Kokkos::finalize();
  
  return 0;
}