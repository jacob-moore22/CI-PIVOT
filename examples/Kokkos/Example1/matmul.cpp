// g++ -std=c++14 -O3 kokkos_matrix_multiply.cpp -o kokkos_matrix_multiply \
//   -I${KOKKOS_PATH}/include -L${KOKKOS_PATH}/lib -lkokkos -DKOKKOS_ENABLE_CUDA

#include <Kokkos_Core.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>

// Print a portion of a matrix (for verification)
template<typename ViewType>
void print_matrix_portion(const ViewType& matrix, int size, int print_size, const std::string& name) {
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
  // Initialize Kokkos
  Kokkos::initialize(argc, argv);
  
  {
    // Matrix dimensions
    const int N = 1024;
    const int print_size = 3;  // Print only the first 3x3 elements
    
    // Create Kokkos Views for matrices A, B, and C
    Kokkos::View<double**> A("A", N, N);
    Kokkos::View<double**> B("B", N, N);
    Kokkos::View<double**> C("C", N, N);
    
    // Initialize matrices A and B with random values
    Kokkos::Random_XorShift64_Pool<> random_pool(12345);
    Kokkos::parallel_for("init_matrices", N * N, KOKKOS_LAMBDA(const int idx) {
      const int i = idx / N;
      const int j = idx % N;
      
      auto rand_gen = random_pool.get_state();
      A(i, j) = rand_gen.drand(0.0, 1.0);
      B(i, j) = rand_gen.drand(0.0, 1.0);
      random_pool.free_state(rand_gen);
    });
    
    // Initialize matrix C with zeros
    Kokkos::deep_copy(C, 0.0);
    
    // Print execution space information
    std::cout << "Kokkos execution space: " 
              << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
    
    // Print small portion of matrices A and B for verification
    print_matrix_portion(A, N, print_size, "Matrix A");
    print_matrix_portion(B, N, print_size, "Matrix B");
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform matrix multiplication: C = A * B
    Kokkos::parallel_for("matrix_multiply", 
                         Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, N}),
                         KOKKOS_LAMBDA(const int i, const int j) {
      double sum = 0.0;
      for (int k = 0; k < N; ++k) {
        sum += A(i, k) * B(k, j);
      }
      C(i, j) = sum;
    });
    
    // Ensure all operations are complete
    Kokkos::fence();
    
    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Print result portion for verification
    print_matrix_portion(C, N, print_size, "Result Matrix C");
    
    // Print performance information
    std::cout << "Matrix multiplication completed in " << duration << " ms" << std::endl;
    
    // Verify result (compute a small portion on CPU for comparison)
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
    
    // Performance metrics
    double gflops = 2.0 * N * N * N / (duration * 1e6);
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
  }
  
  // Finalize Kokkos
  Kokkos::finalize();
  
  return 0;
}