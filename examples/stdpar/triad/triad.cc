#include <iostream>
#include <vector>
#include <algorithm>
#include <execution>
#include <numeric>
#include <cmath>

// Function to perform the element-wise operation:
// c[i] = a[i] + scalar * b[i]
void vector_operation(const std::vector<double>& a,
                      const std::vector<double>& b,
                      std::vector<double>& c,
                      double scalar) {
  // Ensure vectors have the same size
  if (a.size() != b.size() || a.size() != c.size()) {
    throw std::runtime_error("Vectors must have the same size.");
  }

  // Use std::transform with a lambda function and the execution::par
  // policy for parallel execution
  std::transform(std::execution::par, a.begin(), a.end(), b.begin(), c.begin(),
                 [scalar](double a_val, double b_val) {
                   return a_val + scalar * b_val;
                 });
}

int main() {
  // Example usage
  size_t size = 100;
  std::vector<double> a(size);
  std::vector<double> b(size);
  std::vector<double> c(size); // Result vector
  double scalar = 2.0;

  // Initialize vectors a and b with some values (e.g., using iota and a transformation)
  std::iota(a.begin(), a.end(), 1.0); // Fill with 1, 2, 3, ..., 10
  std::transform(a.begin(), a.end(), b.begin(),
                 [](double x){ return std::sqrt(x * 10);}); // Fill b with sqrt(i*10)

  // Perform the vector operation
  try {
    vector_operation(a, b, c, scalar);

    // Print the input vectors and the result
    std::cout << "Vector a: ";
    for (double val : a) {
      std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "Vector b: ";
    for (double val : b) {
      std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "Scalar: " << scalar << std::endl;

    std::cout << "Result Vector c: ";
    for (double val : c) {
      std::cout << val << " ";
    }
    std::cout << std::endl;

    // Check answers
    for (int i=0; i<size; ++i) {
      if (fabs(c[i] - (a[i] + scalar*b[i])) > 1e-8) {
        std::cout << "Error: " << c[i] << " != " << (a[i] + scalar*b[i]) << std::endl;
      }
    }
  } catch (const std::runtime_error& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1; // Return a non-zero value to indicate an error
  }

  return 0;
}
