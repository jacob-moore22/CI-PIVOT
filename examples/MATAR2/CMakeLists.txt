cmake_minimum_required(VERSION 3.10)
project(matar_matmul)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Kokkos REQUIRED)
find_package(MATAR REQUIRED)

# Add executable
add_executable(matmul matar_matmul.cpp)

# Link against required libraries
target_link_libraries(matmul PRIVATE 
    Kokkos::kokkos
    MATAR::matar
)

# Add include directories
target_include_directories(matmul PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Print configuration information
message(STATUS "Building MATAR matrix multiplication example")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Kokkos found: ${Kokkos_FOUND}")
message(STATUS "MATAR found: ${MATAR_FOUND}") 