cmake_minimum_required(VERSION 3.0)
project(HealthData)
set(CMAKE_CXX_STANDARD 11)  # Changed to C++ standard since we're using .cpp

# Set optimization flags for various compilers
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") # using Clang
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstrict-aliasing -march=native -mtune=native")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Rpass-analysis=loop-vectorize")
   set(CMAKE_NOVEC_FLAGS "${CMAKE_NOVEC_FLAGS} -fnovectorize")
   set(CMAKE_VEC_FLAGS "${CMAKE_VEC_FLAGS} -fvectorize")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") # using GCC
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstrict-aliasing -march=native -mtune=native")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-loop-optimized")
   set(CMAKE_NOVEC_FLAGS "${CMAKE_NOVEC_FLAGS} -fno-tree-vectorize")
   set(CMAKE_VEC_FLAGS "${CMAKE_VEC_FLAGS} -ftree-vectorize")
   if ("${CMAKE_CXX_COMPILER_VERSION}" VERSION_GREATER "7.4.0")
      set(CMAKE_VEC_FLAGS "${CMAKE_VEC_FLAGS} -mprefer-vector-width=512")
   endif ("${CMAKE_CXX_COMPILER_VERSION}" VERSION_GREATER "7.4.0")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel") # using Intel C++
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ansi-alias")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report-phase=openmp,loop,vec")
   set(CMAKE_NOVEC_FLAGS "${CMAKE_NOVEC_FLAGS} -no-vec")
   set(CMAKE_VEC_FLAGS "${CMAKE_VEC_FLAGS} -xHOST")
   if ("${CMAKE_CXX_COMPILER_VERSION}" VERSION_GREATER "17.0.4")
     set(CMAKE_VEC_FLAGS "${CMAKE_VEC_FLAGS} -qopt-zmm-usage=high")
   endif ("${CMAKE_CXX_COMPILER_VERSION}" VERSION_GREATER "17.0.4")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Qvec-report:2")
   set(CMAKE_NOVEC_FLAGS "${CMAKE_NOVEC_FLAGS}")
   set(CMAKE_VEC_FLAGS "${CMAKE_VEC_FLAGS}")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "XL")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qalias=ansi -qhot -qarch=pwr9 -qtune=pwr9")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qreport")
   set(CMAKE_NOVEC_FLAGS "${CMAKE_NOVEC_FLAGS} -qsimd=noauto")
   set(CMAKE_VEC_FLAGS "${CMAKE_VEC_FLAGS} -qsimd=auto")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Cray")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -h restrict=a")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -h msgs -h negmsgs -h list=a")
   set(CMAKE_NOVEC_FLAGS "${CMAKE_NOVEC_FLAGS} -h vector=0")
   set(CMAKE_VEC_FLAGS "${CMAKE_VEC_FLAGS} -h vector=3")
endif()

# Find OpenMP package
find_package(OpenMP)

# Add the HealthData executable
add_executable(HealthData HealthData.cpp)
set_target_properties(HealthData PROPERTIES COMPILE_FLAGS ${CMAKE_NOVEC_FLAGS})

# Check if OpenMP was found and add flags if it was
if(OpenMP_CXX_FOUND)
    set_target_properties(HealthData PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
    set_target_properties(HealthData PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
endif()

# Cleanup using: cmake --build . --target distclean
add_custom_target(distclean COMMAND rm -rf CMakeCache.txt CMakeFiles
                  Makefile cmake_install.cmake HealthData.dSYM ipo_out.optrpt)
