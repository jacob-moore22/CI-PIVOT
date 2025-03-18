#include <stdio.h>
#include <iostream>
#include <matar.h>

using namespace mtr; // matar namespace


#define MATRIX_SIZE 1024 //

// main
int main(int argc, char* argv[])
{
    MATAR_KOKKOS_INIT
    { // kokkos scope
    printf("Starting MATAR Matrix Multiplication test \n");



    // Create arrays on the device, where the device is either the CPU or GPU depending on how it is compiled
    CArrayKokkos<int> A(MATRIX_SIZE, MATRIX_SIZE);
    CArrayKokkos<int> B(MATRIX_SIZE, MATRIX_SIZE);
    CArrayKokkos<int> C(MATRIX_SIZE, MATRIX_SIZE);

    // Initialize arrays (NOTE: This is on the device)
    A.set_values(2);
    B.set_values(2);
    C.set_values(0);

    // Perform C = A * B
    FOR_ALL(i, 0, MATRIX_SIZE,
            j, 0, MATRIX_SIZE,
            k, 0, MATRIX_SIZE, {
        
        C(i,j) += A(i,k) * B(k,j);
    });

  
    } // end kokkos scope
    MATAR_KOKKOS_FINALIZE

    return 0;
}