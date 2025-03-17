#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int count = 10;
    double xsend[count], xrecv[count];
    for (int i=0; i<count; i++){
        xsend[i] = (double)i;
    }

    // Get the rank and number of processes
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    // Check if the number of processes is even
    if (nprocs % 2 == 1) {
        if (rank == 0) {
            printf("Error: Must be called with an even number of processes\n");
        }
        exit(1);
    }

    // Calculate the tag and partner rank
    // The calculation creates pairs of processes (0-1, 2-3, 4-5, etc.) where:
    // Each pair shares the same tag value (which is the pair number)
    // Within each pair, processes communicate with each other (partner_rank)
    // The program requires an even number of processes to ensure every process has a partner
    // This pairing scheme is a common pattern in MPI programs for point-to-point communication, where processes need to exchange data in pairs.
    int tag = rank/2;
    int partner_rank = (rank/2)*2 + (rank+1)%2;
    MPI_Comm comm = MPI_COMM_WORLD;

    // Fix deadlock by having even ranks send first, odd ranks receive first
    if (rank % 2 == 0) {
        // Even ranks: Send first, then receive
        MPI_Send(xsend, count, MPI_DOUBLE, partner_rank, tag, comm);
        MPI_Recv(xrecv, count, MPI_DOUBLE, partner_rank, tag, comm, MPI_STATUS_IGNORE);
    } else {
        // Odd ranks: Receive first, then send
        MPI_Recv(xrecv, count, MPI_DOUBLE, partner_rank, tag, comm, MPI_STATUS_IGNORE);
        MPI_Send(xsend, count, MPI_DOUBLE, partner_rank, tag, comm);
    }

    if (rank == 0) printf("SendRecv successfully completed\n");

    // Verify that the data was correctly received
    int errors = 0;
    for (int i = 0; i < count; i++) {
        // The received values should match what the partner sent (0,1,2,...,9)
        if (xrecv[i] != (double)i) {
            errors++;
            printf("Process %d: Error in received data at index %d. Expected %f, got %f\n", 
                   rank, i, (double)i, xrecv[i]);
        }
    }
    
    // Use a reduction to collect error counts from all processes
    int total_errors;
    MPI_Reduce(&errors, &total_errors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // Print results
    if (rank == 0) {
        if (total_errors == 0) {
            printf("SendRecv successfully completed: All data was correctly transferred!\n");
        } else {
            printf("SendRecv completed with %d errors in data transfer\n", total_errors);
        }
    }

    MPI_Finalize();
    return 0;
}