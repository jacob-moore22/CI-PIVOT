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
    int tag = rank/2;
    int partner_rank = (rank/2)*2 + (rank+1)%2;
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Request requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    MPI_Irecv(xrecv, count, MPI_DOUBLE, partner_rank, tag, comm, &requests[0]);
    MPI_Isend(xsend, count, MPI_DOUBLE, partner_rank, tag, comm, &requests[1]);
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

   if (rank == 0) printf("SendRecv successfully completed\n");

   MPI_Finalize();
   return 0;
}