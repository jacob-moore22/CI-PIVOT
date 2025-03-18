#include <stdio.h>
#include <omp.h>
#include "timer.h"

// large enough to force into main memory
#define ARRAY_SIZE 80000000
static double a[ARRAY_SIZE], b[ARRAY_SIZE], c[ARRAY_SIZE];

void vector_add(double *c, double *a, double *b, int n);

int main(int argc, char *argv[]){
   int num_devices = omp_get_num_devices();
   printf("Number of available devices: %d\n", num_devices);
   
   // Check if we have GPU devices available
   if (num_devices < 1) {
      printf("No GPU devices available, falling back to CPU execution\n");
   } else {
      printf("Using GPU device 0\n");
      omp_set_default_device(0);
   }

   #pragma omp parallel
      if (omp_get_thread_num() == 0)
         printf("Running with %d host thread(s)\n", omp_get_num_threads());

   struct timespec tstart;
   double time_sum = 0.0;

   // Initialize arrays on the host
   #pragma omp parallel for
   for (int i=0; i<ARRAY_SIZE; i++) {
      a[i] = 1.0;
      b[i] = 2.0;
   }

   cpu_timer_start(&tstart);
   vector_add(c, a, b, ARRAY_SIZE);
   time_sum += cpu_timer_stop(tstart);

   // Verify result on a small subset
   int errors = 0;
   for (int i=0; i < 100; i++) {
      if (c[i] != 3.0) {
         errors++;
         if (errors < 10) {  // Limit output for large error counts
            printf("Verification error at index %d: c[%d] = %f, expected 3.0\n", i, i, c[i]);
         }
      }
   }
   
   if (errors == 0) {
      printf("Verification passed for first 100 elements\n");
   } else {
      printf("Verification failed: %d errors found in first 100 elements\n", errors);
   }

   printf("Runtime is %lf msecs\n", time_sum);
}

void vector_add(double *c, double *a, double *b, int n)
{
   #pragma omp target data map(to: a[0:n], b[0:n]) map(from: c[0:n])
   {
      #pragma omp target teams distribute parallel for
      for (int i=0; i < n; i++){
         c[i] = a[i] + b[i];
      }
   }
}