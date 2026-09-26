/*
 * Exercise 05: Replace Reduce with Allreduce
 *
 * Student: IT24101022
 * Subject: SE3082 - Parallel Computing
 * Lab: 07 - MPI Collective Communication
 *
 * Strategy:
 *   - MPI_Allreduce makes the total_sum available on ALL processes
 *   - No root parameter needed
 *   - Every process prints its local_sum, total_sum, and % contribution
 *
 * Compile: mpicc -o sum_allreduce sum_allreduce.c
 * Run:     mpirun -np 4 ./sum_allreduce
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = N / size;

    /* Only root allocates full array */
    int *array = NULL;
    if (rank == 0) {
        array = (int *)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++)
            array[i] = i + 1;
        printf("Root filled array with values 1 to %d\n\n", N);
    }

    int *local_chunk = (int *)malloc(chunk_size * sizeof(int));

    double start = MPI_Wtime();

    /* Scatter chunks */
    MPI_Scatter(array, chunk_size, MPI_INT,
                local_chunk, chunk_size, MPI_INT,
                0, MPI_COMM_WORLD);

    /* Each process computes its local sum */
    long long local_sum = 0;
    for (int i = 0; i < chunk_size; i++)
        local_sum += local_chunk[i];

    /*
     * ALLREDUCE: All processes get the result.
     *
     * Difference from MPI_Reduce:
     *   - Reduce:    only ROOT gets total_sum
     *   - Allreduce: EVERY process gets total_sum
     *
     * Internally uses butterfly / recursive-doubling algorithm.
     * More efficient than MPI_Reduce + MPI_Bcast separately.
     *
     * No 'root' parameter — result goes to ALL processes.
     */
    long long total_sum = 0;
    MPI_Allreduce(
        &local_sum,    /* sendbuf: value from each process */
        &total_sum,    /* recvbuf: result stored on ALL processes */
        1,             /* count */
        MPI_LONG_LONG,
        MPI_SUM,       /* operation */
        MPI_COMM_WORLD /* no root parameter! */
    );

    /*
     * Now EVERY process can use total_sum.
     * Each process prints its contribution percentage.
     */
    double percentage = 100.0 * (double)local_sum / (double)total_sum;
    printf("  Rank %d: local_sum = %lld | total_sum = %lld | contribution = %.2f%%\n",
           rank, local_sum, total_sum, percentage);

    /* Barrier to ensure all prints finish before root prints summary */
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        double elapsed = MPI_Wtime() - start;
        long long expected = (long long)N * (N + 1) / 2;
        printf("\n[Allreduce] Total sum = %lld\n", total_sum);
        printf("[Allreduce] Expected  = %lld\n", expected);
        printf("[Allreduce] Correct?  = %s\n", total_sum == expected ? "YES" : "NO");
        printf("[Allreduce] Time      = %.4f sec\n", elapsed);
        free(array);
    }

    free(local_chunk);
    MPI_Finalize();
    return 0;
}
