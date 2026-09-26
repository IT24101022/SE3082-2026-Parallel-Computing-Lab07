/*
 * Exercise 04: Replace Gather with Reduce
 *
 * Student: IT24101022
 * Subject: SE3082 - Parallel Computing
 * Lab: 07 - MPI Collective Communication
 *
 * Strategy:
 *   - MPI_Scatter distributes chunks
 *   - MPI_Reduce combines local_sums using MPI_SUM in one call
 *   - No need for all_sums array or manual summation loop on root
 *   - Result available ONLY on root (rank 0)
 *
 * Compile: mpicc -o sum_reduce sum_reduce.c
 * Run:     mpirun -np 4 ./sum_reduce
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
        printf("Root filled array with values 1 to %d\n", N);
    }

    int *local_chunk = (int *)malloc(chunk_size * sizeof(int));

    double start = MPI_Wtime();

    /* Scatter chunks to all processes */
    MPI_Scatter(array, chunk_size, MPI_INT,
                local_chunk, chunk_size, MPI_INT,
                0, MPI_COMM_WORLD);

    /* Each process computes its local sum */
    long long local_sum = 0;
    for (int i = 0; i < chunk_size; i++)
        local_sum += local_chunk[i];

    printf("  Rank %d: local_sum = %lld\n", rank, local_sum);

    /*
     * REDUCE: Combines all local_sum values using MPI_SUM.
     *
     * Replaces Exercise 3's Gather + manual summation loop.
     * MPI uses a tree-based algorithm: O(log P) steps instead of O(P).
     *
     * IMPORTANT: total_sum is ONLY valid on root (rank 0).
     * Non-root processes will have garbage in total_sum after this call.
     *
     * MPI_Reduce operations: MPI_SUM, MPI_MAX, MPI_MIN, MPI_PROD,
     *                        MPI_LAND, MPI_BAND, MPI_LOR, MPI_BOR, etc.
     */
    long long total_sum = 0;
    MPI_Reduce(
        &local_sum,   /* sendbuf: value from each process */
        &total_sum,   /* recvbuf: result stored here (root only) */
        1,            /* count: 1 value per process */
        MPI_LONG_LONG,
        MPI_SUM,      /* operation: sum all values */
        0,            /* root: process 0 receives the result */
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        double elapsed = MPI_Wtime() - start;
        long long expected = (long long)N * (N + 1) / 2;
        printf("\n[Reduce] Total sum = %lld\n", total_sum);
        printf("[Reduce] Expected  = %lld\n", expected);
        printf("[Reduce] Correct?  = %s\n", total_sum == expected ? "YES" : "NO");
        printf("[Reduce] Time      = %.4f sec\n", elapsed);
        free(array);
    }

    free(local_chunk);
    MPI_Finalize();
    return 0;
}
