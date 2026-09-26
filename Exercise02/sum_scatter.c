/*
 * Exercise 02: Replace Broadcast with Scatter
 *
 * Student: IT24101022
 * Subject: SE3082 - Parallel Computing
 * Lab: 07 - MPI Collective Communication
 *
 * Strategy:
 *   - Only root allocates the full array
 *   - MPI_Scatter distributes chunks to each process
 *   - Each process only receives its portion (memory efficient)
 *   - Results collected using MPI_Send / MPI_Recv (unchanged)
 *
 * Compile: mpicc -o sum_scatter sum_scatter.c
 * Run:     mpirun -np 4 ./sum_scatter
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

    /*
     * MEMORY IMPROVEMENT over Exercise 1:
     * Only root allocates the full array.
     * All other processes only allocate their small chunk.
     * This reduces memory usage from O(N * P) to O(N + P * chunk_size).
     */
    int *array = NULL;
    if (rank == 0) {
        array = (int *)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++)
            array[i] = i + 1;
        printf("Root filled array with values 1 to %d\n", N);
    }

    /* Each process allocates only its local chunk */
    int *local_chunk = (int *)malloc(chunk_size * sizeof(int));

    double start = MPI_Wtime();

    /*
     * SCATTER: Root distributes 'chunk_size' elements to each process.
     *
     * Key difference from Bcast:
     *   - Bcast sends the ENTIRE array to every process (wasteful)
     *   - Scatter sends ONLY the relevant portion to each process
     *
     * sendcount = chunk_size (elements per process, NOT total N!)
     *
     * Common mistake: setting sendcount to N causes each process
     * to receive N elements, overflowing the local_chunk buffer.
     */
    MPI_Scatter(
        array,       /* sendbuf:   full array on root */
        chunk_size,  /* sendcount: elements sent TO EACH process */
        MPI_INT,
        local_chunk, /* recvbuf:   local buffer for this process */
        chunk_size,  /* recvcount: elements this process receives */
        MPI_INT,
        0,           /* root */
        MPI_COMM_WORLD
    );

    /* Each process sums its local chunk */
    long long local_sum = 0;
    for (int i = 0; i < chunk_size; i++)
        local_sum += local_chunk[i];

    printf("  Rank %d: local_sum of %d elements = %lld\n",
           rank, chunk_size, local_sum);

    /*
     * Collection via Send/Recv (same as Exercise 1, unchanged)
     * Root receives partial sums from all other processes
     */
    if (rank != 0) {
        MPI_Send(&local_sum, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    } else {
        long long total_sum = local_sum;
        for (int r = 1; r < size; r++) {
            long long recv_sum;
            MPI_Recv(&recv_sum, 1, MPI_LONG_LONG, r, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_sum += recv_sum;
        }

        double elapsed = MPI_Wtime() - start;
        long long expected = (long long)N * (N + 1) / 2;
        printf("\n[Scatter] Total sum = %lld\n", total_sum);
        printf("[Scatter] Expected  = %lld\n", expected);
        printf("[Scatter] Correct?  = %s\n", total_sum == expected ? "YES" : "NO");
        printf("[Scatter] Time      = %.4f sec\n", elapsed);
    }

    free(local_chunk);
    if (rank == 0) free(array);
    MPI_Finalize();
    return 0;
}
