/*
 * Exercise 01: Broadcast + Send/Recv (Given Program)
 *
 * Student: IT24101022
 * Subject: SE3082 - Parallel Computing
 * Lab: 07 - MPI Collective Communication
 *
 * Strategy:
 *   - Root fills the full array and broadcasts it to ALL processes
 *   - Every process allocates the FULL array (memory inefficient)
 *   - Each process sums its own portion
 *   - Non-root processes send partial sums to root via MPI_Send
 *   - Root collects results using MPI_Recv loop
 *
 * Things to note:
 *   - Every process allocates N elements even though it only needs N/P
 *   - The Send/Recv loop on root is O(P) steps
 *   - All processes MUST call MPI_Bcast (not just root) or deadlock occurs
 *
 * Compile: mpicc -o sum_bcast sum_bcast.c
 * Run:     mpirun -np 4 ./sum_bcast
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

    /*
     * Every process allocates the FULL array.
     * This is the downside of broadcast — wastes memory.
     */
    int *array = (int *)malloc(N * sizeof(int));

    /* Root fills the array with values 1 to N */
    if (rank == 0) {
        for (int i = 0; i < N; i++)
            array[i] = i + 1;
        printf("Root filled array with values 1 to %d\n", N);
    }

    double start = MPI_Wtime();

    /*
     * BROADCAST: Root sends the entire array to ALL processes.
     * Every process must call this — not just root!
     * After this call, every process has a complete copy of array[].
     */
    MPI_Bcast(array, N, MPI_INT, 0, MPI_COMM_WORLD);

    /*
     * Each process computes the sum of its own portion.
     * Process i handles indices [i * chunk_size .. (i+1) * chunk_size - 1]
     */
    int chunk_size = N / size;
    int start_idx  = rank * chunk_size;
    int end_idx    = start_idx + chunk_size;

    long long local_sum = 0;
    for (int i = start_idx; i < end_idx; i++)
        local_sum += array[i];

    printf("  Rank %d: summed indices [%d, %d) => local_sum = %lld\n",
           rank, start_idx, end_idx, local_sum);

    /* Collect results using point-to-point communication */
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
        printf("\n[Bcast] Total sum   = %lld\n", total_sum);
        printf("[Bcast] Expected    = %lld\n", expected);
        printf("[Bcast] Correct?    = %s\n", total_sum == expected ? "YES" : "NO");
        printf("[Bcast] Time        = %.4f sec\n", elapsed);
    }

    free(array);
    MPI_Finalize();
    return 0;
}
