/*
 * Exercise 06: Prefix Sums with MPI_Scan
 *
 * Student: IT24101022
 * Subject: SE3082 - Parallel Computing
 * Lab: 07 - MPI Collective Communication
 *
 * Strategy:
 *   - MPI_Scan computes prefix (cumulative) sum across all processes
 *   - Each process gets a DIFFERENT result (unlike Allreduce)
 *   - Rank k receives: sum of local_sums from rank 0 through rank k
 *   - Last rank's prefix_sum == total global sum
 *
 * Compile: mpicc -o sum_scan sum_scan.c
 * Run:     mpirun -np 4 ./sum_scan
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
     * SCAN (Prefix Reduction):
     *
     * After MPI_Scan with MPI_SUM:
     *   Rank 0: prefix_sum = local_sum[0]
     *   Rank 1: prefix_sum = local_sum[0] + local_sum[1]
     *   Rank 2: prefix_sum = local_sum[0] + local_sum[1] + local_sum[2]
     *   Rank k: prefix_sum = sum of all local_sums from rank 0..k
     *
     * Each process gets a DIFFERENT value — unlike Allreduce!
     *
     * Practical use: global offsets, cumulative distributions,
     *                global index assignment, load-balanced work splitting.
     */
    long long prefix_sum = 0;
    MPI_Scan(
        &local_sum,    /* sendbuf */
        &prefix_sum,   /* recvbuf: cumulative result up to this rank */
        1,             /* count */
        MPI_LONG_LONG,
        MPI_SUM,       /* operation */
        MPI_COMM_WORLD
    );

    /*
     * sum_before_me = total sum of all chunks BEFORE this process.
     * This is the global offset for this process's work.
     *
     * Practical example:
     *   If writing output to a global array, sum_before_me tells
     *   where to start writing without any extra communication.
     */
    long long sum_before_me = prefix_sum - local_sum;

    /* Verification using formula: sum(1..K) = K*(K+1)/2 */
    long long K = (long long)(rank + 1) * chunk_size;
    long long expected_prefix = K * (K + 1) / 2;
    int prefix_correct = (prefix_sum == expected_prefix);

    printf("  Rank %d: local_sum = %lld | prefix_sum = %lld | sum_before_me = %lld | Verify: %s\n",
           rank, local_sum, prefix_sum, sum_before_me,
           prefix_correct ? "CORRECT" : "WRONG");

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        double elapsed = MPI_Wtime() - start;
        printf("\n[Scan] Time = %.4f sec\n", elapsed);
        free(array);
    }

    /* Last rank's prefix_sum = global total */
    if (rank == size - 1) {
        long long expected_total = (long long)N * (N + 1) / 2;
        printf("[Scan] Last rank prefix_sum = %lld (expected %lld) => %s\n",
               prefix_sum, expected_total,
               prefix_sum == expected_total ? "CORRECT" : "WRONG");
    }

    free(local_chunk);
    MPI_Finalize();
    return 0;
}
