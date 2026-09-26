/*
 * Exercise 03: Replace Send/Recv with Gather
 *
 * Student: IT24101022
 * Subject: SE3082 - Parallel Computing
 * Lab: 07 - MPI Collective Communication
 *
 * Strategy:
 *   - MPI_Scatter distributes chunks (from Exercise 2)
 *   - MPI_Gather collects local_sum from all processes to root
 *   - Root sums the gathered values (manual loop still needed)
 *
 * Compile: mpicc -o sum_gather sum_gather.c
 * Run:     mpirun -np 4 ./sum_gather
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

    /* Only root allocates full array (from Exercise 2) */
    int *array = NULL;
    if (rank == 0) {
        array = (int *)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++)
            array[i] = i + 1;
        printf("Root filled array with values 1 to %d\n", N);
    }

    int *local_chunk = (int *)malloc(chunk_size * sizeof(int));

    double start = MPI_Wtime();

    /* Scatter: same as Exercise 2 */
    MPI_Scatter(array, chunk_size, MPI_INT,
                local_chunk, chunk_size, MPI_INT,
                0, MPI_COMM_WORLD);

    /* Each process computes its local sum */
    long long local_sum = 0;
    for (int i = 0; i < chunk_size; i++)
        local_sum += local_chunk[i];

    printf("  Rank %d: local_sum = %lld\n", rank, local_sum);

    /*
     * GATHER: Collect one long long from each process into 'all_sums' on root.
     *
     * MPI_Gather collects values in rank order:
     *   all_sums[0] = local_sum from rank 0
     *   all_sums[1] = local_sum from rank 1
     *   ...
     *   all_sums[size-1] = local_sum from rank size-1
     *
     * Replaces the entire Send/Recv loop from Exercise 1 & 2.
     *
     * Note: Gather does NOT compute — root still sums manually.
     */
    long long *all_sums = NULL;
    if (rank == 0)
        all_sums = (long long *)malloc(size * sizeof(long long));

    MPI_Gather(
        &local_sum,  /* sendbuf:   what each process contributes */
        1,           /* sendcount: each process sends 1 value */
        MPI_LONG_LONG,
        all_sums,    /* recvbuf:   array to hold all values (root only) */
        1,           /* recvcount: elements received FROM EACH process */
        MPI_LONG_LONG,
        0,           /* root */
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        long long total_sum = 0;
        for (int r = 0; r < size; r++) {
            printf("  all_sums[%d] = %lld\n", r, all_sums[r]);
            total_sum += all_sums[r];
        }

        double elapsed = MPI_Wtime() - start;
        long long expected = (long long)N * (N + 1) / 2;
        printf("\n[Gather] Total sum = %lld\n", total_sum);
        printf("[Gather] Expected  = %lld\n", expected);
        printf("[Gather] Correct?  = %s\n", total_sum == expected ? "YES" : "NO");
        printf("[Gather] Time      = %.4f sec\n", elapsed);

        free(all_sums);
        free(array);
    }

    free(local_chunk);
    MPI_Finalize();
    return 0;
}
