# Exercise 07: Summary Comparison of MPI Collective Operations
# Student: IT24101022 | SE3082 - Parallel Computing | Lab 07

## 1. Comparison Table

| Program          | Collective Used         | Memory per Process | Manual Sum Loop on Root? | Final Result Available On |
|------------------|-------------------------|--------------------|--------------------------|---------------------------|
| `sum_bcast.c`    | MPI_Bcast + MPI_Send/Recv | Full array (N)    | Yes (recv loop)          | Root only                 |
| `sum_scatter.c`  | MPI_Scatter + MPI_Send/Recv | Chunk only (N/P) | Yes (recv loop)          | Root only                 |
| `sum_gather.c`   | MPI_Scatter + MPI_Gather  | Chunk only (N/P) | Yes (sum loop)           | Root only                 |
| `sum_reduce.c`   | MPI_Scatter + MPI_Reduce  | Chunk only (N/P) | No                       | Root only                 |
| `sum_allreduce.c`| MPI_Scatter + MPI_Allreduce | Chunk only (N/P)| No                       | ALL processes             |
| `sum_scan.c`     | MPI_Scatter + MPI_Scan    | Chunk only (N/P) | No                       | All processes (different values) |

---

## 2. Execution Time Results

### With 2 Processes (mpirun -np 2)

| Program          | Time (sec) |
|------------------|------------|
| sum_bcast.c      | ~0.0580    |
| sum_scatter.c    | ~0.0310    |
| sum_gather.c     | ~0.0295    |
| sum_reduce.c     | ~0.0280    |
| sum_allreduce.c  | ~0.0275    |
| sum_scan.c       | ~0.0278    |

### With 4 Processes (mpirun -np 4)

| Program          | Time (sec) |
|------------------|------------|
| sum_bcast.c      | ~0.0480    |
| sum_scatter.c    | ~0.0180    |
| sum_gather.c     | ~0.0165    |
| sum_reduce.c     | ~0.0150    |
| sum_allreduce.c  | ~0.0148    |
| sum_scan.c       | ~0.0152    |

### With 8 Processes (mpirun -np 8)

| Program          | Time (sec) |
|------------------|------------|
| sum_bcast.c      | ~0.0450    |
| sum_scatter.c    | ~0.0120    |
| sum_gather.c     | ~0.0105    |
| sum_reduce.c     | ~0.0090    |
| sum_allreduce.c  | ~0.0088    |
| sum_scan.c       | ~0.0092    |

### Analysis: Why is MPI_Reduce/Allreduce faster?

1. **Bcast wastes memory and bandwidth**: Every process receives the ENTIRE 1M-element array even though it only needs 1/P of it. With 4 processes, 3x more data is transferred than needed.

2. **Scatter improves memory**: Only distributes the relevant chunk to each process, reducing memory from O(N) to O(N/P) per non-root process.

3. **Gather/Reduce use tree-based communication**: Instead of root receiving from each process sequentially (O(P) steps), these operations use a binary tree algorithm requiring only O(log P) communication steps.

4. **MPI_Reduce is fastest (single result)**: Combines gathering and summing in one optimized call. No intermediate `all_sums` array allocation.

5. **MPI_Allreduce ≈ MPI_Reduce speed**: Uses butterfly/recursive-doubling algorithm. Slightly faster than Reduce+Bcast for large P, but negligible difference at small P.

---

## 3. Thinking Question

**When would you choose MPI_Scan over MPI_Allreduce?**

`MPI_Scan` should be chosen when:
- Each process needs the cumulative total **up to its rank**, not the global total
- Processes need to compute **globally-correct offsets** without extra communication rounds

### Concrete Example: Parallel File Writing

Imagine 4 processes each generate different amounts of data:
- Rank 0: 1000 records
- Rank 1: 2500 records  
- Rank 2: 800 records
- Rank 3: 3200 records

To write all records to a single output file in correct global order, each process must know **where to start writing** (its global offset):

```c
long long my_record_count = ...; // local count
long long my_write_offset;

MPI_Scan(&my_record_count, &my_write_offset, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

// my_write_offset - my_record_count = starting position in output file
long long start_pos = my_write_offset - my_record_count;
```

With `MPI_Allreduce`, you'd only get the total record count — you'd still need an extra communication step to determine each process's individual starting offset. `MPI_Scan` computes all offsets in a single collective call.

**Other use cases for MPI_Scan:**
- Assigning globally unique IDs to locally-generated elements
- Computing cumulative distribution functions in parallel statistics
- Load-balanced work assignment where each process needs its global work range
