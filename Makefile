# Makefile - SE3082 Lab 07: MPI Collective Communication
# Student: IT24101022
#
# Compile:  make
# Run (4P): make run
# Run (2P): make run NP=2
# Run (8P): make run NP=8
# Clean:    make clean

CC     = mpicc
CFLAGS = -O2 -Wall
NP     = 4

TARGETS = sum_bcast sum_scatter sum_gather sum_reduce sum_allreduce sum_scan

.PHONY: all run clean

all: $(TARGETS)

sum_bcast: Exercise01/sum_bcast.c
	$(CC) $(CFLAGS) -o $@ $<

sum_scatter: Exercise02/sum_scatter.c
	$(CC) $(CFLAGS) -o $@ $<

sum_gather: Exercise03/sum_gather.c
	$(CC) $(CFLAGS) -o $@ $<

sum_reduce: Exercise04/sum_reduce.c
	$(CC) $(CFLAGS) -o $@ $<

sum_allreduce: Exercise05/sum_allreduce.c
	$(CC) $(CFLAGS) -o $@ $<

sum_scan: Exercise06/sum_scan.c
	$(CC) $(CFLAGS) -o $@ $<

run: all
	@echo "=== Exercise 1: sum_bcast ($(NP) procs) ==="
	mpirun -np $(NP) ./sum_bcast
	@echo ""
	@echo "=== Exercise 2: sum_scatter ($(NP) procs) ==="
	mpirun -np $(NP) ./sum_scatter
	@echo ""
	@echo "=== Exercise 3: sum_gather ($(NP) procs) ==="
	mpirun -np $(NP) ./sum_gather
	@echo ""
	@echo "=== Exercise 4: sum_reduce ($(NP) procs) ==="
	mpirun -np $(NP) ./sum_reduce
	@echo ""
	@echo "=== Exercise 5: sum_allreduce ($(NP) procs) ==="
	mpirun -np $(NP) ./sum_allreduce
	@echo ""
	@echo "=== Exercise 6: sum_scan ($(NP) procs) ==="
	mpirun -np $(NP) ./sum_scan

clean:
	rm -f $(TARGETS)
