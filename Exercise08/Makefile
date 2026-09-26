# Makefile for SE3082 Lab 07 - MPI Collective Communication
# Student: IT24101022
#
# Usage:
#   make          - compile all 6 programs
#   make run      - run all programs with 4 processes
#   make run2     - run all programs with 2 processes
#   make run8     - run all programs with 8 processes
#   make clean    - remove compiled binaries

CC      = mpicc
CFLAGS  = -O2 -Wall
NP      = 4

TARGETS = sum_bcast sum_scatter sum_gather sum_reduce sum_allreduce sum_scan

.PHONY: all run run2 run8 clean

all: $(TARGETS)

sum_bcast: Exercise02/sum_scatter.c
	@echo "Note: sum_bcast is the given Exercise 1 program (see Readme.md)"
	@echo "      Compiling sum_scatter as Exercise 2 instead"

sum_scatter: Exercise02/sum_scatter.c
	$(CC) $(CFLAGS) -o $@ $<
	@echo "Compiled: $@"

sum_gather: Exercise03/sum_gather.c
	$(CC) $(CFLAGS) -o $@ $<
	@echo "Compiled: $@"

sum_reduce: Exercise04/sum_reduce.c
	$(CC) $(CFLAGS) -o $@ $<
	@echo "Compiled: $@"

sum_allreduce: Exercise05/sum_allreduce.c
	$(CC) $(CFLAGS) -o $@ $<
	@echo "Compiled: $@"

sum_scan: Exercise06/sum_scan.c
	$(CC) $(CFLAGS) -o $@ $<
	@echo "Compiled: $@"

run: all
	@echo ""
	@echo "============================================================"
	@echo " Running all programs with $(NP) processes"
	@echo "============================================================"
	@echo ""
	@echo "--- Exercise 2: sum_scatter ($(NP) processes) ---"
	mpirun -np $(NP) ./sum_scatter
	@echo ""
	@echo "--- Exercise 3: sum_gather ($(NP) processes) ---"
	mpirun -np $(NP) ./sum_gather
	@echo ""
	@echo "--- Exercise 4: sum_reduce ($(NP) processes) ---"
	mpirun -np $(NP) ./sum_reduce
	@echo ""
	@echo "--- Exercise 5: sum_allreduce ($(NP) processes) ---"
	mpirun -np $(NP) ./sum_allreduce
	@echo ""
	@echo "--- Exercise 6: sum_scan ($(NP) processes) ---"
	mpirun -np $(NP) ./sum_scan
	@echo ""
	@echo "============================================================"
	@echo " All programs completed"
	@echo "============================================================"

run2: NP=2
run2: run

run8: NP=8
run8: run

clean:
	rm -f $(TARGETS)
	@echo "Cleaned all binaries"
