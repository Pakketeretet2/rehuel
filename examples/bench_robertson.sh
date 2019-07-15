#!/bin/sh
#
# Benchmarks some methods on the brusselator problem.
#
#

g++ -O3 -I../ -fopenmp -march=native -lrehuel -larmadillo \
    robertson.cpp -o robertson

for M in RADAU_IIA_32 RADAU_IIA_53 RADAU_IIA_137 CASH_KARP_54 \
		      DORMAND_PRINCE_54 BOGACKI_SHAMPINE_32; do
	OPENMP_NUM_THREADS=4 ./robertson $M > "robertson_"$M".dat"
done
