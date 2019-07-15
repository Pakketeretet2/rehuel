#!/bin/sh
#
# Benchmarks some methods on the brusselator problem.
#
#

g++ -O3 -I../ -fopenmp -march=native -lrehuel -larmadillo \
    robertson.cpp -o robertson

# There is no point in trying to solve a Robertson reaction
# with a non-stiff solver, you will just blow out your RAM
for M in RADAU_IIA_32 RADAU_IIA_53 RADAU_IIA_74 \
	LOBATTO_IIIA_43 LOBATTO_IIIC_43
do
	OPENMP_NUM_THREADS=4 ./robertson $M > "robertson_"$M".dat"
done
