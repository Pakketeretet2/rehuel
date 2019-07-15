#!/bin/sh
#
# Benchmarks some methods on the Lorenz attractor.
#
#

g++ -O3 -I../ -fopenmp -march=native -lrehuel -larmadillo \
    lorenz.cpp -o lorenz

for M in RADAU_IIA_32 RADAU_IIA_53 GAUSS_LEGENDRE_147 LOBATTO_IIIC_43 \
       	CASH_KARP_54 DORMAND_PRINCE_54 BOGACKI_SHAMPINE_32; do
	OPENMP_NUM_THREADS=4 ./lorenz $M > "lorenz_"$M".dat"
done
