#!/bin/zsh
#
#
#

for M in LOBATTO_IIIA_127 # GAUSS_LEGENDRE_{42,147} RADAU_IIA_{32,53,95,137}
do
	./rehuel_orders -m $M > "errs_"$M".dat"
done



