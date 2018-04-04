#!/bin/zsh
#
#
#

for M in RADAU_IIA_95
do
	./rehuel_orders -m $M > "errs_"$M".dat"
done



