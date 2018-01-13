#!/bin/zsh
#
#
#

mi=1
for M in LOBATTO_IIIA_65 \
	LOBATTO_IIIC_65
do
	FILE="err_"$M".dat"
	if [ -f $FILE ]; then
		rm $FILE;
		touch $FILE;
	fi
	for dt in 2e-5 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2 2e-2 5e-2 1e-1 2e-1 5e-1 1 2;
	do
		../rehuel --t0 0.0 --t1 10.0 --dt $dt --method $M >> $FILE
	done
	
	mi=$(( mi + 1 ))
done
