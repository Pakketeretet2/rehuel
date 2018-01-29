#!/bin/zsh
#
#

for order in 1 2 3 4 5 6
do
	FILE="bdf_"$order".dat"
	if [ -f $FILE ]; then
		rm $FILE
		touch $FILE
	fi
	for dt in 1e-3 2e-3 5e-3 1e-2 2e-2 5e-2 1e-1 2e-1;
	do
		../rehuel --t0 0.0 --t1 1000.0 --dt $dt --method BDF \
			  --multistep-order $order --test-multistep >> $FILE
	done
done
