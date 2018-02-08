#!/bin/zsh
#
#
#

if [ $# -eq 0 ]; then
	METHODS=$( ../rehuel --print-all-methods )
else
	FIRST_CHAR=$( echo $1 | cut -c1 )
	if [ "$FIRST_CHAR" = "-" ]; then
		# Do nothing
		METHODS=$( ../rehuel --print-all-methods )
	else
		METHODS=$1
		shift
	fi
fi

OTHER_OPTS="$@"
echo "Other options are "$OTHER_OPTS
for M in $( echo $METHODS )
do
	FILE="err_"$M".dat"
	if [ -f $FILE ]; then
		rm $FILE;
		touch $FILE;
	fi
	for dt in 5e-4 1e-3 2e-3 5e-3 1e-2 2e-2 5e-2 1e-1 2e-1 5e-1 1 2;
	do
		../rehuel --t0 0.0 --t1 100.0 --dt $dt --method $M \
			--test-exponential $OTHER_OPTS >> $FILE
	done
	
	mi=$(( mi + 1 ))
done
