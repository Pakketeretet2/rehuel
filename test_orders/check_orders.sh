#!/bin/zsh
#
#
#

mi=1
for M in EXPLICIT_EULER       \
	IMPLICIT_EULER        \
	IMPLICIT_MIDPOINT     \
	RUNGE_KUTTA_4         \
	BOGACKI_SHAMPINE_32   \
	CASH_KARP_54          \
	DORMAND_PRINCE_54     \
	FEHLBERG_54           \
	RADAU_IA_32           \
	RADAU_IIA_32	      \
	LOBATTO_IIIA_21	      \
	LOBATTO_IIIC_21	      \
	LOBATTO_IIIA_43	      \
	LOBATTO_IIIC_43	      \
	GAUSS_LEGENDRE_42     \
	RADAU_IA_54           \
	RADAU_IIA_54
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
