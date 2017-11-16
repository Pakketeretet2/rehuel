#!/bin/zsh
#
# Runs some tests. Assumes rehuel is compiled and works.
#

METHODS=( implicit_euler
          runge_kutta_4
          lobatto_IIIA_43
	  radau_IIA_32
	  gauss_legendre_43
	  bogacki_shampine_23
	  dormand_prince_54
	  cash_karp_54 )


DTS=( 0.01
      0.01
      0.01
      0.01
      0.01
      0.01
      0.01
      0.01 )

for (( i = 1; i <= ${#METHODS}; ++i ));
do
	METHOD=$METHODS[$i]
	DT=$DTS[$i]
	echo "i = $i, method = "$METHOD" and dt = "$DT > "/dev/stderr"
	../rehuel -m $METHOD -dt $DT > "ode_"$METHOD".dat"
done
