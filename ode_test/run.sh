#!/bin/zsh
#
#

METHODS=( implicit_euler
          runge_kutta_4
          lobatto_IIIA_43
	  radau_IIA_32
	  gauss_legendre_65 )

DTS=( 0.001
      0.0001
      0.01
      0.01
      0.01)

for (( i = 1; i <= ${#METHODS}; ++i ));
do
	METHOD=$METHODS[$i]
	DT=$DTS[$i]
	echo "i = $i, method = "$METHOD" and dt = "$DT > "/dev/stderr"
	../irk -m $METHOD -dt $DT > "ode_"$METHOD".dat"
done
