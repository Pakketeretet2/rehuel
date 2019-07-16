#!/bin/zsh
#
# Runs the Rehuel example.
#

OUTPUT_DIR=output
mkdir -p $OUTPUT_DIR

METHODS=(RADAU_IIA_53 LOBATTO_IIIC_85 DORMAND_PRINCE_54 BOGACKI_SHAMPINE_32)

for METHOD in $METHODS
do
	OUTPUT=$OUTPUT_DIR"/stiff_equation_"$METHOD".dat"
	./rehuel_example --method $METHOD --equation stiff-equation > $OUTPUT
done


for METHOD in $METHODS
do
	OUTPUT=$OUTPUT_DIR"/three_body_"$METHOD".dat"
	./rehuel_example --method $METHOD --equation three-body > $OUTPUT
done

for METHODS in $METHODS
do
	OUTPUT=$OUTPUT_DIR"/van_der_pol_"$METHOD".dat"
	./rehuel_example --method $METHOD --equation van-der-pol > $OUTPUT
done



