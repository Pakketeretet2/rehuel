#!/bin/zsh
#
# Runs the Rehuel example.
#

OUTPUT_DIR=output
mkdir -p $OUTPUT_DIR

#METHODS=(RADAU_IIA_53 RADAU_IIA_95 LOBATTO_IIIC_43 LOBATTO_IIIC_85 DORMAND_PRINCE_54 FEHLBERG_54 BOGACKI_SHAMPINE_32)
METHODS=(LOBATTO_IIIC_43 LOBATTO_IIIC_85 DORMAND_PRINCE_54)  # FEHLBERG_54 BOGACKI_SHAMPINE_32)

for METHOD in ${METHODS[@]}
do
	echo " ==> stiff equation, $METHOD"
	OUTPUT=$OUTPUT_DIR"/stiff_equation_"$METHOD".dat"
	./rehuel_example --method $METHOD --equation stiff-equation > $OUTPUT
done


#for METHOD in DORMAND_PRINCE_54 FEHLBERG_54 BOGACKI_SHAMPINE_32
#do
#	OUTPUT=$OUTPUT_DIR"/three_body_"$METHOD".dat"
#	./rehuel_example --method $METHOD --equation three-body > $OUTPUT
#done

for METHOD in ${METHODS[@]}
do
	echo " ==> Van der Pol equation, $METHOD"
	OUTPUT=$OUTPUT_DIR"/van_der_pol_"$METHOD".dat"
	data_out=$OUTPUT_DIR"/van_der_pol_"$METHOD"_data.dat"
	./rehuel_example --method $METHOD --equation van-der-pol --output-file $data_out  > $OUTPUT
done

for METHOD in RADAU_IIA_53 RADAU_IIA_95 LOBATTO_IIIC_43 LOBATTO_IIIC_85;
do
	echo " ==> Robertson equation, $METHOD"
	OUTPUT=$OUTPUT_DIR"/robertson_"$METHOD".dat"
	data_out=$OUTPUT_DIR"/roberton_"$METHOD"_data.dat"
	./rehuel_example --method $METHOD --equation robertson --out-interval 1e10 --time-span 0 1e12 --output-file $data_out > $OUTPUT
done
