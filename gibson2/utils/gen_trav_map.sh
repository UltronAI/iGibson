declare scenes=("Beechwood_1_int" "Benevolence_0_int" "Ihlen_0_int" "Ihlen_1_int" "Pomaria_0_int" "Wainscott_1_int" "Beechwood_0_int" "Benevolence_1_int" "Benevolence_2_int" "Pomaria_1_int" "Pomaria_2_int" "Wainscott_0_int")
for file in "${scenes[@]}"
do
    echo $file
    python generate_trav_map.py $(basename $file)
done