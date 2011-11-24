#!/bin/bash

echo "Photon source energy varying (0.1-10MeV)"

cd ..
file=tests/energy_result_$(date '+%s')
file2=tests/energy_log_$(date '+%s')

for i in `seq 1 10`;
do
    x=`echo "scale=2; $i*0.01" | bc`
    echo "Testcase: Energy: $x Running..."
    ./scintillator --energy=$x --printresult 2>> $file >> $file2
    ./scintillator --energy=$x --printresult 2>> $file >> $file2
    ./scintillator --energy=$x --printresult 2>> $file >> $file2
done
