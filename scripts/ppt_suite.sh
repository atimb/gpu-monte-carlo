#!/bin/bash

echo "Photons/thread number varying (1-100)"

cd ..
file=tests/ppt_result_$(date '+%s')
file2=tests/ppt_log_$(date '+%s')

for i in `seq 1 120`;
do
    x=$((1*$i))
    echo "Testcase: Photons/thread: $x Running..."
    ./scintillator --photonsperthread=$x --printresult 2>> $file >> $file2
    ./scintillator --photonsperthread=$x --printresult 2>> $file >> $file2
    ./scintillator --photonsperthread=$x --printresult 2>> $file >> $file2
done
