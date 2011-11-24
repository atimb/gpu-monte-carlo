#!/bin/bash

echo "Repeat number varying (1-20)"

cd ..
file=tests/repeat_result_$(date '+%s')
file2=tests/repeat_log_$(date '+%s')

for i in `seq 1 20`;
do
    x=$((1*$i))
    echo "Testcase: Repeat: $x Running..."
    ./scintillator --repeat=$x --printresult 2>> $file >> $file2
    ./scintillator --repeat=$x --printresult 2>> $file >> $file2
    ./scintillator --repeat=$x --printresult 2>> $file >> $file2
done
