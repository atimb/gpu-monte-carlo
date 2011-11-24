#!/bin/bash

echo "Thread number varying (4-512)"

cd ..
file=tests/thread_result_$(date '+%s')
file2=tests/thread_log_$(date '+%s')

for i in `seq 1 128`;
do
    x=$((4*$i))
    echo "Testcase: Threads: $x. Running..."
    ./scintillator --threads=$x --printresult 2>> $file >> $file2
    ./scintillator --threads=$x --printresult 2>> $file >> $file2
    ./scintillator --threads=$x --printresult 2>> $file >> $file2
done
