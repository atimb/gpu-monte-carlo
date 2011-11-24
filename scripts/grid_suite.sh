#!/bin/bash

echo "Grid number varying (1-1024)"

cd ..
file=tests/grid_result_$(date '+%s')
file2=tests/grid_log_$(date '+%s')

for i in `seq 1 128`;
do
    x=$((8*$i))
    echo "Testcase: Threads: $x. Running..."
    ./scintillator --grids=$x --printresult 2>> $file >> $file2
    ./scintillator --grids=$x --printresult 2>> $file >> $file2
    ./scintillator --grids=$x --printresult 2>> $file >> $file2
done
