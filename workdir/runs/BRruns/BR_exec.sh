#!/bin/sh
# a line of text describing what this task does
for i in 20 ; do
    nohup python3 workdir/runs/BRruns/BR_core.py "BR_mono_100_01" "$i" "1000" &> output.txt &
done