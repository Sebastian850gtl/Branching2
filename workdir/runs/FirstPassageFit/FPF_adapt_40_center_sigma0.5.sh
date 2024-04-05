#!/bin/bash
for seed in $(seq 1 4)
do
	nohup python3 workdir/fig_builders/FirstPassageFit/FPF_adapt_40_center_sigma0.5.py "$seed" "1" &> out.txt &
done

