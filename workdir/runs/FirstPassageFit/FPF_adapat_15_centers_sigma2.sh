#!/bin/bash
for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
        nohup python3 workdir/fig_builders/FirstPassageFit/FPF_adapt_15_center_sigma2.py "$seed" "1" &> out.txt &
done

