!#/bin/bash
for seed in $(seq 1 8)
do
	nohup python3 workdir/fig_builders/FirstPassageFit/FPF_adapt_15_sigma1sqrt2.py "$seed" "1" &> workdir/fig_builders/FirstPassageFit/FPF_adapt_40_center_sigma1.txt &
done

