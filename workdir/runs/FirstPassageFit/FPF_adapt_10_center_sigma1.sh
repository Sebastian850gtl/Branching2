!#/bin/bash
for seed in $(seq 1 10)
do
	nohup python3 workdir/fig_builders/FirstPassageFit/FPF_adapt_10_center_sigma1.py "$seed" "1" &> workdir/fig_builders/FirstPassageFit/FPF_adapt_40_center_sigma1.txt &
done

