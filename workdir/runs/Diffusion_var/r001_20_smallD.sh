!#/bin/bash
for seed in $(seq 1 61)
do
	nohup python3 workdir/fig_builders/CumSumdt/CS_dtmindtmax_0.005_0.07.py "$seed" "1" &> workdir/fig_builders/CumsSumdt/out005.txt &
done
