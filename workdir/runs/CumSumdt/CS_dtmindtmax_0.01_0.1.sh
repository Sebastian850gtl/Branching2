!#/bin/bash
for seed in $(seq 1 20)
do
	nohup python3 workdir/fig_builders/CumSumdt/CS_dtmindtmax_0.01_0.1.py "$seed" "1" &> out00101.txt &
done
