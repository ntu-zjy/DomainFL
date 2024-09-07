#!/bin/bash
# Define constants

bs=32
algo="ours_dp"
ss=50
gr=200
did=0
ien="ViT-B-32"
data=domainnet
sram=("cluster" "random")
ratio=0.1
dps=(0.05 0.1 0.5 1 5 10)
fixdpp=0.1
fixdps=0.05
dpp=(0.1 0.2 0.5 0.8 0.9)

for s in "${dps[@]}"; do
    python "$algo.py" -bs $bs -gr $gr -did $did -ien $ien -d $data -ss $ss -sram "average" -sra 1 -dps $s -dpp $fixdpp

    for m in "${sram[@]}"; do
        python "$algo.py" -bs $bs -gr $gr -did $did -ien $ien -d $data -ss $ss -sram "$m" -sra $ratio -dps $s -dpp $fixdpp
    done
done

for p in "${dpp[@]}"; do
    python "$algo.py" -bs $bs -gr $gr -did $did -ien $ien -d $data -ss $ss -sram "average" -sra 1 -dps $fixdps -dpp $p

    for m in "${sram[@]}"; do
        python "$algo.py" -bs $bs -gr $gr -did $did -ien $ien -d $data -ss $ss -sram "$m" -sra $ratio -dps $fixdps -dpp $p
    done
done





