#!/bin/bash
# Define constants

bs=32
algo="ours_mixed"
ss=50
gr=200
did=0
ien="ViT-B-32"
data=domainnet
sram=("cluster" "random")
ratio=(0.1 0.3)
mixed_ratios=(0.3 0.4 0.5)

for mr in "${mixed_ratios[@]}"; do
    python "$algo.py" -bs $bs -gr $gr -did $did -ien $ien -d $data -ss $ss -sram "average" -sra 1 -mr $mr

    for m in "${sram[@]}"; do
        for r in "${ratio[@]}"; do
            python "$algo.py" -bs $bs -gr $gr -did $did -ien $ien -d $data -ss $ss -sram "$m" -sra $r -mr $mr
        done
    done
done




