#!/bin/bash
# Define constants

bs=32
algo="ours"
# sss=(50 100 150)
sss=(50)
gr=200
did=0
ien="RN50x4"
data=domainnet
sram=("cluster" "random")
ratio=(0.1 0.3)

for ss in "${sss[@]}"; do
    python "$algo.py" -bs $bs -gr $gr -did $did -ien $ien -d $data -ss $ss -sram "average" -sra 1
    for m in "${sram[@]}"; do
        for r in "${ratio[@]}"; do
            python "$algo.py" -bs $bs -gr $gr -did $did -ien $ien -d $data -ss $ss -sram "$m" -sra $r
        done
    done
done



