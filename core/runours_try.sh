#!/bin/bash
# Define constants

bs=32
algo="ours"
# sss=(50 100 150)
sss=(50)
gr=200
did=0
ien="RN50"
data=domainnet
sram=("cluster" "random")
# ratio=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
ratio=(1.0)

for ss in "${sss[@]}"; do
    # python "$algo.py" -bs $bs -gr $gr -did $did -ien $ien -d $data -ss $ss -sram "average" -sra 1

    for m in "${sram[@]}"; do
        for r in "${ratio[@]}"; do
            python "$algo.py" -bs $bs -gr $gr -did $did -ien $ien -d $data -ss $ss -sram "$m" -sra $r
        done
    done
done



