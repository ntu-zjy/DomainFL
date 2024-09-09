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

python "$algo.py" -bs $bs -gr $gr -did $did -ien $ien -d $data -ss $ss -sram "average" -sra 1

for m in "${sram[@]}"; do
    for r in "${ratio[@]}"; do
        python "$algo.py" -bs $bs -gr $gr -did $did -ien $ien -d $data -ss $ss -sram "$m" -sra $r
    done
done



