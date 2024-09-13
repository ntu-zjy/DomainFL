#!/bin/bash
# Define constants

bs=32
algo="ours"
sss=(50 100 150)
gr=200
did=0
ien="ViT-B-32"
data=domainnet
sram="mixed"
ratio=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# ratio=(0.5 0.6 0.7 0.8 0.9)

for ss in "${sss[@]}"; do
    for r in "${ratio[@]}"; do
        python "$algo.py" -bs $bs -gr $gr -did $did -ien $ien -d $data -ss $ss -sram "$sram" -sra $r
    done
done





