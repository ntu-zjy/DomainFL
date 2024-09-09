#!/bin/bash
# Define constants

bs=32
algo=("local_mixed" "fedavg_mixed" "fedprox_mixed" "fedditto_mixed" "fedmoon_mixed" "fedproto_mixed" "fedavgDBE_mixed")
ss=50
gr=200
did=1
ien="ViT-B-32"
data=domainnet
mixed_ratios=(0.3 0.4 0.5)

for mr in "${mixed_ratios[@]}"; do
    for a in "${algo[@]}"; do
        python "$a.py" -bs $bs -gr $gr -did $did -ien "$ien" -d $data -ss $ss -mr $mr
    done
done



