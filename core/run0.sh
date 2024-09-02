#!/bin/bash
# Define constants

bs=32
algo=("local" "zeroshot" "fedavg" "fedprox" "fedditto" "fedmoon" "fedproto" "fedavgDBE" "ours")
ss=100
gr=200
did=3
ien="ViT-B-16"
data=domainnet

for a in "${algo[@]}"; do
    python "$a.py" -bs $bs -gr $gr -did $did -ien "$ien" -d $data -ss $ss
done


