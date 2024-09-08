#!/bin/bash
# Define constants

bs=32
algo=("local" "zeroshot" "fedavg" "fedprox" "fedditto" "fedmoon" "fedproto" "fedavgDBE" "ours")
ss=50
gr=200
did=0
ien="ViT-B-32"

for a in "${algo[@]}"; do
    python "$a.py" -bs $bs -gr $gr -did $did -ien "$ien" -d data
done


