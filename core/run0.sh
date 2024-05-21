#!/bin/bash
# Define constants

bs=32
algo=("local" "zeroshot" "fedavg" "fedprox" "fedditto" "fedmoon" "fedproto" "fedavgDBE" "ours")
ss=10
gr=200
did=0
ien="convnext_base"

for a in "${algo[@]}"; do
    python "$a.py" -bs $bs -gr $gr -did $did -ien "$ien" -d office
done


