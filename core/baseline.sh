#!/bin/bash
# Define constants

bs=32
algo=("local" "zeroshot" "fedavg" "fedprox" "fedditto" "fedmoon" "fedproto" "fedavgDBE")
sss=(50 100)
gr=200
did=0
ien="ViT-B-32"
data=domainnet

# python fedavg.py -bs 32 -gr 200 -did 0 -ien "ViT-B-32" -d domainnet -ss 50
for ss in "${sss[@]}"; do
    for a in "${algo[@]}"; do
        python "$a.py" -bs $bs -gr $gr -did $did -ien "$ien" -d $data -ss $ss
    done
done


