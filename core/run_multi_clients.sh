#!/bin/bash
# Define constants

bs=32
algo=("local" "fedavg" "fedprox" "fedditto" "fedmoon" "fedproto" "fedavgDBE" "ours")
ss=50
gr=200
did=0
ien="ViT-B-32"
sram=("cluster" "random")
ratio=(0.1 0.3)
split_num=(3 4 5)
data=domainnet

for sn in "${split_num[@]}"; do

    for a in "${algo[@]}"; do
        python "$a.py" -bs $bs -gr $gr -did $did -ien "$ien" -d $data -split_num $sn -ss $ss
    done


    python "ours.py" -bs $bs -gr $gr -did $did -ien $ien -d $data -ss $ss -sram "average" -sra 1 -split_num $sn

    for m in "${sram[@]}"; do
        for r in "${ratio[@]}"; do
            python "ours.py" -bs $bs -gr $gr -did $did -ien $ien -d $data -ss $ss -sram "$m" -sra $r -split_num $sn
        done
    done

done
