#!/bin/bash
# Define constants

bs=32
algo=("fedditto")
ss=50
gr=200
did=0
ien="ViT-B-32"
sram=("cluster" "random")
ratio=(0.1 0.3)
split_num=(4 5)
data=domainnet

for sn in "${split_num[@]}"; do

    for a in "${algo[@]}"; do
        python "$a.py" -bs $bs -gr $gr -did $did -ien "$ien" -d $data -split_num $sn -ss $ss
    done

done
