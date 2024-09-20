#!/bin/bash
# Define constants

bs=32
algo=("fedavg" "fedproto")
sss=(50 100 150)
gr=200
did=0
ratio=(0.1 0.3)
sram=("cluster" "random")
ien="ViT-B-32"

# # DomainNet
# for ss in "${sss[@]}"; do
#     # baseline.sh
#     for a in "${algo[@]}"; do
#         python "$a.py" -bs $bs -gr $gr -did $did -ien "$ien" -d "domainnet" -ss $ss
#     done

#     # runours.sh
#     python "ours.py" -bs $bs -gr $gr -did $did -ien $ien -d "domainnet" -ss $ss -sram "average" -sra 1
# done

for ss in "${sss[@]}"; do
    for m in "${sram[@]}"; do
        for r in "${ratio[@]}"; do
            python "ours.py" -bs $bs -gr $gr -did $did -ien $ien -d "domainnet" -ss $ss -sram "$m" -sra $r
        done
    done
done

# # PACS
# baseline.sh
# for a in "${algo[@]}"; do
#     python "$a.py" -bs $bs -gr $gr -did $did -ien "convnext_base"  -d "PACS" -ss 7
# done

# runours.sh
# python "ours.py" -bs $bs -gr $gr -did $did -ien "convnext_base" -d "PACS" -ss 7 -sram "average" -sra 1

# for m in "${sram[@]}"; do
#     for r in "${ratio[@]}"; do
#         python "ours.py" -bs $bs -gr $gr -did $did -ien "convnext_base" -d "PACS" -ss 7 -sram "$m" -sra $r
#     done
# done