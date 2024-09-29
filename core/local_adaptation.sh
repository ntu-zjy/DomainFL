#!/usr/bin/env bash

data=domainnet
sss=(50 100 150)
fewshots=(1 2 4 8 16 32 64)
kdws=(0 10 100 1000)

for ss in "${sss[@]}"; do
    for kdw in "${kdws[@]}"; do
        python ours_full_adaptation.py -d $data -ss $ss -kdw $kdw
    done
done

for ss in "${sss[@]}"; do
    for fewshot in "${fewshots[@]}"; do
        for kdw in "${kdws[@]}"; do
            python ours_local_adaptation.py -d $data -ss $ss -fewshot $fewshot -kdw $kdw
        done
    done
done