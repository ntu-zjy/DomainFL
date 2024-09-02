#!/usr/bin/env bash

data=domainnet

python ours_local_adaptation.py -d $data -ss 50 -fewshot 1 -kdw 0
python ours_local_adaptation.py -d $data -ss 50 -fewshot 2 -kdw 0
python ours_local_adaptation.py -d $data -ss 50 -fewshot 4 -kdw 0
python ours_local_adaptation.py -d $data -ss 50 -fewshot 8 -kdw 0
python ours_local_adaptation.py -d $data -ss 50 -fewshot 16 -kdw 0
python ours_local_adaptation.py -d $data -ss 50 -fewshot 32 -kdw 0
python ours_local_adaptation.py -d $data -ss 50 -fewshot 64 -kdw 0

python ours_local_adaptation.py -d $data -ss 50 -fewshot 1 -kdw 10
python ours_local_adaptation.py -d $data -ss 50 -fewshot 2 -kdw 10
python ours_local_adaptation.py -d $data -ss 50 -fewshot 4 -kdw 10
python ours_local_adaptation.py -d $data -ss 50 -fewshot 8 -kdw 10
python ours_local_adaptation.py -d $data -ss 50 -fewshot 16 -kdw 10
python ours_local_adaptation.py -d $data -ss 50 -fewshot 32 -kdw 10
python ours_local_adaptation.py -d $data -ss 50 -fewshot 64 -kdw 10

python ours_local_adaptation.py -d $data -ss 50 -fewshot 1 -kdw 100
python ours_local_adaptation.py -d $data -ss 50 -fewshot 2 -kdw 100
python ours_local_adaptation.py -d $data -ss 50 -fewshot 4 -kdw 100
python ours_local_adaptation.py -d $data -ss 50 -fewshot 8 -kdw 100
python ours_local_adaptation.py -d $data -ss 50 -fewshot 16 -kdw 100
python ours_local_adaptation.py -d $data -ss 50 -fewshot 32 -kdw 100
python ours_local_adaptation.py -d $data -ss 50 -fewshot 64 -kdw 100

python ours_local_adaptation.py -d $data -ss 50 -fewshot 1 -kdw 1000
python ours_local_adaptation.py -d $data -ss 50 -fewshot 2 -kdw 1000
python ours_local_adaptation.py -d $data -ss 50 -fewshot 4 -kdw 1000
python ours_local_adaptation.py -d $data -ss 50 -fewshot 8 -kdw 1000
python ours_local_adaptation.py -d $data -ss 50 -fewshot 16 -kdw 1000
python ours_local_adaptation.py -d $data -ss 50 -fewshot 32 -kdw 1000
python ours_local_adaptation.py -d $data -ss 50 -fewshot 64 -kdw 1000

python ours_local_adaptation.py -d $data -ss 100 -fewshot 1 -kdw 0
python ours_local_adaptation.py -d $data -ss 100 -fewshot 2 -kdw 0
python ours_local_adaptation.py -d $data -ss 100 -fewshot 4 -kdw 0
python ours_local_adaptation.py -d $data -ss 100 -fewshot 8 -kdw 0
python ours_local_adaptation.py -d $data -ss 100 -fewshot 16 -kdw 0
python ours_local_adaptation.py -d $data -ss 100 -fewshot 32 -kdw 0
python ours_local_adaptation.py -d $data -ss 100 -fewshot 64 -kdw 0

python ours_local_adaptation.py -d $data -ss 100 -fewshot 1 -kdw 10
python ours_local_adaptation.py -d $data -ss 100 -fewshot 2 -kdw 10
python ours_local_adaptation.py -d $data -ss 100 -fewshot 4 -kdw 10
python ours_local_adaptation.py -d $data -ss 100 -fewshot 8 -kdw 10
python ours_local_adaptation.py -d $data -ss 100 -fewshot 16 -kdw 10
python ours_local_adaptation.py -d $data -ss 100 -fewshot 32 -kdw 10
python ours_local_adaptation.py -d $data -ss 100 -fewshot 64 -kdw 10

python ours_local_adaptation.py -d $data -ss 100 -fewshot 1 -kdw 100
python ours_local_adaptation.py -d $data -ss 100 -fewshot 2 -kdw 100
python ours_local_adaptation.py -d $data -ss 100 -fewshot 4 -kdw 100
python ours_local_adaptation.py -d $data -ss 100 -fewshot 8 -kdw 100
python ours_local_adaptation.py -d $data -ss 100 -fewshot 16 -kdw 100
python ours_local_adaptation.py -d $data -ss 100 -fewshot 32 -kdw 100
python ours_local_adaptation.py -d $data -ss 100 -fewshot 64 -kdw 100

python ours_local_adaptation.py -d $data -ss 100 -fewshot 1 -kdw 1000
python ours_local_adaptation.py -d $data -ss 100 -fewshot 2 -kdw 1000
python ours_local_adaptation.py -d $data -ss 100 -fewshot 4 -kdw 1000
python ours_local_adaptation.py -d $data -ss 100 -fewshot 8 -kdw 1000
python ours_local_adaptation.py -d $data -ss 100 -fewshot 16 -kdw 1000
python ours_local_adaptation.py -d $data -ss 100 -fewshot 32 -kdw 1000
python ours_local_adaptation.py -d $data -ss 100 -fewshot 64 -kdw 1000

python ours_local_adaptation.py -d $data -ss 150 -fewshot 1 -kdw 0
python ours_local_adaptation.py -d $data -ss 150 -fewshot 2 -kdw 0
python ours_local_adaptation.py -d $data -ss 150 -fewshot 4 -kdw 0
python ours_local_adaptation.py -d $data -ss 150 -fewshot 8 -kdw 0
python ours_local_adaptation.py -d $data -ss 150 -fewshot 16 -kdw 0
python ours_local_adaptation.py -d $data -ss 150 -fewshot 32 -kdw 0
python ours_local_adaptation.py -d $data -ss 150 -fewshot 64 -kdw 0

python ours_local_adaptation.py -d $data -ss 150 -fewshot 1 -kdw 10
python ours_local_adaptation.py -d $data -ss 150 -fewshot 2 -kdw 10
python ours_local_adaptation.py -d $data -ss 150 -fewshot 4 -kdw 10
python ours_local_adaptation.py -d $data -ss 150 -fewshot 8 -kdw 10
python ours_local_adaptation.py -d $data -ss 150 -fewshot 16 -kdw 10
python ours_local_adaptation.py -d $data -ss 150 -fewshot 32 -kdw 10
python ours_local_adaptation.py -d $data -ss 150 -fewshot 64 -kdw 10

python ours_local_adaptation.py -d $data -ss 150 -fewshot 1 -kdw 100
python ours_local_adaptation.py -d $data -ss 150 -fewshot 2 -kdw 100
python ours_local_adaptation.py -d $data -ss 150 -fewshot 4 -kdw 100
python ours_local_adaptation.py -d $data -ss 150 -fewshot 8 -kdw 100
python ours_local_adaptation.py -d $data -ss 150 -fewshot 16 -kdw 100
python ours_local_adaptation.py -d $data -ss 150 -fewshot 32 -kdw 100
python ours_local_adaptation.py -d $data -ss 150 -fewshot 64 -kdw 100

python ours_local_adaptation.py -d $data -ss 150 -fewshot 1 -kdw 1000
python ours_local_adaptation.py -d $data -ss 150 -fewshot 2 -kdw 1000
python ours_local_adaptation.py -d $data -ss 150 -fewshot 4 -kdw 1000
python ours_local_adaptation.py -d $data -ss 150 -fewshot 8 -kdw 1000
python ours_local_adaptation.py -d $data -ss 150 -fewshot 16 -kdw 1000
python ours_local_adaptation.py -d $data -ss 150 -fewshot 32 -kdw 1000
python ours_local_adaptation.py -d $data -ss 150 -fewshot 64 -kdw 1000