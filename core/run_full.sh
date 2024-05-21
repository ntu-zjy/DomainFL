#!/usr/bin/env bash

python ours_full_local_adaptation.py -d data -ss 50 -kdw 0
python ours_full_local_adaptation.py -d data -ss 50 -kdw 10
python ours_full_local_adaptation.py -d data -ss 50 -kdw 100
python ours_full_local_adaptation.py -d data -ss 50 -kdw 1000

python ours_full_local_adaptation.py -d data -ss 100 -kdw 0
python ours_full_local_adaptation.py -d data -ss 100 -kdw 10
python ours_full_local_adaptation.py -d data -ss 100 -kdw 100
python ours_full_local_adaptation.py -d data -ss 100 -kdw 1000

python ours_full_local_adaptation.py -d data -ss 150 -kdw 0
python ours_full_local_adaptation.py -d data -ss 150 -kdw 10
python ours_full_local_adaptation.py -d data -ss 150 -kdw 100
python ours_full_local_adaptation.py -d data -ss 150 -kdw 1000
