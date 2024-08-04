# Config the environment

```
conda create -n DomainFL python=3.10
```

```
conda activate DomainFL
```

```
pip install -r requirements.txt
```

go to the core directory

```
cd core
```

# Visualization

You can see some visulization result by playing with the 'visualize.ipynb'.

# Basic experiment

run the run0.sh

for the first time, the data (DomainNet) will be downloaded automatically.

you can change the config in the run0.sh file

```
bash run0.sh
```

the result will be saved in results directory.

# Local adaptation

run the local_adaptation.sh, and run_full.sh (this file run the situation of all the local data is used for local adaptation)

```
bash local_adaptation.sh
bash run_full.sh
```

# Compare the results

You can compare the results by runing 'read_result.ipynb' under the 'results' directory.
