# Config the environment

Create the environment.

```
conda create -n DomainFL python=3.11
```

Activate the environment.

```
conda activate DomainFL
```

Install the required repository.

```
pip install -r requirements.txt
```

Go to the core file directory.

```
cd core
```

# Visualization

You can see some visulization result in our paper by playing with the 'visualize.ipynb'.

# Basic experiment

run the **runours.sh** to check MPFT results.

for the first time, the data (DomainNet) will be downloaded automatically.

you can change the config in the run0.sh file

```
bash run0.sh
```

the result will be saved in results directory.

run the **baseline.sh** to see the results of baselines.

```
bash baseline.sh
```

# Local adaptation

switch the branch on `local_adaptation`

```
cd ..
```

run the local_adaptation.sh, and run_full.sh (this file run the situation of all the local data is used for local adaptation)

```
bash local_adaptation.sh
bash run_full.sh
```

# Compare the results

You can compare the results by runing 'read_result.ipynb' under the 'results' directory.
