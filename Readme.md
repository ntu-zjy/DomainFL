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

# Visualization (appendix A)

You can see some visulization result in our paper by playing with the 'visualize.ipynb'.

# Basic experiment (section 5.1 and 5.2)

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

# Mixed domain in one client (section 5.3)

switch the branch to `mixed`

```
# switch to the root dir of project
cd ..
git checkout mixed
```

then follow the instrcution in `mixed` branch

# Local adaptation (section 5.4)

switch the branch to `local_adaptation`

```
# make sure you are in the root dir of project
git checkout local_adaptation
```

# Computation and communication cost (section 5.5)

switch the branch to `cost`

```
# make sure you are in the root dir of project
git checkout cost
```

# Differential privacy (section 5.6)

switch the branch to `dp`

```
# make sure you are in the root dir of project
git checkout dp
```

# Compare the results

you can calculate the results by running `read_results.ipynb` in `results` dir
