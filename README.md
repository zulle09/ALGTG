# ALGTG
## Active Learning strategy using Graph Transduction Game
In recent years, internet technology development has led to an era of abundant data, sparking strong interest in **Deep Learning (DL)** which is known to be greedy for data and requires a large amount of information supply to optimize a massive number of parameters. In the same period **Active learning (AL)** which attempts to maximize a model’s performance gain while annotating the fewest samples possible, was set aside, since traditional machine learning required few labeled samples before DL's rise, making AL undervalued. DL's success relies on existing annotated datasets, however acquiring high-quality annotations demands significant manpower, posing challenges in specialized fields like speech recognition and medical imaging. Consequently, AL is now gaining the attention it deserves.

In this thesis we want to exploit the combination of these two fields by leveraging a non-cooperative game that aims to classify unlabelled objects starting from a small set of labelled ones, the **Graph Transduction Game (GTG)**. We introduce **Active Learning through Graph Transduction Game (ALGTG)**, a novel query-acquisition function that based its behavior on the sudden GTG algorithm.

GTG formulates the classification task as an evolutionary non-cooperative game between *N* players (samples) with *M* strategies (labels) and aims to assign the most suitable pseudo-label to unmarked observations. Once we reach this situation we are in a situation of Nash Equilibria since all the samples are labeled consistently.

To this end, the selection of samples to be labeled in our AL model is based on: tracking the evolution of the entropy along the iteration of the aforementioned dynamical system and creating an ad-hoc payoff function such that similar samples (already seen) are discouraged to emerge in the subsequent iterations. We exploit the performance of our approach on five publicly available image classification benchmarks: *CIFAR10/100*, *SVHN*, *Fashion-MNIST*, and *Tiny-ImageNET*.

## Requirements

Change the last row of **requirements.txt** to the path of yout conda enviroments. Then type in the bash the following lines.
```
conda create --name <env> --file requirements.txt
conda activate <env>
```

### Multi GPUs
If you want to perform experiments leveraging multiple gpus, you should edit **ActiveLearner.py** file, *train_evaluate* function, more precisely:
```python
os.environ["MASTER_ADDR"] = "xxxxxxxxxx" # -> replace with the correct address
os.environ["MASTER_PORT"] = "yyyyyyyyyy" # -> repalce with a free port
```

## Application Start-Up
Example of application start-up:
```
python3 app/main.py -gpus 1 -m random entropy coreset badge bald cdal tavaal ll ll_gtg -ds cifar10 -tr 5 -am corr -am_s mixed -am_ts mean -e_s mean
```

Accepted Arguments:
```
usage: main.py [-h] [-gpus GPUS] -m
               {random,entropy,coreset,badge,bald,cdal,tavaal,alphamix,tidal,ll,gtg,ll_gtg,llmlp_gtg,lsmlps_gtg,lstmreg_gtg,lstmbc_gtg}
               [{random,entropy,coreset,badge,bald,cdal,tavaal,alphamix,tidal,ll,gtg,ll_gtg,llmlp_gtg,lsmlps_gtg,lstmreg_gtg,lstmbc_gtg} ...]
               -ds {cifar10,cifar100,svhn,fmnist,caltech256,tinyimagenet}
               [{cifar10,cifar100,svhn,fmnist,caltech256,tinyimagenet} ...]
               [-tr TRIALS] [-bbone] [-tulp]
               [-am {corr,cos_sim,rbfk} [{corr,cos_sim,rbfk} ...]]
               [-am_s {uncertanity,diversity,mixed}]
               [-am_ts {threshold,mean,none} [{threshold,mean,none} ...]]
               [-am_t AFFINITY_MATRIX_THRESHOLD] [-e_s {mean,integral}]
               [-gtg_iter GTG_ITERATIONS] [-gtg_t GTG_TOLLERANCE]
               [-bsgtgo BATCH_SIZE_GTG_ONLINE]

options:
  -h, --help            show this help message and exit
  -gpus GPUS, --gpus GPUS
                        Number of GPUs to use during training
  -m {random,entropy,coreset,badge,bald,cdal,tavaal,alphamix,tidal,ll,gtg,ll_gtg,llmlp_gtg,lsmlps_gtg,lstmreg_gtg,lstmbc_gtg} [{random,entropy,coreset,badge,bald,cdal,tavaal,alphamix,tidal,ll,gtg,ll_gtg,llmlp_gtg,lsmlps_gtg,lstmreg_gtg,lstmbc_gtg} ...], 
      --methods {random,entropy,coreset,badge,bald,cdal,tavaal,alphamix,tidal,ll,gtg,ll_gtg,llmlp_gtg,lsmlps_gtg,lstmreg_gtg,lstmbc_gtg} [{random,entropy,coreset,badge,bald,cdal,tavaal,alphamix,tidal,ll,gtg,ll_gtg,llmlp_gtg,lsmlps_gtg,lstmreg_gtg,lstmbc_gtg} ...]
                        Possible methods to choose
  -ds {cifar10,cifar100,svhn,fmnist,caltech256,tinyimagenet} [{cifar10,cifar100,svhn,fmnist,caltech256,tinyimagenet} ...], 
      --datasets {cifar10,cifar100,svhn,fmnist,caltech256,tinyimagenet} [{cifar10,cifar100,svhn,fmnist,caltech256,tinyimagenet} ...]
                        Possible datasets to choose
  -tr TRIALS, --trials TRIALS
                        AL trials
  -bbone, --bbone_pre_train
                        BackBone pre-train (binary calssification task) for GTG Module
  -tulp, --temp_unlab_pool
                        Temporary Unlabelled Pool
  -am {corr,cos_sim,rbfk} [{corr,cos_sim,rbfk} ...], --affinity_matrix {corr,cos_sim,rbfk} [{corr,cos_sim,rbfk} ...]
                        Affinity matrix to choose
  -am_s {uncertanity,diversity,mixed}, --affinity_matrix_strategy {uncertanity,diversity,mixed}
                        Different affinity matrix modification
  -am_ts {threshold,mean,none} [{threshold,mean,none} ...], --affinity_matrix_threshold_strategies {threshold,mean,none} [{threshold,mean,none} ...]
                        Possible treshold strategy types to choose to apply in the affinity matrix
  -am_t AFFINITY_MATRIX_THRESHOLD, --affinity_matrix_threshold AFFINITY_MATRIX_THRESHOLD
                        Affinity Matrix Threshold for our method, when threshold_strategy = mean, this is ignored
  -e_s {mean,integral}, --entropy_strategy {mean,integral}
                        Entropy strategy to sum up the entropy history
  -gtg_iter GTG_ITERATIONS, --gtg_iterations GTG_ITERATIONS
                        Maximum GTG iterations to perorm
  -gtg_t GTG_TOLLERANCE, --gtg_tollerance GTG_TOLLERANCE
                        GTG tollerance
  -bsgtgo BATCH_SIZE_GTG_ONLINE, --batch_size_gtg_online BATCH_SIZE_GTG_ONLINE
                        Initial batch size for the online GTG version

```

