# ALGTG
## Active Learning strategy using Graph Transduction Game
In recent years, internet technology development has led to an era of abundant data, sparking strong interest in **Deep Learning (DL)** which is known to be greedy for data and requires a large amount of information supply to optimize a massive number of parameters. In the same period **Active learning (AL)** which attempts to maximize a model’s performance gain while annotating the fewest samples possible, was set aside, since traditional machine learning required few labeled samples before DL's rise, making AL undervalued. DL's success relies on existing annotated datasets, however acquiring high-quality annotations demands significant manpower, posing challenges in specialized fields like speech recognition and medical imaging. Consequently, AL is now gaining the attention it deserves.

In this thesis we want to exploit the combination of these two fields by leveraging a non-cooperative game that aims to classify unlabelled objects starting from a small set of labelled ones, the **Graph Transduction Game (GTG)**. We introduce **Active Learning through Graph Transduction Game (ALGTG)**, a novel query-acquisition function that based its behavior on the sudden GTG algorithm.

GTG formulates the classification task as an evolutionary non-cooperative game between *N* players (samples) with *M* strategies (labels) and aims to assign the most suitable pseudo-label to unmarked observations. Once we reach this situation we are in a situation of Nash Equilibria since all the samples are labeled consistently.

To this end, the selection of samples to be labeled in our AL model is based on: tracking the evolution of the entropy along the iteration of the aforementioned dynamical system and creating an ad-hoc payoff function such that similar samples (already seen) are discouraged to emerge in the subsequent iterations. We exploit the performance of our approach on five publicly available image classification benchmarks: *CIFAR10/100*, *SVHN*, *Fashion-MNIST*, and *Tiny-ImageNET*.


## Method overview

ALGTG picks the samples to annotate by watching how the entropy of each unlabelled point evolves while the GTG dynamical system propagates labels from the labelled set. The affinity (payoff) matrix is built from the backbone features, labelled nodes are clamped to their known labels, and the replicator dynamics are iterated to (near-)convergence. Two families are provided:

- **Offline GTG** (`gtg`, `ll_gtg`) — after the backbone is trained, embeddings are extracted, the affinity matrix is built, GTG is run to convergence, and the unlabelled samples with the highest entropy history (`mean` or `integral`) are queried. `ll_gtg` additionally trains the backbone with the Learning-Loss auxiliary before the same offline query.
- **Online / end-to-end GTG** (`llmlp_gtg`, `lsmlps_gtg`, `lsmlpsA_gtg`, `lstmreg_gtg`, `lstmbc_gtg`) — a learnable module runs a **differentiable** GTG on intermediate features during training and learns to predict the entropy signal used for acquisition, so the acquisition head is trained jointly with the backbone.

**Affinity matrix** (`-am`): cosine similarity (`cos_sim`), Pearson correlation (`corr`), or an adaptive RBF kernel (`rbfk`, bandwidth from the 7th-nearest neighbour).
**Post-processing** (`-am_s`): `diversity` (`1 − similarity`), `mixed` (labelled–labelled block turned into a distance, the rest kept as similarity), or `uncertanity`.
**Sparsification** (`-am_ts`, `-am_t`): drop weak connections with a fixed threshold or the matrix mean.
**Entropy aggregation** (`-e_s`): `mean` or `integral` (trapezoid) of the entropy history.

### Methods available (`-m`)

| Group | Methods |
|-------|---------|
| Baselines | `random`, `entropy`, `ll` (Learning Loss) |
| Competitors | `coreset`, `badge`, `bald`, `cdal`, `alphamix`, `tavaal`, `tidal` |
| Proposed — offline GTG | `gtg`, `ll_gtg` |
| Proposed — online GTG | `llmlp_gtg`, `lsmlps_gtg`, `lsmlpsA_gtg`, `lstmreg_gtg`, `lstmbc_gtg` |


## Repository structure

```
app/
├── main.py             # entry point: argument parsing, trial loop, result plotting
├── config.py           # dataset configs, AL params, per-dataset training hyperparameters
├── dataset.py          # dataset wrappers, initial labelled/unlabelled split, cold start
├── init.py             # backbone / module / strategy factories
├── ActiveLearner.py    # base AL loop (train → query → update sets)
├── utils.py            # entropy, plotting, checkpoints, seeding
├── models/
│   ├── BBone_Module.py # Master_Model (backbone + optional module)
│   ├── backbones/      # ResNet18, VGG16-bn
│   └── modules/
│       ├── LossNet.py  # Learning-Loss module
│       └── GTG/        # online GTGModule + MLP/LSTM sub-modules
├── strategies/
│   ├── baselines/      # Random, Entropy, LearningLoss
│   ├── competitors/    # CoreSet, BADGE, BALD, CDAL, AlphaMix, TiDAL, TA-VAAL
│   └── gtg/            # GTG (online) and GTG_off (offline) — the proposed methods
├── train_evaluate/     # training workers (single-GPU + DDP)
├── auto_encoder/       # autoencoder used for cold-start initial sampling
└── checkpoints/        # saved model checkpoints
datasets/               # downloaded datasets
results/                # experiment outputs (plots, csv, logs)
docs/                   # thesis & presentation PDFs
```


## Requirements

Change the last row of **environment.yml** to the path of your conda environments. Then run the following lines in the shell.
```
conda env create -f environment.yml
conda activate <env>
```

Supported backbones are **ResNet18** (used for CIFAR10/100, SVHN, Fashion-MNIST, Tiny-ImageNet) and **VGG16-bn** (used for Caltech256). CIFAR10/100, SVHN and Fashion-MNIST download automatically; Tiny-ImageNet and Caltech256 are fetched and prepared on first use.


## Application start-up

Example:
```
python app/main.py -gpus 1 -m random entropy coreset badge bald cdal tavaal ll ll_gtg -ds cifar10 -tr 5 -am corr -am_s mixed -am_ts mean -e_s mean
```

> **Note:** only single-GPU training (`-gpus 1`) is currently supported. The multi-GPU DDP path is a work in progress.

### Arguments

| Argument | Values / type | Default | Description |
|----------|---------------|---------|-------------|
| `-gpus`, `--gpus` | int | `1` | Number of GPUs to use during training |
| `-ma`, `--master_address` | str | `localhost` | Master address for distributed training |
| `-mp`, `--master_port` | str | `12355` | Master port for distributed training |
| `-m`, `--methods` | see *Methods available* | *(required)* | One or more methods to run |
| `-ds`, `--datasets` | `cifar10` `cifar100` `svhn` `fmnist` `caltech256` `tinyimagenet` | *(required)* | One or more datasets |
| `-tr`, `--trials` | int | `5` | Number of AL trials (repetitions with different seeds) |
| `-bbone`, `--bbone_pre_train` | flag | off | Pre-train the backbone for the GTG module (requires a cold-start strategy) |
| `-cs_s`, `--cold_start_strategy` | `none` `mse` `kmeans` | `none` | Cold-start strategy for the initial labelled set |
| `-tulp`, `--temp_unlab_pool` | flag | off | Use a temporary unlabelled pool per iteration |
| `-am`, `--affinity_matrix` | `corr` `cos_sim` `rbfk` | `corr` | Affinity matrix function(s) |
| `-am_md`, `--affinity_matrix_mixed_distance` | `LL` `UU` `LU` `UL` | `LL` | Affinity-matrix block(s) converted to a distance under the `mixed` strategy |
| `-am_sl`, `--affnity_matrix_self_loop` | flag | off (no self loop) | Add self loops to the affinity matrix (diagonal set to 1) |
| `-am_s`, `--affinity_matrix_strategy` | `uncertanity` `diversity` `mixed` | `mixed` | Affinity-matrix post-processing |
| `-am_ts`, `--affinity_matrix_threshold_strategies` | `threshold` `mean` `none` | `mean none` | Threshold strategy/strategies for sparsification |
| `-am_t`, `--affinity_matrix_threshold` | float | `0.5` | Threshold value (ignored when the strategy is `mean`) |
| `-e_s`, `--entropy_strategy` | `mean` `integral` | `mean` | How the entropy history is aggregated |
| `-gtg_iter`, `--gtg_iterations` | int | `30` | Maximum number of GTG iterations |
| `-gtg_t`, `--gtg_tollerance` | float | `0.0001` | GTG convergence tolerance |
| `-bsgtgo`, `--batch_size_gtg_online` | int | `32` | Initial batch size for the online GTG version |


## Active Learning protocol

The AL loop is configured in `app/config.py` (`al_params`):

- **Initial labelled set:** 1000 samples
- **AL iterations:** 10 cycles
- **Budget per cycle:** 1000 newly labelled samples
- **Unlabelled pool sampled each cycle:** 10000 random samples
- **Training:** 200 epochs per cycle, SGD backbone, per-dataset batch sizes/optimizers in `config.py`

Each cycle retrains the model from the initial checkpoint on the current labelled set, runs the selected query strategy, and moves the chosen samples into the labelled set. Results (test accuracy, class counts, t-SNE and entropy plots) are written under `results/<timestamp>/`.
