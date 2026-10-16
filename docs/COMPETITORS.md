# Competitor strategies

Query strategies ported from published papers. Each one implements only `query()`: backbone, initial
labelled set, seeds, budgets, training recipe and metric are the benchmark's, so a competitor runs under
the protocol of ALGTG and of the baselines (`docs/architecture.md`). Ports are pure PyTorch / NumPy /
scikit-learn; no reference repository is imported at runtime. Hyperparameters live in
`app/competitor_config.py`, each with the place it comes from.

| `-m` | Paper | Ported from | Inputs | Cost per query | Licence of the source |
|---|---|---|---|---|---|
| `typiclust` | Hacohen, Dekel, Weinshall, *Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets*, ICML 2022, [arXiv 2202.02794](https://arxiv.org/abs/2202.02794). Official code, linked in the abstract: [avihu111/TypiClust](https://github.com/avihu111/TypiClust) | dal-toolbox `typiclust.py`, `randomclust.py` @`13ee9ca`; authors' `deep-al/pycls/al/typiclust.py` @`4097a71` | features of labelled + candidates; no logits, no labels | 4-6 s, one thread (N = 10k, L <= 9k, D = 512) | BSD-3-Clause (dal-toolbox), MIT (authors) |

## Running

```
python app/main.py -gpus 1 -m typiclust -ds cifar10 -tr 5                          # model features (default)
python app/main.py -gpus 1 -m typiclust -ds cifar10 -tr 5 -fs file -fp feats.npy   # precomputed features
```

**Feature source (`-fs`, `-fp`).** `model` (default) clusters and measures distances in the penultimate
embedding of the ResNet-18 trained in the current cycle, the same features CoreSet and BADGE use. `file`
reads a `.npy` of shape `(training set size, D)`, one row per training sample in the dataset's torchvision
order, for the self-supervised features the papers use. This repository does not train them. A strategy run
on a file is named `..._ssl` so its rows never merge with the model-feature run in `results.csv`; the file
itself is not part of the name, so use one `-o` per feature file. A feature file is representation learning
on the whole unlabelled set, which the baselines do not get: report `_ssl` rows as a separate group.

## TypiClust

**What the paper used.** Features: SimCLR trained with the code of SCAN (Van Gansbeke et al.) on the whole
labelled + unlabelled pool (ResNet-18, 128-d projection head, 500 epochs, SGD lr 0.4 with cosine decay,
batch 512, weight decay 1e-4), then the 512-d penultimate layer, L2-normalised (DINO ViT-S/16 for
ImageNet). Clustering: `K = min(|L| + B, max_clusters)` k-means clusters, scikit-learn `KMeans` for
`K <= 50` and `MiniBatchKMeans` above. Each step takes the largest of the clusters with the fewest labelled
points (clusters with fewer than 5 points dropped) and queries its most typical point; typicality is the
inverse mean distance to the 20 nearest neighbours (App. F.1, Sec. 3).

| Parameter | Value | Source |
|---|---|---|
| `k_nn` | 20 | paper Sec. 3.1, footnote 1 |
| `min_cluster_size` | 5 (clusters of size <= 5 are dropped) | App. F.1; the official code tests `size > 5` |
| `max_clusters` | 500; 1000 for Tiny-ImageNet | App. F.1 (500 CIFAR-10/100; none given for SVHN, FMNIST, Caltech256, so 500) |
| `l2_normalize` | on | App. F.1 |
| `minibatch_threshold`, `minibatch_size` | 50, 5000 | App. F.1; official code |
| `selection` | `authors` | see below |

**Deviations from the paper / the reference code.**

1. **Selection rule.** `authors` (default) is the rule of the official code: clusters are sorted once by
   (fewest labelled points, largest), then visited cyclically, one most-typical point per visit. The paper's
   text describes the same preference re-evaluated at every step; the two coincide while uncovered clusters
   last. dal-toolbox's `TypiClust` does something else: it queries the largest *uncovered* cluster and, once
   every cluster holds a labelled point, picks uniformly at random. With `K` capped at 500 and `B = 1000`
   that is most of the budget (72-87 % random picks measured on synthetic 512-d class-structured features,
   L = 1000...9000). It stays available as `selection: 'dal'` (name `ResNet_TypiClust_dal`) and is what the
   golden test checks it against.
2. **Candidate pool.** The protocol scores 10,000 random unlabelled samples per cycle; the paper queries from
   the whole unlabelled set.
3. **Features.** Default is the supervised ResNet-18 embedding, not SimCLR (see Running). With the default,
   labelled points embed more tightly than unlabelled ones, which biases which clusters look covered.
4. **Regime.** `L >= 1000` and `B = 1000` is outside the low-budget regime TypiClust is designed for (the
   paper argues typical points are best queried at low budgets and atypical ones at large budgets), and `K` is
   always below `B`, so every cluster is visited more than once. Do not expect the paper's low-budget gains.
5. **Typicality** uses squared distances plus `1e-5`, as both reference codes do; the paper's Eq. 4 is the plain
   Euclidean distance.
6. **Guards the official code lacks.** Reproduced on crafted inputs: it raises (empty `argmax`) when a visited
   cluster has no member left, and can raise (array lengths disagree) when k-means leaves a cluster empty. The
   port skips exhausted clusters, visits the dropped small clusters only after every kept cluster is empty,
   clamps the budget to the pool size (the one case that returns fewer than `B` indices) and clamps `K` to the
   number of points.
7. **Randomness.** k-means and the `dal` fallback draw from a generator seeded by
   (dataset, trial, cycle, strategy name); the global RNGs are never read or advanced.

**Tests.** `app/tests/test_typiclust.py` (shape, no duplicates, no labelled index, determinism and untouched
global RNGs, budget above the pool, empty labelled set, degenerate clusterings, pool-position mapping,
registration) and `app/tests/test_competitor_common.py`. The golden tests compare both rules with the
reference code under an identical injected clustering; `app/tests/golden/typiclust_golden.json` is produced by
`app/tests/golden/make_typiclust_golden.py`, the only place the reference code runs:

```
python app/tests/golden/make_typiclust_golden.py --dal-toolbox <dal-toolbox clone> \
    --typiclust-original <avihu111/TypiClust deep-al/pycls/al/typiclust.py> --typiclust-commit <sha>
```

Smoke: `python app/tests/smoke_run.py typiclust [-fs file -fp feats.npy]`.

## Not ported

- **BADGE, BALD, CoreSet** are already in the benchmark (`strategies/competitors/`).
- **BoSS** is an oracle (`dal-toolbox/publications/boss_oracle`), not a deployable strategy.
- **TAILOR** ([jifanz/TAILOR](https://github.com/jifanz/TAILOR) @`8a23e14`) is not multi-label only: its README
  documents a multi-label and a multi-class launcher. It is a bandit *meta-algorithm* (`thompson_pos`,
  `thompson_div`, `albl_meta`, `random_meta`) that chooses among candidate query strategies (`galaxy`,
  `confidence`, `badge`, `mlp`, `similar`, ...), run on imbalanced datasets (`cifar10_imb_2`, `svhn_imb_2`,
  `cifar100_imb_10`, ...) with pretrained backbones fine-tuned for 8-15 epochs and logged to Weights & Biases.
  It is not a single-model query strategy under this benchmark's training recipe.

## Licences of the reference repositories

This repository is Apache-2.0. Ported files keep the notices of what they were ported from.

| Repository | Licence | Consequence for ports |
|---|---|---|
| `dal-toolbox` (dhuseljic) | BSD-3-Clause, (c) 2020 The dal_toolbox developers | Keep the notice, conditions and disclaimer in the source; no endorsement by the holders' names |
| `avihu111/TypiClust` | MIT, (c) 2022 Avihu Dekel and Guy Hacohen | Keep the notice |
| `uherding` (BorealisAI) | CC BY-NC-SA 4.0 | NonCommercial, and adaptations must be shared under the same licence: UHerding / MaxHerding ports would be adapted material |
| `uherding/scan` | CC BY-NC 4.0 | NonCommercial; not ported (no SimCLR / SCAN training code is copied) |
| `badge` (JordanAsh) | none found (no licence file, no notice in any source or README) | All rights reserved by default: nothing is ported from it |
| `TAILOR` (jifanz) | MIT, (c) 2023 Jifan Zhang | Not ported |
