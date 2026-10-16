
# Paper-default hyperparameters of the competitor strategies in app/strategies/competitors/.
# Every value names where it comes from, and none was tuned on test data. Strategies copy their
# entry at construction, so a test or an ablation can override a value without editing this file.
# Mutable module global, like `al_params` in config.py.

competitor_params = {
    'typiclust': {
        # Hacohen, Dekel and Weinshall, ICML 2022 (arXiv 2202.02794), App. F.1 unless noted.
        # 'authors': rule of the official code (github.com/avihu111/TypiClust): visit clusters sorted by
        #            fewest labelled points, then largest, cycling through them.
        # 'dal':     rule of dal-toolbox's TypiClust: largest *uncovered* cluster, then uniform random
        #            once every cluster holds a labelled point (that is almost every pick when B > K).
        'selection': 'authors',
        'k_nn': 20,                        # Sec. 3.1, footnote 1
        'min_cluster_size': 5,             # clusters with <= 5 points are dropped (official code: size > 5)
        'max_clusters': 500,               # CIFAR-10/100; K = min(|L| + B, max_clusters)
        'max_clusters_by_dataset': {'tinyimagenet': 1000},
        'l2_normalize': True,              # "We used the L2 normalized penultimate layer as embedding"
        'minibatch_threshold': 50,         # KMeans if K <= 50, MiniBatchKMeans otherwise
        'minibatch_size': 5000,            # official code: MiniBatchKMeans(batch_size=5000)
    },
    'probcover': {
        # Yehuda, Dekel, Hacohen and Weinshall, NeurIPS 2022 (arXiv 2205.11320).
        # None: the paper's rule, the largest radius at which `purity_threshold` of the balls are pure with
        #       k-means pseudo-labels (k = number of classes), re-estimated on the features of every query.
        # float: a fixed radius. The authors' scripts use 0.5-0.6 for L2-normalised SimCLR features
        #       (tools/scripts/run_budget.sh of the uherding copy), dal-toolbox's default is 0.6.
        'delta': None,
        'purity_threshold': 0.95,          # alpha, Sec. 3
        'l2_normalize': True,              # the authors' code L2-normalises the features
        'minibatch_threshold': 50,         # pseudo-label k-means: exact up to 50 clusters, mini-batch above
        'minibatch_size': 5000,
    },
    'bait': {
        # Ash, Goel, Krishnamurthy and Kakade, NeurIPS 2021 (arXiv 2106.09675). Needs the trained model's own
        # penultimate features and logits, so it has no -fs file mode.
        # `bait` is the Fisher information of the softmax last layer, dimension C*D (the paper's BAIT). Exact, and
        # only feasible for few classes: the (C*D x C*D) matrix is 210 MB at C = 10, 21 GB at C = 100.
        # `bait_bce` is dal-toolbox's grad_likelihood='binary_cross_entropy', a Bernoulli likelihood on the top
        # class probability, dimension D whatever C is. An approximation, not the paper's Fisher.
        'lambda_reg': 1.0,                 # lambda = 1 in the paper's experiments except CIFAR-10 (App. C; badge run.py)
        'lambda_by_dataset': {'cifar10': 0.01},
        'oversample': 2,                   # 2B forward steps, then B backward removals (Alg. 1)
        'max_embedding_dim': 12000,        # refuse a larger Fisher matrix instead of running out of memory mid-sweep
        'device': 'cpu',                   # 'cuda' is much faster; the linear algebra is float64 either way
    },
}
