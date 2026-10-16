
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
}
