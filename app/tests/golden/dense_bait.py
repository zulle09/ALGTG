"""
BAIT with explicit matrices and explicit inverses: the objective is evaluated from scratch for every candidate
at every step, with no Woodbury identity and no Kronecker structure, so it shares nothing with the port but the
definition. Used by tests/test_bait.py and tests/golden/make_bait_golden.py. Plain numpy, small sizes only.
"""

import numpy as np


def softmax(logits):
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def explicit_embeddings(logits, features, kind):
    """X_i (rank x dim) for every sample, built entry by entry from the Fisher definition."""
    probs = softmax(logits)
    n_samples, n_classes = probs.shape
    dim = features.shape[1]
    embeddings = []
    for i in range(n_samples):
        if kind == 'ce':
            x = np.zeros((n_classes, n_classes * dim))
            for j in range(n_classes):
                for h in range(n_classes):
                    x[j, h * dim:(h + 1) * dim] = np.sqrt(probs[i, j]) * ((j == h) - probs[i, h]) * features[i]
        else:
            top = probs[i].max()
            x = np.stack([np.sqrt(top) * (1 - top) * features[i], -np.sqrt(1 - top) * top * features[i]])
        embeddings.append(x)
    return np.array(embeddings)


def dense_bait(labelled_logits, labelled_features, pool_logits, pool_features, budget, lam, kind='ce', oversample=2):
    """(selected, gaps): the picks in selection order, and the relative gap between the best and the second
    best objective at every step (a fixture is only trustworthy when no gap is tiny)."""
    x_pool = explicit_embeddings(pool_logits, pool_features, kind)
    rank, dim = x_pool.shape[1:]
    x_lab = explicit_embeddings(labelled_logits, labelled_features, kind) if len(labelled_features) else np.zeros((0, rank, dim))
    n_lab = len(x_lab)
    gram = lambda x: np.einsum('nrd,nre->de', x, x)
    fisher = (gram(x_lab) + gram(x_pool)) / (n_lab + len(x_pool))
    m = lam * np.eye(dim) + gram(x_lab) / (n_lab + budget)
    x_cand = x_pool * np.sqrt(budget / (n_lab + budget))

    def relative_gap(objectives):
        ordered = np.sort(objectives[np.isfinite(objectives)])
        return (ordered[1] - ordered[0]) / abs(ordered[0]) if len(ordered) > 1 else np.inf

    selected, gaps = [], []
    for _ in range(min(oversample * budget, len(x_pool))):
        objectives = np.full(len(x_pool), np.inf)
        for i in range(len(x_pool)):
            if i not in selected:
                objectives[i] = np.trace(np.linalg.inv(m + x_cand[i].T @ x_cand[i]) @ fisher)
        best = int(np.argmin(objectives))
        gaps.append(relative_gap(objectives))
        selected.append(best)
        m = m + x_cand[best].T @ x_cand[best]
    while len(selected) > budget:
        objectives = np.array([np.trace(np.linalg.inv(m - x_cand[s].T @ x_cand[s]) @ fisher) for s in selected])
        position = int(np.argmin(objectives))
        gaps.append(relative_gap(objectives))
        m = m - x_cand[selected[position]].T @ x_cand[selected[position]]
        selected.pop(position)
    return selected, gaps
