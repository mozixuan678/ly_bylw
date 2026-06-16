from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import NearestNeighbors


def _minmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(v))
    hi = float(np.max(v))
    if hi - lo < 1e-12:
        return np.zeros_like(v, dtype=np.float64)
    return (v - lo) / (hi - lo)


def fisher_score(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    overall = X.mean(axis=0)
    numerator = np.zeros(X.shape[1], dtype=np.float64)
    denominator = np.zeros(X.shape[1], dtype=np.float64)
    for cls in np.unique(y):
        Xc = X[y == cls]
        if len(Xc) == 0:
            continue
        mu = Xc.mean(axis=0)
        numerator += len(Xc) * (mu - overall) ** 2
        denominator += len(Xc) * Xc.var(axis=0)
    return numerator / (denominator + 1e-12)


def mutual_info_score(X: np.ndarray, y: np.ndarray, random_state: int = 0) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    try:
        return mutual_info_classif(X, y, discrete_features=False, random_state=random_state)
    except Exception:
        # A deterministic fallback keeps the pipeline runnable if MI estimation
        # fails for near-constant columns.
        return fisher_score(X, y)


def relief_score(X: np.ndarray, y: np.ndarray, n_neighbors: int = 1, batch_size: int = 256) -> np.ndarray:
    """Relief score using class-wise nearest-neighbor queries."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n_samples, n_features = X.shape
    ranges = X.max(axis=0) - X.min(axis=0)
    ranges[ranges < 1e-12] = 1.0
    scores = np.zeros(n_features, dtype=np.float64)

    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        other_idx = np.flatnonzero(y != cls)
        if len(cls_idx) <= 1 or len(other_idx) == 0:
            continue
        X_cls = X[cls_idx]
        X_other = X[other_idx]
        k_hit = min(n_neighbors + 1, len(cls_idx))
        k_miss = min(n_neighbors, len(other_idx))
        hit_nn = NearestNeighbors(n_neighbors=k_hit, algorithm="auto").fit(X_cls)
        miss_nn = NearestNeighbors(n_neighbors=k_miss, algorithm="auto").fit(X_other)
        hit_ind = hit_nn.kneighbors(X_cls, return_distance=False)
        miss_ind = miss_nn.kneighbors(X_cls, return_distance=False)
        if k_hit > 1:
            hit_ind = hit_ind[:, 1:]
        for start in range(0, len(cls_idx), batch_size):
            stop = min(start + batch_size, len(cls_idx))
            base = X_cls[start:stop]
            hit = X_cls[hit_ind[start:stop]].mean(axis=1)
            miss = X_other[miss_ind[start:stop]].mean(axis=1)
            scores -= (np.abs(base - hit) / ranges).sum(axis=0)
            scores += (np.abs(base - miss) / ranges).sum(axis=0)
    return scores / max(1, n_samples)


def _inf_apply_a(eta: np.ndarray, v: np.ndarray) -> np.ndarray:
    return eta * float(np.dot(eta, v)) - (eta * eta) * v


def _power_radius_for_inf(eta: np.ndarray, n_iter: int = 50) -> float:
    rng = np.random.RandomState(0)
    v = rng.rand(len(eta))
    v /= np.linalg.norm(v) + 1e-12
    radius = 0.0
    for _ in range(n_iter):
        nv = _inf_apply_a(eta, v)
        radius = float(np.linalg.norm(nv))
        if radius < 1e-12:
            return 0.0
        v = nv / radius
    return radius


def inf_fs_score(X: np.ndarray, y: np.ndarray, random_state: int = 0, max_terms: int = 100) -> np.ndarray:
    """Infinite Feature Selection score without materializing the full graph.

    The graph used here is A_ij = eta_i eta_j with zero diagonal, which allows
    O(d) matrix-vector products.
    """
    X = np.asarray(X, dtype=np.float64)
    fs = _minmax(fisher_score(X, y))
    mi = _minmax(mutual_info_score(X, y, random_state=random_state))
    sd = _minmax(X.std(axis=0))
    eta = (fs + mi + sd) / 3.0
    rho = max(_power_radius_for_inf(eta), 1e-12)
    gamma = 0.9 / rho
    ones = np.ones(X.shape[1], dtype=np.float64)
    term = gamma * _inf_apply_a(eta, ones)
    total = term.copy()
    for _ in range(max_terms - 1):
        term = gamma * _inf_apply_a(eta, term)
        total += term
        if np.linalg.norm(term) < 1e-8 * (np.linalg.norm(total) + 1e-12):
            break
    return total


def _q_max_matvec(sorted_s: np.ndarray, sorted_v: np.ndarray) -> np.ndarray:
    prefix_v = np.cumsum(sorted_v)
    prefix_sv = np.cumsum(sorted_s * sorted_v)
    total_sv = float(prefix_sv[-1])
    return sorted_s * prefix_v + (total_sv - prefix_sv)


def ecfs_score(X: np.ndarray, y: np.ndarray, alpha: float = 0.5, random_state: int = 0, n_iter: int = 100) -> np.ndarray:
    """ECFS eigenvector centrality score with memory-friendly matvecs."""
    X = np.asarray(X, dtype=np.float64)
    f = _minmax(fisher_score(X, y))
    m = _minmax(mutual_info_score(X, y, random_state=random_state))
    s = _minmax(X.std(axis=0))
    order = np.argsort(s)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(len(order))
    s_sorted = s[order]

    r = np.ones(X.shape[1], dtype=np.float64)
    r /= np.linalg.norm(r) + 1e-12
    for _ in range(n_iter):
        k_part = f * float(np.dot(m, r))
        r_sorted = r[order]
        q_sorted = _q_max_matvec(s_sorted, r_sorted)
        q_part = q_sorted[inv_order]
        nr = alpha * k_part + (1.0 - alpha) * q_part
        nr = np.maximum(nr, 0.0)
        norm = float(np.linalg.norm(nr))
        if norm < 1e-12:
            break
        nr /= norm
        if np.linalg.norm(nr - r) < 1e-7:
            r = nr
            break
        r = nr
    return r


def rank_from_scores(scores: np.ndarray) -> np.ndarray:
    return np.argsort(np.asarray(scores))[::-1]


def jcfs(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    beta: Tuple[float, float, float] = (1.0 / 3, 1.0 / 3, 1.0 / 3),
    ecfs_alpha: float = 0.5,
    relief_neighbors: int = 1,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    k = min(int(k), X.shape[1])
    s_relief = relief_score(X, y, n_neighbors=relief_neighbors)
    s_inf = inf_fs_score(X, y, random_state=random_state)
    s_ec = ecfs_score(X, y, alpha=ecfs_alpha, random_state=random_state)
    score = beta[0] * _minmax(s_relief) + beta[1] * _minmax(s_inf) + beta[2] * _minmax(s_ec)
    rank = rank_from_scores(score)
    return rank[:k].copy(), rank.copy(), {
        "relief": s_relief,
        "inf_fs": s_inf,
        "ecfs": s_ec,
        "jcfs": score,
    }


def select_features_by_method(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    method: str,
    random_state: int = 0,
    relief_neighbors: int = 1,
    ecfs_alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    method = method.lower()
    if method == "relief":
        score = relief_score(X, y, n_neighbors=relief_neighbors)
        scores = {"relief": score}
    elif method in {"inf", "inf-fs", "inffs"}:
        score = inf_fs_score(X, y, random_state=random_state)
        scores = {"inf_fs": score}
    elif method == "ecfs":
        score = ecfs_score(X, y, alpha=ecfs_alpha, random_state=random_state)
        scores = {"ecfs": score}
    elif method == "jcfs":
        return jcfs(X, y, k, relief_neighbors=relief_neighbors, ecfs_alpha=ecfs_alpha, random_state=random_state)
    else:
        raise ValueError("Unknown feature selection method: %s" % method)
    rank = rank_from_scores(score)
    k = min(int(k), X.shape[1])
    return rank[:k].copy(), rank.copy(), scores

