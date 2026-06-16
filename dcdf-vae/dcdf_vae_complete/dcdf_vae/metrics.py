from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import rankdata
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _mean_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores)
    if scores.ndim == 4:
        return scores.mean(axis=(0, 1))
    if scores.ndim == 3:
        return scores.mean(axis=0)
    if scores.ndim == 2:
        return scores
    raise ValueError(f"Unsupported score shape {scores.shape}")


def evaluate_graph_scores(scores: np.ndarray, true_adj: np.ndarray, include_diag: bool = True) -> dict[str, float]:
    mat = _mean_scores(scores)
    truth = np.asarray(true_adj).astype(int)
    mask = np.ones_like(truth, dtype=bool)
    if not include_diag:
        np.fill_diagonal(mask, False)
    y_true = truth[mask].ravel()
    y_score = mat[mask].ravel()
    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "best_f1": float("nan"), "threshold": 0.5}
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    idx = int(np.nanargmax(f1))
    threshold = thresholds[max(0, idx - 1)] if len(thresholds) else 0.5
    pred = (y_score >= threshold).astype(int)
    tp = np.sum((pred == 1) & (y_true == 1))
    fp = np.sum((pred == 1) & (y_true == 0))
    fn = np.sum((pred == 0) & (y_true == 1))
    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "best_f1": float(f1[idx]),
        "threshold": float(threshold),
        "precision": float(tp / (tp + fp + 1e-12)),
        "recall": float(tp / (tp + fn + 1e-12)),
        "density": float(pred.mean()),
    }


def js_divergence(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    p = np.abs(a).ravel().astype(np.float64) + eps
    q = np.abs(b).ravel().astype(np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def simple_ssim(a: np.ndarray, b: np.ndarray) -> float:
    x = a.ravel().astype(np.float64)
    y = b.ravel().astype(np.float64)
    c = 1e-4
    mux, muy = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = np.mean((x - mux) * (y - muy))
    return float(((2 * mux * muy + c) * (2 * cov + c)) / ((mux**2 + muy**2 + c) * (vx + vy + c)))


def group_difference_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    return {
        "JSD": js_divergence(a, b),
        "SSIM": simple_ssim(a, b),
        "NND": float(np.linalg.svd(diff, compute_uv=False).sum()),
        "FN": float(np.linalg.norm(diff, ord="fro")),
        "MAE": float(np.mean(np.abs(diff))),
    }


def active_mask(
    subject_ec: np.ndarray,
    alpha: float = 0.01,
    min_abs: float = 1e-4,
    quantile: float = 0.94,
) -> np.ndarray:
    subject_ec = np.asarray(subject_ec)
    if subject_ec.ndim == 2:
        threshold = max(min_abs, float(np.quantile(subject_ec, quantile)))
        return np.abs(subject_ec) > threshold
    t_stat, pval = stats.ttest_1samp(subject_ec, popmean=0.0, axis=0, nan_policy="omit")
    mean = subject_ec.mean(axis=0)
    threshold = max(min_abs, float(np.quantile(mean, quantile)))
    return (pval < alpha) & (mean > threshold) & np.isfinite(t_stat)


def active_connection_counts(
    subject_ec: np.ndarray,
    alpha: float = 0.01,
    quantile: float = 0.94,
    self_quantile: float = 0.90,
) -> dict[str, float]:
    mask = active_mask(subject_ec, alpha=alpha, quantile=quantile)
    mean = subject_ec if np.asarray(subject_ec).ndim == 2 else np.asarray(subject_ec).mean(axis=0)
    diag = np.diag(mean)
    diag_threshold = float(np.quantile(diag, self_quantile))
    diag_mask = diag > diag_threshold
    np.fill_diagonal(mask, False)
    np.fill_diagonal(mask, diag_mask)
    nodes = mask.shape[0]
    sc = int(np.trace(mask))
    uc = 0
    bc_pairs = 0
    for i in range(nodes):
        for j in range(i + 1, nodes):
            a, b = bool(mask[i, j]), bool(mask[j, i])
            if a and b:
                bc_pairs += 1
            elif a or b:
                uc += 1
    bc_edges = 2 * bc_pairs
    total = uc + bc_edges + sc
    return {
        "UCs": float(uc),
        "BCs": float(bc_edges),
        "SCs": float(sc),
        "Active": float(total),
        "UC_ratio": float(uc / total) if total else 0.0,
        "BC_ratio": float(bc_edges / total) if total else 0.0,
        "SC_ratio": float(sc / total) if total else 0.0,
    }


def pearson_subject_connectivity(x: np.ndarray) -> np.ndarray:
    mats = []
    for sample in x:
        mat = np.corrcoef(sample.T)
        mats.append(np.nan_to_num(mat, nan=0.0).astype(np.float32))
    return np.stack(mats)


def spearman_subject_connectivity(x: np.ndarray) -> np.ndarray:
    mats = []
    for sample in x:
        ranks = np.apply_along_axis(rankdata, 0, sample)
        mat = np.corrcoef(ranks.T)
        mats.append(np.nan_to_num(mat, nan=0.0).astype(np.float32))
    return np.stack(mats)


def lagged_dependency_scores(x: np.ndarray, max_lag: int = 2) -> np.ndarray:
    """Data-driven source-past to target-future dependency score in [0, 1]."""

    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        samples = x[None, ...]
    elif x.ndim == 3:
        samples = x
    else:
        raise ValueError("x must be [T,N] or [S,T,N]")
    nodes = samples.shape[-1]
    score = np.zeros((nodes, nodes), dtype=np.float64)
    count = 0
    for sample in samples:
        for lag in range(1, max_lag + 1):
            if sample.shape[0] <= lag + 1:
                continue
            past = sample[:-lag]
            future = sample[lag:]
            past = (past - past.mean(axis=0, keepdims=True)) / (past.std(axis=0, keepdims=True) + 1e-8)
            future = (future - future.mean(axis=0, keepdims=True)) / (future.std(axis=0, keepdims=True) + 1e-8)
            corr = np.abs(past.T @ future / max(1, past.shape[0] - 1))
            score += corr
            count += 1
    if count:
        score /= count
    score -= score.min()
    score /= score.max() + 1e-8
    return score.astype(np.float32)


def mutual_information_lag_scores(
    x: np.ndarray,
    max_lag: int = 3,
    n_neighbors: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Nonlinear lagged dependency score for Lorenz-like systems."""

    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        samples = x[None, ...]
    elif x.ndim == 3:
        samples = x
    else:
        raise ValueError("x must be [T,N] or [S,T,N]")
    nodes = samples.shape[-1]
    score = np.zeros((nodes, nodes), dtype=np.float64)
    for sample in samples:
        sample_score = np.zeros((nodes, nodes), dtype=np.float64)
        for lag in range(1, max_lag + 1):
            if sample.shape[0] <= lag + 2:
                continue
            past = sample[:-lag]
            future = sample[lag:]
            for target in range(nodes):
                mi = mutual_info_regression(
                    past,
                    future[:, target],
                    n_neighbors=n_neighbors,
                    random_state=seed,
                )
                sample_score[:, target] = np.maximum(sample_score[:, target], mi)
        score += sample_score
    score /= max(1, len(samples))
    score -= score.min()
    score /= score.max() + 1e-8
    return score.astype(np.float32)
