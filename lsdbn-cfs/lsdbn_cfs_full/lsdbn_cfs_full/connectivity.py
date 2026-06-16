from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from tqdm import tqdm


def sliding_windows(ts: np.ndarray, window: int = 25, step: int = 5) -> np.ndarray:
    starts = list(range(0, ts.shape[0] - window + 1, step))
    if not starts:
        raise ValueError("No sliding windows can be generated.")
    return np.stack([ts[start:start + window] for start in starts], axis=0)


def flatten_no_self(mat: np.ndarray) -> np.ndarray:
    r = mat.shape[0]
    out = np.empty(r * (r - 1), dtype=np.float32)
    k = 0
    for i in range(r):
        for j in range(r):
            if i == j:
                continue
            out[k] = mat[i, j]
            k += 1
    return out


def gaussian_te_matrix(window_data: np.ndarray, lag: int = 1, eps: float = 1e-8) -> np.ndarray:
    """Vectorized linear-Gaussian transfer entropy matrix.

    Entry [i, j] estimates source ROI i -> target ROI j:
    TE = -0.5 * log(1 - corr(Y_future, X_past | Y_past)^2).
    """
    x = np.asarray(window_data, dtype=np.float64)
    if x.shape[0] <= lag + 2:
        return np.zeros((x.shape[1], x.shape[1]), dtype=np.float32)

    past = x[:-lag]
    future = x[lag:]
    past = past - past.mean(axis=0, keepdims=True)
    future = future - future.mean(axis=0, keepdims=True)

    n_roi = x.shape[1]
    te = np.zeros((n_roi, n_roi), dtype=np.float64)
    x_past = past
    for target in range(n_roi):
        z = past[:, target]
        y = future[:, target]
        zz = float(np.dot(z, z) + eps)
        y_res = y - z * (float(np.dot(z, y)) / zz)
        var_y = float(np.mean(y_res * y_res) + eps)

        beta_x = (z[:, None] * x_past).sum(axis=0) / zz
        x_res = x_past - z[:, None] * beta_x[None, :]
        cov = (x_res * y_res[:, None]).mean(axis=0)
        var_x = (x_res * x_res).mean(axis=0) + eps
        r2 = (cov * cov) / (var_x * var_y)
        r2 = np.clip(r2, 0.0, 1.0 - eps)
        te[:, target] = -0.5 * np.log(1.0 - r2)
    np.fill_diagonal(te, 0.0)
    return te.astype(np.float32)


def subject_dynamic_ec(ts: np.ndarray, window: int = 25, step: int = 5, lag: int = 1, eps: float = 1e-8) -> np.ndarray:
    wins = sliding_windows(ts, window=window, step=step)
    rows = [flatten_no_self(gaussian_te_matrix(win, lag=lag, eps=eps)) for win in wins]
    return np.stack(rows, axis=0).astype(np.float32)


def build_dynamic_ec_dataset(
    signals: np.ndarray,
    subject_indices: np.ndarray,
    window: int = 25,
    step: int = 5,
    lag: int = 1,
    eps: float = 1e-8,
    show_progress: bool = True,
) -> np.ndarray:
    rows = []
    iterator: Iterable[int] = subject_indices
    if show_progress:
        iterator = tqdm(subject_indices, desc="dEC subjects", leave=False)
    for idx in iterator:
        rows.append(subject_dynamic_ec(signals[idx], window=window, step=step, lag=lag, eps=eps))
    return np.concatenate(rows, axis=0).astype(np.float32)


def load_or_build_dynamic_ec(
    cache_path: str,
    signals: np.ndarray,
    split: Tuple[np.ndarray, np.ndarray, np.ndarray],
    window: int,
    step: int,
    lag: int,
    eps: float,
    force: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (not force) and cache_path and __import__("os").path.exists(cache_path):
        z = np.load(cache_path)
        return z["X_train"], z["X_fs"], z["X_test"]

    train_idx, fs_idx, test_idx = split
    X_train = build_dynamic_ec_dataset(signals, train_idx, window, step, lag, eps)
    X_fs = build_dynamic_ec_dataset(signals, fs_idx, window, step, lag, eps)
    X_test = build_dynamic_ec_dataset(signals, test_idx, window, step, lag, eps)
    if cache_path:
        np.savez_compressed(cache_path, X_train=X_train, X_fs=X_fs, X_test=X_test)
    return X_train, X_fs, X_test

