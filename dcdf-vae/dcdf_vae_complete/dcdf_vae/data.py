from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import solve_ivp


def standardize(x: np.ndarray, axis: int | tuple[int, ...] = 0) -> np.ndarray:
    return ((x - x.mean(axis=axis, keepdims=True)) / (x.std(axis=axis, keepdims=True) + 1e-8)).astype(
        np.float32
    )


def generate_var(
    n_nodes: int = 20,
    timesteps: int = 500,
    lag: int = 1,
    in_degree: int = 2,
    beta_range: tuple[float, float] = (0.25, 0.65),
    seed: int = 0,
    noise_std: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    adj = np.eye(n_nodes, dtype=np.float32)
    coefs: list[np.ndarray] = []
    for _ in range(lag):
        a = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        for target in range(n_nodes):
            candidates = np.delete(np.arange(n_nodes), target)
            parents = rng.choice(candidates, size=min(in_degree, len(candidates)), replace=False)
            src = np.concatenate([[target], parents])
            weights = rng.uniform(beta_range[0], beta_range[1], size=len(src))
            weights *= rng.choice([-1.0, 1.0], size=len(src))
            a[src, target] = weights / lag
            adj[src, target] = 1.0
        eig = max(abs(np.linalg.eigvals(a))) if np.any(a) else 1.0
        if eig > 0.85:
            a = (a / eig * 0.85).astype(np.float32)
        coefs.append(a)
    burn = 200
    x = np.zeros((timesteps + burn + lag, n_nodes), dtype=np.float32)
    x[:lag] = rng.normal(scale=noise_std, size=(lag, n_nodes))
    for t in range(lag, timesteps + burn + lag):
        val = np.zeros(n_nodes, dtype=np.float32)
        for l_idx, coef in enumerate(coefs, start=1):
            val += x[t - l_idx] @ coef
        x[t] = val + rng.normal(scale=noise_std, size=n_nodes)
    x = standardize(x[burn + lag :], axis=0)
    return x, adj.astype(np.float32), coefs


def _lorenz96_rhs(_, x: np.ndarray, forcing: float) -> np.ndarray:
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + forcing


def generate_lorenz96(
    n_nodes: int = 30,
    timesteps: int = 500,
    forcing: float = 10.0,
    dt: float = 0.05,
    seed: int = 0,
    noise_std: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, None]:
    rng = np.random.default_rng(seed)
    x0 = forcing * np.ones(n_nodes, dtype=np.float64)
    x0 += rng.normal(scale=0.01, size=n_nodes)
    total = (timesteps + 400) * dt
    t_eval = np.arange(0, total, dt)
    sol = solve_ivp(
        lambda t, y: _lorenz96_rhs(t, y, forcing),
        (0, total),
        x0,
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
    )
    x = sol.y.T[-timesteps:]
    x += rng.normal(scale=noise_std, size=x.shape)
    x = standardize(x, axis=0)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for target in range(n_nodes):
        for src in [(target - 2) % n_nodes, (target - 1) % n_nodes, (target + 1) % n_nodes, target]:
            adj[src, target] = 1.0
    return x.astype(np.float32), adj, None


def _as_subject_time_node(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected [S,T,N] or [S,N,T], got shape {arr.shape}")
    # PNC cached arrays in this repo are [subjects, 264, time].
    if arr.shape[1] > arr.shape[2]:
        arr = arr.transpose(0, 2, 1)
    return arr.astype(np.float32)


def load_group_npy(path: str | Path, standardize_subjects: bool = True) -> np.ndarray:
    x = _as_subject_time_node(np.load(path))
    if standardize_subjects:
        x = standardize(x, axis=1)
    return x


def load_pnc_mat(
    path: str | Path,
    age_group_id: int | None = None,
    standardize_subjects: bool = True,
    limit: int | None = None,
) -> np.ndarray:
    """Load MATLAB v7.3 PNC files used in this repo.

    age_group_id follows the stored PNC meta field when present.  Passing None
    returns all subjects.
    """

    path = Path(path)
    with h5py.File(path, "r") as h5:
        root_name = next(k for k in h5.keys() if k != "#refs#")
        root = h5[root_name]
        refs = root["img_time_serie"][:].ravel()
        meta_refs = root["meta"][:].ravel() if "meta" in root else [None] * len(refs)
        samples: list[np.ndarray] = []
        for data_ref, meta_ref in zip(refs, meta_refs):
            if age_group_id is not None and meta_ref is not None:
                meta = h5[meta_ref]
                if "age_grp_id" not in meta:
                    continue
                gid = int(np.asarray(meta["age_grp_id"]).ravel()[0])
                if gid != age_group_id:
                    continue
            arr = np.asarray(h5[data_ref], dtype=np.float32)
            if arr.shape[0] > arr.shape[1]:
                sample = arr
            else:
                sample = arr.T if arr.shape[0] == 264 else arr
            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                break
    if not samples:
        raise ValueError(f"No samples loaded from {path}")
    x = np.stack(samples).astype(np.float32)
    if standardize_subjects:
        x = standardize(x, axis=1)
    return x


def select_nodes_by_variance(x: np.ndarray, n_nodes: int | None) -> tuple[np.ndarray, np.ndarray]:
    if n_nodes is None or n_nodes >= x.shape[-1]:
        return x, np.arange(x.shape[-1])
    flat = x.reshape(-1, x.shape[-1])
    idx = np.argsort(flat.var(axis=0))[-n_nodes:]
    idx = np.sort(idx)
    return x[..., idx], idx
