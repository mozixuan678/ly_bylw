from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange

from .model import DCDFVAE, DCDFVAEConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_windows(x: np.ndarray, window: int, stride: int = 1, max_windows: int | None = None) -> np.ndarray:
    if x.ndim == 2:
        series = [x]
    elif x.ndim == 3:
        series = list(x)
    else:
        raise ValueError("x must be [T,N] or [S,T,N]")
    windows = []
    for arr in series:
        time = arr.shape[0]
        if time <= window:
            windows.append(arr)
        else:
            for start in range(0, time - window + 1, stride):
                windows.append(arr[start : start + window])
    data = np.stack(windows).astype(np.float32)
    if max_windows is not None and len(data) > max_windows:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(data), size=max_windows, replace=False))
        data = data[idx]
    return data


def current_tau(cfg: DCDFVAEConfig, epoch: int) -> float:
    return max(cfg.tau_min, cfg.tau_start * (cfg.tau_decay**epoch))


def train_model(
    x: np.ndarray,
    cfg: DCDFVAEConfig,
    epochs: int = 80,
    lr: float = 1e-3,
    batch_size: int = 8,
    window: int | None = None,
    stride: int = 8,
    max_windows: int | None = None,
    val_split: float = 0.15,
    device: str | None = None,
    seed: int = 0,
    verbose: bool = True,
    edge_prior: np.ndarray | None = None,
) -> tuple[DCDFVAE, list[dict[str, float]]]:
    set_seed(seed)
    dev = get_device(device)
    if window is None:
        data = x[None, ...] if x.ndim == 2 else x
    else:
        data = make_windows(x, window, stride=stride, max_windows=max_windows)
    tensor = torch.from_numpy(data.astype(np.float32))
    dataset = TensorDataset(tensor)
    if 0.0 < val_split < 0.5 and len(dataset) > 4:
        n_val = max(1, int(round(len(dataset) * val_split)))
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        train_set, val_set = dataset, None
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False) if val_set is not None else None
    model = DCDFVAE(cfg).to(dev)
    prior_tensor = None
    if edge_prior is not None:
        prior_tensor = torch.from_numpy(edge_prior.astype(np.float32)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    iterator = trange(epochs, disable=not verbose)
    for epoch in iterator:
        tau = current_tau(cfg, epoch)
        model.train()
        train_rows = []
        for (xb,) in train_loader:
            xb = xb.to(dev)
            out = model(xb, tau=tau)
            losses = model.loss(xb, out, edge_prior=prior_tensor)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_rows.append([losses[k].item() for k in ["loss", "rec", "kl", "sparse", "temporal", "prior"]])
        train_avg = np.asarray(train_rows, dtype=np.float64).mean(axis=0)
        val_loss = np.nan
        if val_loader is not None:
            val_rows = []
            model.eval()
            with torch.no_grad():
                for (xb,) in val_loader:
                    xb = xb.to(dev)
                    losses = model.loss(xb, model(xb, tau=cfg.tau_min), edge_prior=prior_tensor)
                    val_rows.append(losses["loss"].item())
            val_loss = float(np.mean(val_rows))
        row = {
            "epoch": float(epoch + 1),
            "tau": float(tau),
            "loss": float(train_avg[0]),
            "rec": float(train_avg[1]),
            "kl": float(train_avg[2]),
            "sparse": float(train_avg[3]),
            "temporal": float(train_avg[4]),
            "prior": float(train_avg[5]),
            "val_loss": float(val_loss),
        }
        history.append(row)
        iterator.set_description(f"loss={row['loss']:.4f} val={row['val_loss']:.4f}")
    return model, history


@torch.no_grad()
def infer_dynamic_ec(
    model: DCDFVAE,
    x: np.ndarray,
    batch_size: int = 1,
    device: str | None = None,
) -> np.ndarray:
    dev = get_device(device) if device is not None else next(model.parameters()).device
    model.eval()
    data = x[None, ...] if x.ndim == 2 else x
    loader = DataLoader(TensorDataset(torch.from_numpy(data.astype(np.float32))), batch_size=batch_size)
    chunks = []
    for (xb,) in loader:
        out = model(xb.to(dev), tau=model.cfg.tau_min)
        chunks.append(out["gamma"].detach().cpu().numpy())
    gamma = np.concatenate(chunks, axis=0)
    return gamma[0] if x.ndim == 2 else gamma


@torch.no_grad()
def infer_subject_ec(
    model: DCDFVAE,
    x: np.ndarray,
    batch_size: int = 1,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    gamma = infer_dynamic_ec(model, x, batch_size=batch_size, device=device)
    if gamma.ndim == 3:
        subject_ec = gamma.mean(axis=0, keepdims=True)
        dynamic_mean = gamma
    else:
        subject_ec = gamma.mean(axis=1)
        dynamic_mean = gamma.mean(axis=0)
    return subject_ec.astype(np.float32), dynamic_mean.astype(np.float32)


def save_checkpoint(model: DCDFVAE, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.cfg.to_dict()}, path)
    path.with_suffix(".json").write_text(json.dumps(model.cfg.to_dict(), indent=2), encoding="utf-8")


def load_checkpoint(path: str | Path, device: str | None = None) -> DCDFVAE:
    ckpt = torch.load(path, map_location=device or "cpu")
    cfg = DCDFVAEConfig(**ckpt["config"])
    model = DCDFVAE(cfg)
    model.load_state_dict(ckpt["state_dict"])
    if device is not None:
        model.to(device)
    return model
