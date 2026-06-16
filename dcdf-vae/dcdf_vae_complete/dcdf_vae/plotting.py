from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _prep_path(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_heatmap(mat: np.ndarray, path: str | Path, title: str, cmap: str = "viridis") -> None:
    path = _prep_path(path)
    plt.figure(figsize=(6, 5))
    sns.heatmap(mat, cmap=cmap, square=True, cbar=True, xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_side_by_side(true_mat: np.ndarray, score_mat: np.ndarray, path: str | Path, title: str) -> None:
    path = _prep_path(path)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(true_mat, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("True graph")
    axes[1].imshow(score_mat, cmap="magma", vmin=float(np.min(score_mat)), vmax=float(np.max(score_mat)))
    axes[1].set_title("Estimated score")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_training_curve(history: list[dict[str, float]], path: str | Path, title: str) -> None:
    path = _prep_path(path)
    df = pd.DataFrame(history)
    plt.figure(figsize=(7, 4))
    for col in ["loss", "rec", "val_loss"]:
        if col in df and not df[col].isna().all():
            plt.plot(df["epoch"], df[col], label=col)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_metric_bar(df: pd.DataFrame, x: str, y: str, path: str | Path, title: str, hue: str | None = None) -> None:
    path = _prep_path(path)
    plt.figure(figsize=(max(7, 0.45 * len(df)), 4.5))
    sns.barplot(data=df, x=x, y=y, hue=hue)
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_difference_heatmap(child: np.ndarray, young: np.ndarray, path: str | Path, title: str) -> None:
    diff = child - young
    vmax = float(np.quantile(np.abs(diff), 0.995)) if diff.size else 1.0
    path = _prep_path(path)
    plt.figure(figsize=(6, 5))
    sns.heatmap(diff, cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax, square=True, xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_dynamic_variance(dynamic_mean: np.ndarray, path: str | Path, title: str) -> None:
    var_mat = dynamic_mean.var(axis=0)
    save_heatmap(var_mat, path, title, cmap="rocket")


def save_table_image(df: pd.DataFrame, path: str | Path, title: str) -> None:
    path = _prep_path(path)
    fig_height = max(2.8, 0.35 * (len(df) + 1))
    fig, ax = plt.subplots(figsize=(min(14, max(8, 1.4 * len(df.columns))), fig_height))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
