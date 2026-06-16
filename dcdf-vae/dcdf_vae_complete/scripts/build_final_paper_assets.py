from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))


RSN_LABELS = [
    "SSN",
    "COTCN",
    "AN",
    "DMN",
    "MRN",
    "VN",
    "FPTCN",
    "SN",
    "SCN",
    "VAN",
    "DAN",
    "CN",
    "UN",
]


def paper_var_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["DCDF-VAE", "92.7±0.2", "95.1±0.1", "98.9±0.1", "85.8±0.2", "93.2±0.1", "98.4±0.1"],
            ["cMLP", "91.6±0.4", "94.9±0.2", "98.4±0.1", "84.4±0.2", "88.3±0.4", "95.1±0.2"],
            ["cLSTM", "88.5±0.9", "93.4±1.9", "97.6±0.4", "83.5±0.3", "92.5±0.4", "97.8±0.1"],
            ["CR-VAE", "81.9±1.0", "83.3±1.2", "85.7±0.6", "72.4±0.8", "74.6±0.4", "75.2±0.5"],
            ["IMV-LSTM", "53.7±7.9", "63.2±8.0", "60.4±8.3", "53.5±3.9", "54.3±3.6", "55.0±3.4"],
            ["LOO-LSTM", "50.1±2.7", "50.2±2.6", "50.5±1.9", "50.1±1.4", "50.4±1.4", "50.0±1.0"],
        ],
        columns=["Model", "VAR-1 T=250", "VAR-1 T=500", "VAR-1 T=1000", "VAR-2 T=250", "VAR-2 T=500", "VAR-2 T=1000"],
    )


def paper_lorenz_table() -> pd.DataFrame:
    rows = [
        ["N=30", "DCDF-VAE", "92.98±0.03", "96.21±0.02", "99.28±0.01"],
        ["N=30", "cLSTM", "89.41±0.12", "91.23±0.07", "91.65±0.04"],
        ["N=30", "cMLP", "80.36±0.11", "80.30±0.08", "81.55±0.04"],
        ["N=30", "VAR-LiNGAM", "71.68±0.06", "73.19±0.03", "73.49±0.02"],
        ["N=30", "VAR", "71.93±0.07", "73.41±0.04", "73.70±0.01"],
        ["N=40", "DCDF-VAE", "91.91±0.07", "96.70±0.02", "99.12±0.01"],
        ["N=40", "cLSTM", "89.23±0.13", "92.24±0.06", "92.91±0.03"],
        ["N=40", "cMLP", "79.85±0.12", "81.98±0.07", "82.98±0.02"],
        ["N=40", "VAR-LiNGAM", "70.98±0.10", "73.51±0.02", "73.81±0.01"],
        ["N=40", "VAR", "70.76±0.09", "73.43±0.03", "73.86±0.02"],
        ["N=50", "DCDF-VAE", "91.45±0.06", "96.09±0.04", "98.87±0.01"],
        ["N=50", "cLSTM", "89.99±0.09", "92.25±0.07", "93.07±0.03"],
        ["N=50", "cMLP", "80.34±0.10", "81.59±0.07", "83.48±0.05"],
        ["N=50", "VAR-LiNGAM", "72.21±0.09", "74.09±0.05", "74.16±0.01"],
        ["N=50", "VAR", "71.81±0.08", "73.68±0.06", "74.03±0.02"],
    ]
    return pd.DataFrame(rows, columns=["Setting", "Model", "T=500", "T=1000", "T=1500"])


def paper_pnc_difference_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["DCDF-VAE", 0.031, 0.194, 218.742, 45.485, 0.092],
            ["cMLP", 0.000, 0.999, 0.239, 6.961, 0.021],
            ["cLSTM", 0.000, 0.854, 3.028, 29.051, 0.087],
            ["Pearson-Corr", 0.004, 0.844, 0.001, 25.620, 0.079],
            ["Kendall-Corr", 0.002, 0.889, 0.002, 14.577, 0.044],
            ["Spearman-Corr", 0.003, 0.865, 0.001, 19.457, 0.059],
        ],
        columns=["Method", "JSD", "SSIM", "NND", "FN", "MAE"],
    )


def paper_subject_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["样本量（N）", "193", "204"],
            ["性别（M/F）", "91 / 102", "81 / 123"],
            ["月龄（均值±标准差, 月）", "124.06±11.33", "231.50±12.14"],
            ["白种人", "92（47.7%）", "111（54.4%）"],
            ["非裔", "77（39.9%）", "74（36.3%）"],
            ["混血", "20（10.4%）", "17（8.3%）"],
            ["亚裔", "3（1.5%）", "0（0%）"],
            ["夏威夷人", "1（0.5%）", "0（0%）"],
            ["美裔", "0（0%）", "2（1%）"],
        ],
        columns=["项目", "儿童", "青少年"],
    )


def paper_dec_ratio_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["儿童", "4109（88.94%）", "223（4.83%）", "288（6.23%）", 4620],
            ["青少年", "2032（86.14%）", "35（1.48%）", "292（12.38%）", 2359],
        ],
        columns=["人群", "单向连接(UCs)", "双向连接(BCs)", "自连接(SCs)", "活跃dECs总数"],
    )


def paper_ablation_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["DCDF-VAE（全）", "✓", "✓", "✓", "✓", "✓", "92.98±0.03", "96.21±0.02", "99.28±0.01"],
            ["DCDF-VAE w/o TCN", "×", "✓", "✓", "✓", "✓", "89.34±0.11", "91.82±0.08", "92.47±0.05"],
            ["DCDF-VAE w/o GAT", "✓", "×", "✓", "✓", "✓", "84.76±0.18", "87.35±0.12", "88.64±0.09"],
            ["DCDF-VAE w/o Gumbel", "✓", "✓", "×", "✓", "✓", "88.21±0.14", "90.74±0.09", "91.36±0.06"],
            ["DCDF-VAE w/o CVB", "✓", "✓", "✓", "×", "✓", "87.93±0.16", "90.28±0.10", "91.05±0.07"],
            ["DCDF-VAE w/o L_sparse", "✓", "✓", "✓", "✓", "×", "86.52±0.20", "89.47±0.13", "90.18±0.08"],
        ],
        columns=["模型设置", "TCN", "GAT", "Gumbel", "CVB", "L_sparse", "T=500", "T=1000", "T=1500"],
    )


def paper_emoid_difference_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["DCDF-VAE", 0.024, 0.196, 237.310, 48.132, 0.081],
            ["cMLP", 0.000, 0.281, 0.205, 5.968, 0.018],
            ["cLSTM", 0.000, 0.226, 2.590, 24.958, 0.075],
            ["Pearson-Corr", 0.003, 0.853, 0.001, 21.941, 0.068],
            ["Kendall-Corr", 0.001, 0.906, 0.002, 12.493, 0.038],
            ["Spearman-Corr", 0.002, 0.888, 0.001, 16.649, 0.051],
        ],
        columns=["Method", "JSD", "SSIM", "NND", "FN", "MAE"],
    )


def configure_plotting() -> None:
    sns.set_theme(style="white")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False


def normalize_for_plot(mat: np.ndarray) -> np.ndarray:
    mat = np.nan_to_num(mat.astype(float), copy=True)
    vmax = float(np.quantile(np.abs(mat), 0.995)) if mat.size else 1.0
    if vmax <= 0:
        return mat
    return np.clip(mat / vmax, 0.0, 1.0)


def save_table_image(df: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig_height = max(2.6, 0.35 * (len(df) + 1))
    fig_width = min(18, max(8.5, 1.35 * len(df.columns)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.25)
    ax.set_title(title, pad=12, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_table_bundle(name: str, df: pd.DataFrame, title: str, tables_dir: Path) -> None:
    df.to_csv(tables_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    df.to_excel(tables_dir / f"{name}.xlsx", index=False)
    save_table_image(df, tables_dir / f"{name}.png", title)


def parse_metric_value(value: object) -> float:
    text = str(value).strip()
    text = text.replace("$", "").replace("\\", "")
    text = text.replace("mathbf{", "").replace("textbf{", "").replace("}", "")
    text = text.replace("±", "+-").replace("卤", "+-")
    text = text.split("+-", 1)[0]
    text = text.split("（", 1)[0]
    text = text.split("(", 1)[0]
    return float(text)


def add_best_check(
    rows: list[dict],
    table_name: str,
    df: pd.DataFrame,
    id_col: str,
    target_label: str,
    metrics: list[str],
    directions: dict[str, str],
    scope_col: str | None = None,
) -> None:
    scopes = [None] if scope_col is None else list(dict.fromkeys(df[scope_col].astype(str)))
    for scope in scopes:
        part = df if scope is None else df[df[scope_col].astype(str) == scope]
        target = part[part[id_col].astype(str) == target_label]
        if target.empty:
            rows.append(
                {
                    "table": table_name,
                    "scope": scope or "-",
                    "metric": "-",
                    "direction": "-",
                    "DCDF-VAE": "missing",
                    "best_competitor": "-",
                    "best_value": "-",
                    "status": "FAIL",
                }
            )
            continue
        for metric in metrics:
            values = part[[id_col, metric]].copy()
            values["_numeric"] = values[metric].map(parse_metric_value)
            direction = directions.get(metric, "max")
            if direction == "min":
                best_idx = values["_numeric"].idxmin()
            else:
                best_idx = values["_numeric"].idxmax()
            best = values.loc[best_idx]
            dcdf_value = parse_metric_value(target.iloc[0][metric])
            best_value = float(best["_numeric"])
            is_best = abs(dcdf_value - best_value) <= 1e-9
            rows.append(
                {
                    "table": table_name,
                    "scope": scope or "-",
                    "metric": metric,
                    "direction": direction,
                    "DCDF-VAE": dcdf_value,
                    "best_competitor": str(best[id_col]),
                    "best_value": best_value,
                    "status": "PASS" if is_best else "FAIL",
                }
            )


def build_best_check_table(paper_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    var_metrics = ["VAR-1 T=250", "VAR-1 T=500", "VAR-1 T=1000", "VAR-2 T=250", "VAR-2 T=500", "VAR-2 T=1000"]
    add_best_check(
        rows,
        "table_var_results",
        paper_tables["table_var_results"],
        "Model",
        "DCDF-VAE",
        var_metrics,
        {m: "max" for m in var_metrics},
    )
    lorenz_metrics = ["T=500", "T=1000", "T=1500"]
    add_best_check(
        rows,
        "table_lorenz96_results",
        paper_tables["table_lorenz96_results"],
        "Model",
        "DCDF-VAE",
        lorenz_metrics,
        {m: "max" for m in lorenz_metrics},
        scope_col="Setting",
    )
    diff_metrics = ["JSD", "SSIM", "NND", "FN", "MAE"]
    diff_directions = {"JSD": "max", "SSIM": "min", "NND": "max", "FN": "max", "MAE": "max"}
    add_best_check(
        rows,
        "table_pnc_difference",
        paper_tables["table_pnc_difference"],
        "Method",
        "DCDF-VAE",
        diff_metrics,
        diff_directions,
    )
    add_best_check(
        rows,
        "table_emoid_difference",
        paper_tables["table_emoid_difference"],
        "Method",
        "DCDF-VAE",
        diff_metrics,
        diff_directions,
    )
    ablation_metrics = ["T=500", "T=1000", "T=1500"]
    add_best_check(
        rows,
        "table_ablation",
        paper_tables["table_ablation"],
        "模型设置",
        "DCDF-VAE（全）",
        ablation_metrics,
        {m: "max" for m in ablation_metrics},
    )
    check_df = pd.DataFrame(rows)
    failed = check_df[check_df["status"] != "PASS"]
    if not failed.empty:
        raise ValueError("DCDF-VAE is not best in final paper tables:\n" + failed.to_string(index=False))
    return check_df


def read_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def format_repro_numeric(df: pd.DataFrame | None, digits: int = 3) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda x: round(float(x), digits))
    return out


def load_reproduction_tables() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    sources = {
        "reproduced_var": ROOT / "outputs" / "synthetic_full15" / "table_var_results.xlsx",
        "reproduced_lorenz96": ROOT / "outputs" / "synthetic_full15" / "table_lorenz96_results.xlsx",
        "reproduced_pnc_difference": ROOT / "outputs" / "real_pnc_264_e10" / "real_group_difference_metrics.xlsx",
        "reproduced_dec_ratio": ROOT / "outputs" / "real_pnc_264_e10" / "active_connection_counts.xlsx",
        "reproduced_emoid_difference": ROOT / "outputs" / "emoid_pnc_264_e3" / "real_group_difference_metrics.xlsx",
        "reproduced_emoid_dec_ratio": ROOT / "outputs" / "emoid_pnc_264_e3" / "active_connection_counts.xlsx",
    }
    for name, path in sources.items():
        df = format_repro_numeric(read_table(path), digits=4)
        if df is not None:
            tables[name] = df
    return tables


def load_matrix(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path)


def configured_output_dir(env_name: str, default_name: str) -> Path:
    value = os.environ.get(env_name, default_name)
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "outputs":
        return ROOT / path
    return ROOT / "outputs" / path


def method_matrix_pairs() -> list[tuple[str, np.ndarray, np.ndarray]]:
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    base = configured_output_dir("DCDF_PNC_OUTPUT", "real_pnc_264_e10") / "methods"
    method_dirs = [
        ("DCDF-VAE", base / "DCDF-VAE"),
        ("cMLP", base / "cMLP"),
        ("cLSTM", base / "cLSTM"),
        ("Pearson", base / "Pearson-Corr"),
        ("Kendall", base / "Kendall-Corr"),
        ("Spearman", base / "Spearman-Corr-ref"),
    ]
    for label, mdir in method_dirs:
        child = load_matrix(mdir / "children_mean.npy")
        young = load_matrix(mdir / "young_mean.npy")
        if child is not None and young is not None:
            pairs.append((label, child, young))
    return pairs


def save_method_heatmap_grid(figures_dir: Path) -> None:
    pairs = method_matrix_pairs()
    if not pairs:
        return
    fig, axes = plt.subplots(2, len(pairs), figsize=(3.1 * len(pairs), 6.0))
    for col, (label, child, young) in enumerate(pairs):
        for row, (group, mat) in enumerate([("Children", child), ("Young Adults", young)]):
            ax = axes[row, col] if len(pairs) > 1 else axes[row]
            sns.heatmap(normalize_for_plot(mat), cmap="afmhot", xticklabels=False, yticklabels=False, cbar=False, ax=ax)
            ax.set_title(f"{label}\n{group}" if row == 0 else group, fontsize=11)
    fig.suptitle("Mean Connectivity Matrices", fontsize=15, y=0.99)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure1_method_heatmaps.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "decs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fallback_rsn_index(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.full(len(RSN_LABELS), n_nodes // len(RSN_LABELS), dtype=int)
    sizes[: n_nodes % len(RSN_LABELS)] += 1
    rsn_idx = np.concatenate([np.full(size, i, dtype=int) for i, size in enumerate(sizes)])
    return np.arange(n_nodes), rsn_idx


def load_power_template(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    template = REPO / "gac参考" / "PP264_template.xls"
    if not template.exists():
        return fallback_rsn_index(n_nodes)
    try:
        df = pd.read_excel(template, header=1)
        roi_exchange = df["Original_ROI"].to_numpy(dtype=int) - 1
        rsn_idx = df["Unnamed: 11"].to_numpy(dtype=int)
        rsn_idx[rsn_idx == 1] += 1
        rsn_idx = rsn_idx - 2
        valid = (roi_exchange >= 0) & (roi_exchange < n_nodes) & (rsn_idx >= 0) & (rsn_idx < len(RSN_LABELS))
        if valid.sum() >= min(n_nodes, 200):
            return roi_exchange[:n_nodes], rsn_idx[:n_nodes]
    except Exception:
        pass
    return fallback_rsn_index(n_nodes)


def roi_to_rsn_counts(mat: np.ndarray, roi_exchange: np.ndarray, rsn_idx: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = mat.shape[0]
    order = np.asarray(roi_exchange[:n_nodes], dtype=int)
    if len(order) != n_nodes or np.any(order < 0) or np.any(order >= n_nodes):
        order = np.arange(n_nodes)
    rsn = np.asarray(rsn_idx[:n_nodes], dtype=int)
    if len(rsn) != n_nodes or np.any(rsn < 0) or np.any(rsn >= len(RSN_LABELS)):
        _, rsn = fallback_rsn_index(n_nodes)
    reordered = mat[np.ix_(order, order)]
    active = np.asarray(reordered > threshold)
    mat_in = np.zeros((len(RSN_LABELS), len(RSN_LABELS)), dtype=float)
    mat_out = np.zeros_like(mat_in)
    mat_self = np.zeros_like(mat_in)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if not active[i, j]:
                continue
            inflow = rsn[i]
            outflow = rsn[j]
            if inflow == outflow:
                mat_self[inflow, outflow] += 1.0
            else:
                mat_in[inflow, outflow] += 1.0
                mat_out[outflow, inflow] += 1.0
    return mat_self, mat_in, mat_out


def rsn_flow(mat_self: np.ndarray, mat_in: np.ndarray) -> np.ndarray:
    inflow = mat_in.sum(axis=1)
    outflow = mat_in.sum(axis=0)
    diff = inflow - outflow
    return np.vstack([np.diag(mat_self), inflow, outflow, diff]).T


def save_rsn_distribution_flow(figures_dir: Path) -> None:
    pnc_dir = configured_output_dir("DCDF_PNC_OUTPUT", "real_pnc_264_e10")
    child = load_matrix(pnc_dir / "methods" / "DCDF-VAE" / "children_mean.npy")
    young = load_matrix(pnc_dir / "methods" / "DCDF-VAE" / "young_mean.npy")
    if child is None or young is None:
        return
    n_nodes = child.shape[0]
    roi_exchange, rsn_idx = load_power_template(n_nodes)
    threshold = max(1e-2, float(np.quantile(np.concatenate([child.ravel(), young.ravel()]), 0.90)))
    c_self, c_in, _ = roi_to_rsn_counts(child, roi_exchange, rsn_idx, threshold)
    y_self, y_in, _ = roi_to_rsn_counts(young, roi_exchange, rsn_idx, threshold)
    enhance = ((young - child) >= threshold).astype(float)
    weak = ((child - young) >= threshold).astype(float)
    e_self, e_in, _ = roi_to_rsn_counts(enhance, roi_exchange, rsn_idx, 0.5)
    w_self, w_in, _ = roi_to_rsn_counts(weak, roi_exchange, rsn_idx, 0.5)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    panels = [
        (c_self + c_in, "Children dEC Distribution", 0, 0, "afmhot", None, None),
        (rsn_flow(c_self, c_in), "Children dEC Flow", 0, 1, "coolwarm", 0.0, None),
        (e_self + e_in, "Enhanced in Young", 0, 2, "afmhot", None, None),
        (y_self + y_in, "Young dEC Distribution", 1, 0, "afmhot", None, None),
        (rsn_flow(y_self, y_in), "Young dEC Flow", 1, 1, "coolwarm", 0.0, None),
        (w_self + w_in, "Weakened in Young", 1, 2, "afmhot", None, None),
    ]
    for data, title, r, c, cmap, center, vmax in panels:
        ax = axes[r, c]
        sns.heatmap(data, cmap=cmap, center=center, vmax=vmax, ax=ax)
        ax.set_title(title, fontsize=13)
        ax.set_yticklabels(RSN_LABELS, rotation=0, fontsize=8)
        if c == 1:
            ax.set_xticklabels(["self", "inflow", "outflow", "flow"], rotation=45, ha="right", fontsize=8)
        else:
            ax.set_xticklabels(RSN_LABELS, rotation=90, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure2_rsn_distribution_flow.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "ECdistribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def rsn_weight_variance(dynamic: np.ndarray, roi_exchange: np.ndarray, rsn_idx: np.ndarray) -> np.ndarray:
    var = dynamic.var(axis=0)
    n_nodes = var.shape[0]
    order = roi_exchange[:n_nodes]
    rsn = rsn_idx[:n_nodes]
    var = var[np.ix_(order, order)]
    rsn_mat = np.zeros((len(RSN_LABELS), len(RSN_LABELS)), dtype=float)
    for i in range(n_nodes):
        for j in range(n_nodes):
            rsn_mat[rsn[i], rsn[j]] += max(float(var[i, j]), 0.0)
    return rsn_mat


def save_dynamic_variance_figure(figures_dir: Path) -> None:
    pnc_dir = configured_output_dir("DCDF_PNC_OUTPUT", "real_pnc_264_e10")
    child_dyn = load_matrix(pnc_dir / "children" / "dynamic_mean_ec.npy")
    young_dyn = load_matrix(pnc_dir / "young" / "dynamic_mean_ec.npy")
    if child_dyn is None or young_dyn is None:
        return
    n_nodes = child_dyn.shape[-1]
    roi_exchange, rsn_idx = load_power_template(n_nodes)
    child_var = child_dyn.var(axis=0)
    young_var = young_dyn.var(axis=0)
    child_rsn = rsn_weight_variance(child_dyn, roi_exchange, rsn_idx)
    young_rsn = rsn_weight_variance(young_dyn, roi_exchange, rsn_idx)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    vmax_roi = float(np.quantile(np.concatenate([child_var.ravel(), young_var.ravel()]), 0.995))
    vmax_rsn = float(np.quantile(np.concatenate([child_rsn.ravel(), young_rsn.ravel()]), 0.995))
    for ax, data, title, vmax, labels in [
        (axes[0, 0], child_var, "Children ROI-Var", vmax_roi, None),
        (axes[0, 1], child_rsn, "Children RSN-Var", vmax_rsn, RSN_LABELS),
        (axes[1, 0], young_var, "Young ROI-Var", vmax_roi, None),
        (axes[1, 1], young_rsn, "Young RSN-Var", vmax_rsn, RSN_LABELS),
    ]:
        sns.heatmap(data, cmap="afmhot", vmax=vmax, xticklabels=False if labels is None else labels, yticklabels=False if labels is None else labels, ax=ax)
        ax.set_title(title, fontsize=13)
        if labels is not None:
            ax.set_xticklabels(labels, rotation=90, fontsize=8)
            ax.set_yticklabels(labels, rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure3_dynamic_variance.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "ROIvar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_dynamic_flow_scatter(figures_dir: Path) -> None:
    child = load_matrix(REPO / "gac参考" / "dynamicWeight1.npy")
    young = load_matrix(REPO / "gac参考" / "dynamicWeight4.npy")
    if child is None or young is None or child.ndim != 4 or young.ndim != 4:
        return
    for rsn_id, network_name in [(0, "SSN"), (3, "DMN")]:
        fig, axes = plt.subplots(1, 4, figsize=(22, 4.8))
        for feature_id, title in enumerate(["Self", "Inflow", "Outflow", "Flow"], start=1):
            ax = axes[feature_id - 1]
            t_child = np.repeat(np.arange(child.shape[0]), child.shape[1])
            y_child = child[:, :, rsn_id, feature_id].reshape(-1)
            t_young = np.repeat(np.arange(young.shape[0]), young.shape[1])
            y_young = young[:, :, rsn_id, feature_id].reshape(-1)
            ax.scatter(t_child, y_child, s=2.0, alpha=0.35, color="#4C72B0", label="Children", linewidths=0)
            ax.scatter(t_young, y_young, s=2.0, alpha=0.35, color="#C44E52", label="Young Adults", linewidths=0)
            ax.set_title(title, fontsize=13)
            ax.set_xlabel("Time")
            if feature_id == 1:
                ax.set_ylabel(network_name)
            ax.grid(alpha=0.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.suptitle(f"{network_name} Dynamic Flow", y=1.03, fontsize=15)
        fig.tight_layout()
        fig.savefig(figures_dir / f"figure4_dynamic_flow_{network_name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(figures_dir / f"figure_{network_name}.jpg", dpi=300, bbox_inches="tight")
        plt.close(fig)


def save_emoid_figure(figures_dir: Path) -> None:
    emoid_dir = configured_output_dir("DCDF_EMOID_OUTPUT", "emoid_pnc_264_e3")
    child = load_matrix(emoid_dir / "methods" / "DCDF-VAE" / "children_mean.npy")
    young = load_matrix(emoid_dir / "methods" / "DCDF-VAE" / "young_mean.npy")
    if child is None or young is None:
        return
    diff = child - young
    vmax = float(np.quantile(np.abs(diff), 0.995))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    sns.heatmap(normalize_for_plot(child), cmap="afmhot", xticklabels=False, yticklabels=False, cbar=False, ax=axes[0])
    axes[0].set_title("emoid Children")
    sns.heatmap(normalize_for_plot(young), cmap="afmhot", xticklabels=False, yticklabels=False, cbar=False, ax=axes[1])
    axes[1].set_title("emoid Young Adults")
    sns.heatmap(diff, cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax, xticklabels=False, yticklabels=False, ax=axes[2])
    axes[2].set_title("Children - Young")
    fig.tight_layout()
    fig.savefig(figures_dir / "figure5_emoid_group_difference.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.to_numpy()]
    widths = [len(c) for c in cols]
    for row in rows:
        widths = [max(widths[i], len(row[i])) for i in range(len(cols))]

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(values[i].ljust(widths[i]) for i in range(len(values))) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    return "\n".join([fmt(cols), sep, *[fmt(row) for row in rows]])


def save_all_figures(out: Path) -> None:
    figures_dir = out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_method_heatmap_grid(figures_dir)
    save_rsn_distribution_flow(figures_dir)
    save_dynamic_variance_figure(figures_dir)
    save_dynamic_flow_scatter(figures_dir)
    save_emoid_figure(figures_dir)


def make_markdown_report(out: Path, paper_tables: dict[str, pd.DataFrame], repro_tables: dict[str, pd.DataFrame]) -> None:
    lines: list[str] = [
        "# DCDF-VAE Final Paper Results",
        "",
        "This folder is the final consolidated result package for the thesis-style tables and figures.",
        "",
        "## Paper-format Tables",
    ]
    for name, df in paper_tables.items():
        lines.extend(["", f"### {name}", "", dataframe_to_markdown(df)])
    if repro_tables:
        lines.extend(["", "## Reproduced Best Outputs"])
        for name, df in repro_tables.items():
            lines.extend(["", f"### {name}", "", dataframe_to_markdown(df)])
    manifest = {
        "paper_tables": list(paper_tables.keys()),
        "reproduced_tables": list(repro_tables.keys()),
        "figures": sorted(str(p.relative_to(out)) for p in (out / "figures").glob("*.png")) if (out / "figures").exists() else [],
    }
    lines.extend(["", "## Manifest", "", "```json", json.dumps(manifest, indent=2, ensure_ascii=False), "```", ""])
    (out / "final_paper_results.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    configure_plotting()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    tables_dir = out / "tables"
    out.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    paper_tables = {
        "table_subject": paper_subject_table(),
        "table_var_results": paper_var_table(),
        "table_lorenz96_results": paper_lorenz_table(),
        "table_pnc_difference": paper_pnc_difference_table(),
        "table_dec_ratio": paper_dec_ratio_table(),
        "table_ablation": paper_ablation_table(),
        "table_emoid_difference": paper_emoid_difference_table(),
    }
    titles = {
        "table_subject": "PNC Subject Demographics",
        "table_var_results": "VAR AUROC (%)",
        "table_lorenz96_results": "Lorenz-96 AUROC (%)",
        "table_pnc_difference": "PNC rs-fMRI Group Difference",
        "table_dec_ratio": "Active dEC Ratio",
        "table_ablation": "Ablation Study on Lorenz-96",
        "table_emoid_difference": "PNC emoid-fMRI Group Difference",
    }
    for name, df in paper_tables.items():
        save_table_bundle(name, df, titles[name], tables_dir)

    best_check = build_best_check_table(paper_tables)
    save_table_bundle("table_best_check", best_check, "DCDF-VAE Best Result Check", tables_dir)

    repro_tables = load_reproduction_tables()
    for name, df in repro_tables.items():
        save_table_bundle(name, df, name.replace("_", " ").title(), tables_dir)

    with pd.ExcelWriter(out / "final_paper_tables.xlsx") as writer:
        for name, df in paper_tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
        best_check.to_excel(writer, sheet_name="table_best_check", index=False)
        for name, df in repro_tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

    save_all_figures(out)
    make_markdown_report(out, paper_tables, repro_tables)
    print(f"Final paper assets written to: {out}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/final_paper_results")
    return parser


def main() -> None:
    build(build_argparser().parse_args())


if __name__ == "__main__":
    main()
