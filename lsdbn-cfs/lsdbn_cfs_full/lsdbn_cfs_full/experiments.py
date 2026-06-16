from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .aal90 import AAL90_NAMES, ROI_NETWORK, edge_name, feature_index_to_edge
from .config import ExperimentConfig
from .connectivity import load_or_build_dynamic_ec
from .data import load_adni_data, n_sliding_windows, repeat_subject_labels, save_split_metadata, stratified_subject_split
from .feature_selection import select_features_by_method
from .metrics import classification_summary
from .model import LSDBNCFS


def _classifier(seed: int):
    return LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=seed)


def _metric_row(method: str, repeat: int, scenario: str, metrics: Dict[str, object], n_features: int) -> Dict[str, object]:
    return {
        "repeat": repeat,
        "model_name": method,
        "scenario": scenario,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "n_features": int(n_features),
        "confusion_matrix": json.dumps(metrics["confusion_matrix"], ensure_ascii=False),
    }


def _evaluate_selected_original_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected: np.ndarray,
    seed: int,
) -> Dict[str, object]:
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train[:, selected]).astype(np.float32)
    Xv = scaler.transform(X_test[:, selected]).astype(np.float32)
    clf = _classifier(seed).fit(Xt, y_train)
    pred = clf.predict(Xv)
    return classification_summary(y_test, pred)


def _select_original_with_method(
    model: LSDBNCFS,
    X_select: np.ndarray,
    y_select: np.ndarray,
    method: str,
    k: int,
) -> np.ndarray:
    if model.input_selected_ is None:
        raise RuntimeError("Model must be fitted before selecting original features.")
    Xs = model.scaler_.transform(X_select).astype(np.float32)
    local_X = Xs[:, model.input_selected_]
    selected_local, _, _ = select_features_by_method(
        local_X,
        y_select,
        k=k,
        method=method,
        random_state=model.config.random_state,
        relief_neighbors=model.config.relief_neighbors,
        ecfs_alpha=model.config.ecfs_alpha,
    )
    return model.input_selected_[selected_local]


def _model_specs(methods: str):
    core = [
        ("DBN", False, False, "jcfs"),
        ("Sparse DBN", True, False, "jcfs"),
        ("JCFS-DBN", False, True, "jcfs"),
        ("LSDBN-CFS w/o BCFS", True, True, "jcfs"),
        ("LSDBN-CFS", True, True, "jcfs"),
    ]
    if methods == "core":
        return core
    return [
        ("DBN", False, False, "jcfs"),
        ("Sparse DBN", True, False, "jcfs"),
        ("JCFS-DBN", False, True, "jcfs"),
        ("DBN+Relief", False, True, "relief"),
        ("DBN+Inf-FS", False, True, "inf-fs"),
        ("DBN+ECFS", False, True, "ecfs"),
        ("Sparse DBN+Relief", True, True, "relief"),
        ("Sparse DBN+Inf-FS", True, True, "inf-fs"),
        ("Sparse DBN+ECFS", True, True, "ecfs"),
        ("LSDBN-CFS w/o BCFS", True, True, "jcfs"),
        ("LSDBN-CFS", True, True, "jcfs"),
    ]


def _baseline_selection_method(model_name: str) -> str:
    lower = model_name.lower()
    if "inf" in lower:
        return "inf-fs"
    if "ecfs" in lower:
        return "ecfs"
    if "relief" in lower:
        return "relief"
    if "jcfs" in lower:
        return "jcfs"
    return "relief"


def _summarize_performance(rows: List[Dict[str, object]], reference_name: str = "LSDBN-CFS") -> pd.DataFrame:
    df = pd.DataFrame(rows)
    groups = []
    for (name, scenario), g in df.groupby(["model_name", "scenario"], sort=False):
        acc = g["accuracy"].to_numpy(dtype=np.float64)
        bal = g["balanced_accuracy"].to_numpy(dtype=np.float64)
        p_value = np.nan
        ref = df[(df["model_name"] == reference_name) & (df["scenario"] == scenario)].sort_values("repeat")
        cur = g.sort_values("repeat")
        if len(ref) == len(cur) and len(cur) >= 2 and name != reference_name:
            try:
                from scipy.stats import ttest_rel

                p_value = float(ttest_rel(ref["accuracy"], cur["accuracy"]).pvalue)
            except Exception:
                p_value = np.nan
        groups.append({
            "model_name": name,
            "scenario": scenario,
            "accuracy_mean": float(acc.mean()),
            "accuracy_std": float(acc.std(ddof=0)),
            "balanced_accuracy_mean": float(bal.mean()),
            "balanced_accuracy_std": float(bal.std(ddof=0)),
            "n_features_mean": float(g["n_features"].mean()),
            "p_value_vs_LSDBN_CFS": p_value,
        })
    return pd.DataFrame(groups)


def _selected_features_table(selected: np.ndarray) -> pd.DataFrame:
    rows = []
    for zero_idx in selected:
        feature_id = int(zero_idx) + 1
        src, tgt = feature_index_to_edge(feature_id)
        src_name = AAL90_NAMES[src - 1]
        tgt_name = AAL90_NAMES[tgt - 1]
        rows.append({
            "feature_id": feature_id,
            "source_roi": src,
            "target_roi": tgt,
            "source_name": src_name,
            "target_name": tgt_name,
            "source_network": ROI_NETWORK[src_name],
            "target_network": ROI_NETWORK[tgt_name],
            "edge": edge_name(src, tgt),
        })
    return pd.DataFrame(rows)


def _roi_frequency_table(selected_edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for roi, name in enumerate(AAL90_NAMES, start=1):
        source_count = int((selected_edges["source_roi"] == roi).sum())
        target_count = int((selected_edges["target_roi"] == roi).sum())
        rows.append({
            "roi": roi,
            "roi_name": name,
            "network": ROI_NETWORK[name],
            "source_count": source_count,
            "target_count": target_count,
            "total_count": source_count + target_count,
        })
    return pd.DataFrame(rows).sort_values(["total_count", "source_count", "target_count"], ascending=False)


def _threshold_scan(
    model: LSDBNCFS,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_fs: np.ndarray,
    y_fs: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: ExperimentConfig,
) -> Tuple[pd.DataFrame, np.ndarray]:
    rows = []
    selected_at_cfg = None
    fs_activations = model.layer_activations(X_fs)
    for kappa in cfg.kappa_values:
        selected = model.bcfs(y_fs, kappa=float(kappa), selected_k=cfg.selected_k, activations=fs_activations)
        metrics = _evaluate_selected_original_features(X_train, y_train, X_test, y_test, selected, cfg.random_state)
        rows.append({
            "kappa": float(kappa),
            "n_selected": int(len(selected)),
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        })
        if abs(float(kappa) - cfg.kappa) < 1e-12:
            selected_at_cfg = selected.copy()
    if selected_at_cfg is None:
        selected_at_cfg = model.bcfs(y_fs, kappa=cfg.kappa, selected_k=cfg.selected_k, activations=fs_activations)
    return pd.DataFrame(rows), selected_at_cfg


def _run_one_repeat(cfg: ExperimentConfig, repeat: int, out_dir: Path) -> Tuple[List[Dict[str, object]], Dict[str, object], np.ndarray]:
    seed = cfg.random_state + repeat
    adni = load_adni_data(cfg.data_path)
    split = stratified_subject_split(
        adni.labels,
        random_state=seed,
        train_ratio=cfg.train_ratio,
        fs_ratio=cfg.fs_ratio,
        test_ratio=cfg.test_ratio,
    )
    train_idx, fs_idx, test_idx = split
    n_windows = n_sliding_windows(adni.signals.shape[1], cfg.window, cfg.step)
    y_train = repeat_subject_labels(adni.labels, train_idx, n_windows)
    y_fs = repeat_subject_labels(adni.labels, fs_idx, n_windows)
    y_test = repeat_subject_labels(adni.labels, test_idx, n_windows)

    cache = out_dir / ("dynamic_ec_cache_seed%d_w%d_s%d_l%d.npz" % (seed, cfg.window, cfg.step, cfg.te_lag))
    X_train, X_fs, X_test = load_or_build_dynamic_ec(
        str(cache),
        adni.signals,
        split,
        window=cfg.window,
        step=cfg.step,
        lag=cfg.te_lag,
        eps=cfg.te_eps,
        force=cfg.force_recompute_dec,
    )
    save_split_metadata(str(out_dir / ("subject_split_seed%d.json" % seed)), adni, split)

    rows: List[Dict[str, object]] = []
    summary: Dict[str, object] = {
        "seed": seed,
        "n_subjects": int(len(adni.labels)),
        "n_windows_per_subject": int(n_windows),
        "train_subjects": int(len(train_idx)),
        "feature_selection_subjects": int(len(fs_idx)),
        "test_subjects": int(len(test_idx)),
        "X_train_shape": list(X_train.shape),
        "X_fs_shape": list(X_fs.shape),
        "X_test_shape": list(X_test.shape),
        "y_train_counts": np.bincount(y_train).tolist(),
        "y_fs_counts": np.bincount(y_fs).tolist(),
        "y_test_counts": np.bincount(y_test).tolist(),
    }

    selected_final = None
    for name, use_sparse, use_jcfs, jcfs_method in _model_specs(cfg.methods):
        print("[%s] fitting %s" % (time.strftime("%H:%M:%S"), name))
        model = LSDBNCFS(cfg, use_sparse=use_sparse, use_jcfs=use_jcfs, jcfs_method=jcfs_method)
        model.fit_representation(X_train, y_train)

        model.fit_classifier(X_train, y_train, selected_features=None)
        pred_abs = model.predict(X_test, selected_features=None)
        metrics_abs = classification_summary(y_test, pred_abs)
        rows.append(_metric_row(name, repeat, "abstract_features", metrics_abs, model.transform_top(X_train).shape[1]))

        if name == "LSDBN-CFS":
            threshold_df, selected = _threshold_scan(model, X_train, y_train, X_fs, y_fs, X_test, y_test, cfg)
            threshold_df.to_csv(out_dir / "threshold_curve.csv", index=False, encoding="utf-8-sig")
            selected_final = selected
        elif name == "LSDBN-CFS w/o BCFS":
            method = "jcfs"
            selected = _select_original_with_method(model, X_fs, y_fs, method, cfg.selected_k)
        else:
            method = _baseline_selection_method(name)
            selected = _select_original_with_method(model, X_fs, y_fs, method, cfg.selected_k)

        metrics_sel = _evaluate_selected_original_features(X_train, y_train, X_test, y_test, selected, seed)
        rows.append(_metric_row(name, repeat, "selected_original_features", metrics_sel, len(selected)))

    if selected_final is None:
        raise RuntimeError("LSDBN-CFS did not run; cannot export final selected features.")
    return rows, summary, selected_final


def run_experiment_suite(cfg: ExperimentConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Output directory: %s" % out_dir.resolve())

    all_rows: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    final_selected = None
    started = time.time()
    for repeat in range(cfg.repeats):
        rows, summary, selected = _run_one_repeat(cfg, repeat, out_dir)
        all_rows.extend(rows)
        summaries.append(summary)
        final_selected = selected

    raw_df = pd.DataFrame(all_rows)
    raw_df.to_csv(out_dir / "model_performance_raw.csv", index=False, encoding="utf-8-sig")
    perf_df = _summarize_performance(all_rows)
    perf_df.to_csv(out_dir / "model_performance.csv", index=False, encoding="utf-8-sig")

    ablation_names = ["DBN", "Sparse DBN", "JCFS-DBN", "LSDBN-CFS w/o BCFS", "LSDBN-CFS"]
    ablation = perf_df[
        (perf_df["model_name"].isin(ablation_names))
        & (perf_df["scenario"].isin(["abstract_features", "selected_original_features"]))
    ].copy()
    ablation.to_csv(out_dir / "ablation_study.csv", index=False, encoding="utf-8-sig")

    if final_selected is not None:
        selected_df = _selected_features_table(final_selected)
        selected_df.to_csv(out_dir / "selected_features.csv", index=False, encoding="utf-8-sig")
        roi_df = _roi_frequency_table(selected_df)
        roi_df.to_csv(out_dir / "roi_frequency.csv", index=False, encoding="utf-8-sig")

    run_summary = {
        "config": cfg.__dict__,
        "repeat_summaries": summaries,
        "elapsed_seconds": float(time.time() - started),
        "outputs": {
            "model_performance": str(out_dir / "model_performance.csv"),
            "ablation_study": str(out_dir / "ablation_study.csv"),
            "threshold_curve": str(out_dir / "threshold_curve.csv"),
            "selected_features": str(out_dir / "selected_features.csv"),
            "roi_frequency": str(out_dir / "roi_frequency.csv"),
        },
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print("\nModel performance:")
    print(perf_df.to_string(index=False))
    print("\nFinished in %.1f seconds" % (time.time() - started))
