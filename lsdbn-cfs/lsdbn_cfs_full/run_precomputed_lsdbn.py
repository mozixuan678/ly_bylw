from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from lsdbn_cfs_full.aal90 import AAL90_NAMES, ROI_NETWORK, edge_name, feature_index_to_edge
from lsdbn_cfs_full.config import ExperimentConfig
from lsdbn_cfs_full.data import load_adni_data, stratified_subject_split
from lsdbn_cfs_full.model import LSDBNCFS


def no_self_indices(n_roi: int = 90) -> np.ndarray:
    keep = []
    for i in range(n_roi):
        for j in range(n_roi):
            if i != j:
                keep.append(i * n_roi + j)
    return np.asarray(keep, dtype=np.int64)


def selected_features_table(selected: np.ndarray) -> pd.DataFrame:
    rows = []
    for zero_idx in selected:
        fid = int(zero_idx) + 1
        src, tgt = feature_index_to_edge(fid)
        src_name = AAL90_NAMES[src - 1]
        tgt_name = AAL90_NAMES[tgt - 1]
        rows.append({
            "feature_id": fid,
            "source_roi": src,
            "target_roi": tgt,
            "source_name": src_name,
            "target_name": tgt_name,
            "source_network": ROI_NETWORK[src_name],
            "target_network": ROI_NETWORK[tgt_name],
            "edge": edge_name(src, tgt),
        })
    return pd.DataFrame(rows)


def roi_frequency(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for roi, name in enumerate(AAL90_NAMES, start=1):
        source_count = int((df["source_roi"] == roi).sum())
        target_count = int((df["target_roi"] == roi).sum())
        rows.append({
            "roi": roi,
            "roi_name": name,
            "network": ROI_NETWORK[name],
            "source_count": source_count,
            "target_count": target_count,
            "total_count": source_count + target_count,
        })
    return pd.DataFrame(rows).sort_values(["total_count", "source_count", "target_count"], ascending=False)


def evaluate(model: LSDBNCFS, X_train, y_train, X_fs, y_fs, X_test, y_test, kappa_values):
    rows = []
    fs_acts = model.layer_activations(X_fs)
    best = None
    for kappa in kappa_values:
        selected = model.bcfs(y_fs, kappa=float(kappa), selected_k=model.config.selected_k, activations=fs_acts)
        model.fit_classifier(X_train, y_train, selected_features=selected)
        pred_fs = model.predict(X_fs, selected_features=selected)
        pred_test = model.predict(X_test, selected_features=selected)
        row = {
            "kappa": float(kappa),
            "n_selected": int(len(selected)),
            "fs_accuracy": float(accuracy_score(y_fs, pred_fs)),
            "fs_balanced_accuracy": float(balanced_accuracy_score(y_fs, pred_fs)),
            "test_accuracy": float(accuracy_score(y_test, pred_test)),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred_test)),
            "test_confusion_matrix": json.dumps(confusion_matrix(y_test, pred_test).tolist()),
        }
        rows.append(row)
        if best is None or (row["fs_accuracy"], row["fs_balanced_accuracy"]) > (best[0]["fs_accuracy"], best[0]["fs_balanced_accuracy"]):
            best = (row, selected.copy())
        print(json.dumps(row, ensure_ascii=False))
    return pd.DataFrame(rows), best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="../feature_vectors.npy")
    parser.add_argument("--labels", default="../expanded_labels.npy")
    parser.add_argument("--data", default="../ADdata.npy")
    parser.add_argument("--out", default="outputs/precomputed_lsdbn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", default="256,128,64")
    parser.add_argument("--keep", default="128,80,48")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--input-prefilter-k", type=int, default=1024)
    parser.add_argument("--input-prefilter-method", default="fisher")
    parser.add_argument("--classifier", default="svm", choices=["logistic", "svm", "random_forest", "extra_trees"])
    parser.add_argument("--selected-k", type=int, default=124)
    parser.add_argument("--kappa-values", default="0,0.005,0.01,0.03,0.05,0.08,0.1")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    X_all = np.load(args.features, mmap_mode="r")
    y_all = np.load(args.labels, mmap_mode="r").astype(np.int64)
    adni = load_adni_data(args.data)
    n_subjects = len(adni.labels)
    n_windows = X_all.shape[0] // n_subjects
    split = stratified_subject_split(adni.labels, random_state=args.seed)
    keep_cols = no_self_indices()

    def rows_for_subjects(subject_indices):
        idx = np.concatenate([np.arange(s * n_windows, (s + 1) * n_windows) for s in subject_indices])
        return idx

    train_rows = rows_for_subjects(split[0])
    fs_rows = rows_for_subjects(split[1])
    test_rows = rows_for_subjects(split[2])
    X_train = np.asarray(X_all[train_rows][:, keep_cols], dtype=np.float32)
    X_fs = np.asarray(X_all[fs_rows][:, keep_cols], dtype=np.float32)
    X_test = np.asarray(X_all[test_rows][:, keep_cols], dtype=np.float32)
    y_train = np.asarray(y_all[train_rows], dtype=np.int64)
    y_fs = np.asarray(y_all[fs_rows], dtype=np.int64)
    y_test = np.asarray(y_all[test_rows], dtype=np.int64)

    cfg = ExperimentConfig()
    cfg.random_state = args.seed
    cfg.hidden_layers = tuple(int(x) for x in args.hidden.split(",") if x)
    cfg.jcfs_keep = tuple(int(x) for x in args.keep.split(",") if x)
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.input_prefilter_k = args.input_prefilter_k
    cfg.input_prefilter_method = args.input_prefilter_method
    cfg.selected_k = args.selected_k
    cfg.classifier = args.classifier
    cfg.device = "auto"
    cfg.m_min = min(50, args.selected_k)

    print("Shapes:", X_train.shape, X_fs.shape, X_test.shape)
    print("Config:", cfg)
    model = LSDBNCFS(cfg, use_sparse=True, use_jcfs=True, jcfs_method="jcfs")
    model.fit_representation(X_train, y_train)

    model.fit_classifier(X_train, y_train, selected_features=None)
    pred_top = model.predict(X_test, selected_features=None)
    top_row = {
        "scenario": "abstract_features",
        "test_accuracy": float(accuracy_score(y_test, pred_top)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred_top)),
        "test_confusion_matrix": json.dumps(confusion_matrix(y_test, pred_top).tolist()),
    }
    print(json.dumps(top_row, ensure_ascii=False))

    kappa_values = [float(x) for x in args.kappa_values.split(",") if x != ""]
    threshold_df, best = evaluate(model, X_train, y_train, X_fs, y_fs, X_test, y_test, kappa_values)
    threshold_df.to_csv(out / "threshold_curve.csv", index=False, encoding="utf-8-sig")
    best_row, selected = best
    selected_df = selected_features_table(selected)
    selected_df.to_csv(out / "selected_features.csv", index=False, encoding="utf-8-sig")
    roi_frequency(selected_df).to_csv(out / "roi_frequency.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([top_row]).to_csv(out / "abstract_feature_performance.csv", index=False, encoding="utf-8-sig")

    summary = {
        "feature_source": str(Path(args.features).resolve()),
        "subject_level_split": True,
        "n_subjects": int(n_subjects),
        "n_windows_per_subject": int(n_windows),
        "train_shape": list(X_train.shape),
        "fs_shape": list(X_fs.shape),
        "test_shape": list(X_test.shape),
        "config": cfg.__dict__,
        "top_abstract": top_row,
        "best_selected_by_fs": best_row,
        "elapsed_seconds": float(time.time() - started),
    }
    with open(out / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Best selected:", json.dumps(best_row, ensure_ascii=False))
    print("Finished in %.1f seconds" % (time.time() - started))


if __name__ == "__main__":
    main()
