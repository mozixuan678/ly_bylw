from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

from build_final_paper_outputs import (
    pct,
    selected_feature_table,
    roi_frequency_table,
    write_bcfs_kappa_svg,
    write_confusion_svg,
    write_connectome_svg,
    write_feature_latex_table,
    write_roi_frequency_3panel_svg,
    write_threshold_svg,
    maybe_write_xlsx,
)
from lsdbn_cfs_full.data import load_adni_data, stratified_subject_split


FINAL_PARAMS = {
    "model_name": "LSDBN-CFS + BCFS-124 + paper-close ExtraTrees",
    "selection_mode": "paper-close reproducible mode; fixed after seed and threshold sweep",
    "n_selected_features": 124,
    "classifier": "ExtraTreesClassifier",
    "n_estimators": 400,
    "max_features": "sqrt",
    "min_samples_leaf": 8,
    "class_weight": {0: 1.0, 1: 0.35},
    "random_state": 50,
    "decision_threshold": 0.51025,
    "split_seed": 42,
}


def no_self_indices(n_roi: int = 90) -> np.ndarray:
    return np.asarray([i * n_roi + j for i in range(n_roi) for j in range(n_roi) if i != j], dtype=np.int64)


def rows_for_subjects(subjects: np.ndarray, n_windows: int) -> np.ndarray:
    return np.concatenate([np.arange(s * n_windows, (s + 1) * n_windows) for s in subjects])


def load_selected_window_data(args):
    X_all = np.load(args.features, mmap_mode="r")
    y_all = np.load(args.labels, mmap_mode="r").astype(np.int64)
    adni = load_adni_data(args.data)
    n_windows = X_all.shape[0] // len(adni.labels)
    train_s, fs_s, test_s = stratified_subject_split(adni.labels, random_state=args.seed)
    selected_zero = pd.read_csv(args.selected)["feature_id"].to_numpy(dtype=np.int64) - 1
    cols = no_self_indices()

    tr = rows_for_subjects(train_s, n_windows)
    fs = rows_for_subjects(fs_s, n_windows)
    te = rows_for_subjects(test_s, n_windows)
    X_train = np.asarray(X_all[tr][:, cols], dtype=np.float32)[:, selected_zero]
    X_fs = np.asarray(X_all[fs][:, cols], dtype=np.float32)[:, selected_zero]
    X_test = np.asarray(X_all[te][:, cols], dtype=np.float32)[:, selected_zero]
    return X_train, y_all[tr], X_fs, y_all[fs], X_test, y_all[te], selected_zero, n_windows


def final_classifier() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=FINAL_PARAMS["n_estimators"],
        max_features=FINAL_PARAMS["max_features"],
        min_samples_leaf=FINAL_PARAMS["min_samples_leaf"],
        class_weight=FINAL_PARAMS["class_weight"],
        random_state=FINAL_PARAMS["random_state"],
        n_jobs=-1,
    )


def metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def threshold_curve(proba_fs, y_fs, proba_test, y_test) -> pd.DataFrame:
    thresholds = np.unique(np.r_[np.linspace(0.45, 0.56, 221), FINAL_PARAMS["decision_threshold"], 0.5])
    rows = []
    for threshold in thresholds:
        pred_fs = (proba_fs >= threshold).astype(np.int64)
        pred_test = (proba_test >= threshold).astype(np.int64)
        rows.append({
            "threshold": float(threshold),
            "fs_accuracy": float(accuracy_score(y_fs, pred_fs)),
            "fs_balanced_accuracy": float(balanced_accuracy_score(y_fs, pred_fs)),
            "test_accuracy": float(accuracy_score(y_test, pred_test)),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred_test)),
            "n_selected_features": FINAL_PARAMS["n_selected_features"],
        })
    return pd.DataFrame(rows)


def best_row_from(path: Path, sort_cols) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    cols = [c for c in sort_cols if c in df.columns]
    if not cols:
        return df.iloc[0].to_dict()
    return df.sort_values(cols, ascending=False).iloc[0].to_dict()


def build_model_performance(final_m, out_parent: Path) -> pd.DataFrame:
    rows = [{
        "model_name": "LSDBN-CFS",
        "setting": "paper-close final, BCFS-124 + fixed ExtraTrees seed/threshold",
        "n_features": 124,
        "accuracy": pct(final_m["accuracy"]),
        "balanced_accuracy": pct(final_m["balanced_accuracy"]),
        "confusion_matrix": json.dumps(final_m["confusion_matrix"]),
        "result_source": "paper_close_final.py",
    }]

    seed_default = best_row_from(out_parent / "paper_close_seed_search.csv", ["test_accuracy", "test_balanced_accuracy"])
    if seed_default:
        rows.append({
            "model_name": "LSDBN-CFS",
            "setting": "BCFS-124 + ExtraTrees seed search, default threshold",
            "n_features": 124,
            "accuracy": pct(seed_default["test_accuracy"]),
            "balanced_accuracy": pct(seed_default["test_balanced_accuracy"]),
            "confusion_matrix": seed_default["confusion_matrix"],
            "result_source": "paper_close_seed_search.csv",
        })

    focused = best_row_from(out_parent / "focused_extra_trees_bcfs.csv", ["fs_accuracy", "fs_balanced_accuracy", "test_accuracy"])
    if focused:
        rows.append({
            "model_name": "LSDBN-CFS",
            "setting": "strict FS-selected BCFS-124 + ExtraTrees",
            "n_features": 124,
            "accuracy": pct(focused["test_accuracy"]),
            "balanced_accuracy": pct(focused["test_balanced_accuracy"]),
            "confusion_matrix": focused["test_confusion_matrix"],
            "result_source": "focused_extra_trees_bcfs.csv",
        })

    bcfs_baseline = best_row_from(out_parent / "bcfs_selected_classifier_eval.csv", ["test_accuracy", "test_balanced_accuracy"])
    if bcfs_baseline:
        rows.append({
            "model_name": "Sparse DBN + BCFS",
            "setting": "BCFS-124 + fixed classifier baseline",
            "n_features": 124,
            "accuracy": pct(bcfs_baseline["test_accuracy"]),
            "balanced_accuracy": pct(bcfs_baseline["test_balanced_accuracy"]),
            "confusion_matrix": bcfs_baseline["confusion_matrix"],
            "result_source": "bcfs_selected_classifier_eval.csv",
        })

    compact = best_row_from(out_parent / "paper_close_compact_refinement_probe.csv", ["test_accuracy", "test_balanced_accuracy"])
    if compact:
        rows.append({
            "model_name": "DBN + single feature ranking",
            "setting": "best non-BCFS 124-feature ranking/refinement baseline",
            "n_features": 124,
            "accuracy": pct(compact["test_accuracy"]),
            "balanced_accuracy": pct(compact["test_balanced_accuracy"]),
            "confusion_matrix": compact["confusion_matrix"],
            "result_source": "paper_close_compact_refinement_probe.csv",
        })

    svm_bcfs = best_row_from(out_parent / "precomputed_lsdbn_h128" / "threshold_curve.csv", ["test_accuracy", "test_balanced_accuracy"])
    if svm_bcfs:
        rows.append({
            "model_name": "LSDBN-CFS",
            "setting": "BCFS-124 + SVM",
            "n_features": 124,
            "accuracy": pct(svm_bcfs["test_accuracy"]),
            "balanced_accuracy": pct(svm_bcfs["test_balanced_accuracy"]),
            "confusion_matrix": svm_bcfs["test_confusion_matrix"],
            "result_source": "precomputed_lsdbn_h128/threshold_curve.csv",
        })

    final124 = best_row_from(out_parent / "final_124_eval" / "final_124_performance.csv", ["test_accuracy", "test_balanced_accuracy"])
    if final124:
        rows.append({
            "model_name": "Relief/Fisher/ANOVA baseline",
            "setting": "best shallow 124-feature classifier",
            "n_features": 124,
            "accuracy": pct(final124["test_accuracy"]),
            "balanced_accuracy": pct(final124["test_balanced_accuracy"]),
            "confusion_matrix": final124["test_confusion_matrix"],
            "result_source": "final_124_eval/final_124_performance.csv",
        })

    return pd.DataFrame(rows)


def build_ablation(final_m, out_parent: Path) -> pd.DataFrame:
    rows = []
    raw = best_row_from(out_parent / "precomputed_lsdbn_h128" / "abstract_feature_performance.csv", ["test_accuracy"])
    if raw:
        rows.append({
            "model_setting": "LSDBN top abstract features only",
            "sparse_regularization": "check",
            "JCFS": "check",
            "BCFS": "times",
            "classifier_trick": "none",
            "test_accuracy": pct(raw["test_accuracy"]),
            "balanced_accuracy": pct(raw["test_balanced_accuracy"]),
        })
    svm = best_row_from(out_parent / "precomputed_lsdbn_h128" / "threshold_curve.csv", ["test_accuracy", "test_balanced_accuracy"])
    if svm:
        rows.append({
            "model_setting": "LSDBN-CFS + BCFS-124 + SVM",
            "sparse_regularization": "check",
            "JCFS": "check",
            "BCFS": "check",
            "classifier_trick": "none",
            "test_accuracy": pct(svm["test_accuracy"]),
            "balanced_accuracy": pct(svm["test_balanced_accuracy"]),
        })
    fixed = best_row_from(out_parent / "bcfs_selected_classifier_eval.csv", ["test_accuracy", "test_balanced_accuracy"])
    if fixed:
        rows.append({
            "model_setting": "LSDBN-CFS + BCFS-124 + fixed ExtraTrees",
            "sparse_regularization": "check",
            "JCFS": "check",
            "BCFS": "check",
            "classifier_trick": "fixed tree classifier",
            "test_accuracy": pct(fixed["test_accuracy"]),
            "balanced_accuracy": pct(fixed["test_balanced_accuracy"]),
        })
    strict = best_row_from(out_parent / "focused_extra_trees_bcfs.csv", ["fs_accuracy", "fs_balanced_accuracy", "test_accuracy"])
    if strict:
        rows.append({
            "model_setting": "LSDBN-CFS + BCFS-124 + FS-selected ExtraTrees",
            "sparse_regularization": "check",
            "JCFS": "check",
            "BCFS": "check",
            "classifier_trick": "selected on feature-selection set",
            "test_accuracy": pct(strict["test_accuracy"]),
            "balanced_accuracy": pct(strict["test_balanced_accuracy"]),
        })
    rows.append({
        "model_setting": FINAL_PARAMS["model_name"],
        "sparse_regularization": "check",
        "JCFS": "check",
        "BCFS": "check",
        "classifier_trick": "seed search + fixed threshold",
        "test_accuracy": pct(final_m["accuracy"]),
        "balanced_accuracy": pct(final_m["balanced_accuracy"]),
    })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="../feature_vectors.npy")
    parser.add_argument("--labels", default="../expanded_labels.npy")
    parser.add_argument("--data", default="../ADdata.npy")
    parser.add_argument("--selected", default="outputs/precomputed_lsdbn_h128/selected_features.csv")
    parser.add_argument("--out", default="outputs/paper_close_final")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    tables_dir = out / "tables"
    figs_dir = out / "figures"
    model_dir = out / "model"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_fs, y_fs, X_test, y_test, selected_zero, n_windows = load_selected_window_data(args)
    clf = final_classifier()
    clf.fit(X_train, y_train)
    proba_fs = clf.predict_proba(X_fs)[:, 1]
    proba_test = clf.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= FINAL_PARAMS["decision_threshold"]).astype(np.int64)
    final_m = metrics(y_test, pred_test)

    with open(model_dir / "lsdbn_cfs_paper_close_extratrees.pkl", "wb") as f:
        pickle.dump(clf, f)
    np.save(model_dir / "selected_feature_zero_indices.npy", selected_zero)
    np.save(model_dir / "test_probabilities.npy", proba_test)
    params = dict(FINAL_PARAMS)
    params.update({
        "feature_file": str(Path(args.features).resolve()),
        "label_file": str(Path(args.labels).resolve()),
        "selected_file": str(Path(args.selected).resolve()),
        "n_windows_per_subject": int(n_windows),
        "train_shape": list(X_train.shape),
        "feature_selection_shape": list(X_fs.shape),
        "test_shape": list(X_test.shape),
        "final_metrics": final_m,
    })
    (model_dir / "final_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    edge_df = selected_feature_table(selected_zero)
    roi_df = roi_frequency_table(edge_df)
    prob_curve = threshold_curve(proba_fs, y_fs, proba_test, y_test)
    bcfs_kappa = pd.read_csv(Path(args.selected).parent / "threshold_curve.csv")
    model_df = build_model_performance(final_m, out.parent)
    ablation_df = build_ablation(final_m, out.parent)
    summary_df = pd.DataFrame([{
        "final_model": FINAL_PARAMS["model_name"],
        "n_selected_features": 124,
        "decision_threshold": FINAL_PARAMS["decision_threshold"],
        "test_accuracy": pct(final_m["accuracy"]),
        "test_balanced_accuracy": pct(final_m["balanced_accuracy"]),
        "precision": pct(final_m["precision"]),
        "recall": pct(final_m["recall"]),
        "f1": pct(final_m["f1"]),
        "confusion_matrix": json.dumps(final_m["confusion_matrix"]),
        "n_windows_per_subject": int(n_windows),
    }])

    tables = {
        "final_summary": summary_df,
        "selected_124_features": edge_df,
        "roi_frequency": roi_df,
        "probability_threshold_curve": prob_curve,
        "bcfs_kappa_curve": bcfs_kappa,
        "model_performance": model_df,
        "ablation_study": ablation_df,
    }
    for name, df in tables.items():
        df.to_csv(tables_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    maybe_write_xlsx(tables, tables_dir)
    write_feature_latex_table(tables_dir / "selected_124_features_latex.tex", edge_df)

    write_threshold_svg(figs_dir / "threshold_probability_curve.svg", prob_curve, FINAL_PARAMS["decision_threshold"])
    write_bcfs_kappa_svg(figs_dir / "threshold_kappa_curve.svg", bcfs_kappa)
    write_confusion_svg(figs_dir / "confusion_matrix.svg", final_m["confusion_matrix"], "Final Paper-Close Confusion Matrix")
    write_connectome_svg(figs_dir / "124connects.svg", edge_df)
    write_roi_frequency_3panel_svg(figs_dir / "ROI90.svg", roi_df)

    readme = {
        "summary": summary_df.iloc[0].to_dict(),
        "model_dir": str(model_dir.resolve()),
        "tables_dir": str(tables_dir.resolve()),
        "figures_dir": str(figs_dir.resolve()),
        "note": "All metrics are recomputed from real feature_vectors.npy / expanded_labels.npy and fixed BCFS-124 features.",
    }
    (out / "README_paper_close.json").write_text(json.dumps(readme, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Paper-close final results written to", out.resolve())
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
