from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lsdbn_cfs_full.data import load_adni_data, stratified_subject_split


def no_self_indices(n_roi: int = 90) -> np.ndarray:
    return np.asarray([i * n_roi + j for i in range(n_roi) for j in range(n_roi) if i != j], dtype=np.int64)


def load_bcfs_data(args):
    X_all = np.load(args.features, mmap_mode="r")
    y_all = np.load(args.labels, mmap_mode="r").astype(np.int64)
    adni = load_adni_data(args.data)
    n_windows = X_all.shape[0] // len(adni.labels)
    train_s, fs_s, test_s = stratified_subject_split(adni.labels, random_state=args.seed)
    selected = pd.read_csv(args.selected)["feature_id"].to_numpy(dtype=np.int64) - 1
    cols = no_self_indices()

    def rows_for(subjects):
        return np.concatenate([np.arange(s * n_windows, (s + 1) * n_windows) for s in subjects])

    tr, fs, te = rows_for(train_s), rows_for(fs_s), rows_for(test_s)
    X_train = np.asarray(X_all[tr][:, cols], dtype=np.float32)[:, selected]
    X_fs = np.asarray(X_all[fs][:, cols], dtype=np.float32)[:, selected]
    X_test = np.asarray(X_all[te][:, cols], dtype=np.float32)[:, selected]
    return X_train, np.asarray(y_all[tr]), X_fs, np.asarray(y_all[fs]), X_test, np.asarray(y_all[te]), selected, n_windows


def sample_weight(y, pos_weight: float) -> np.ndarray:
    w = np.ones_like(y, dtype=np.float64)
    w[y == 1] = pos_weight
    return w


def make_model(kind: str, seed: int, params: dict):
    if kind == "extra_trees":
        cw = params.get("class_weight", "balanced")
        return ExtraTreesClassifier(
            n_estimators=int(params.get("n_estimators", 700)),
            max_features=params.get("max_features", "sqrt"),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_depth=params.get("max_depth", None),
            class_weight=cw,
            random_state=seed,
            n_jobs=1,
        )
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 500)),
            max_features=params.get("max_features", "sqrt"),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_depth=params.get("max_depth", None),
            class_weight=params.get("class_weight", "balanced"),
            random_state=seed,
            n_jobs=1,
        )
    if kind == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", 2)),
            subsample=float(params.get("subsample", 0.8)),
            random_state=seed,
        )
    if kind == "rbf_svc":
        return make_pipeline(
            StandardScaler(),
            SVC(
                C=float(params.get("C", 0.03)),
                gamma=params.get("gamma", "scale"),
                class_weight=params.get("class_weight", "balanced"),
                probability=False,
            ),
        )
    if kind == "logistic_l1":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(params.get("C", 0.003)),
                penalty="l1",
                solver="liblinear",
                max_iter=5000,
                class_weight=params.get("class_weight", "balanced"),
                random_state=seed,
            ),
        )
    raise ValueError(kind)


def scores(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def best_threshold_from_fs(fs_score, y_fs, metric: str):
    thresholds = np.unique(np.quantile(fs_score, np.linspace(0.02, 0.98, 193)))
    thresholds = np.r_[thresholds, 0.0, 0.5]
    best_t, best_val = 0.0, -1.0
    for t in thresholds:
        pred = (fs_score >= t).astype(np.int64)
        if metric == "balanced":
            val = balanced_accuracy_score(y_fs, pred)
        else:
            val = accuracy_score(y_fs, pred)
        if val > best_val:
            best_t, best_val = float(t), float(val)
    return best_t, best_val


def evaluate(kind, params, model, X_fs, y_fs, X_test, y_test, strategy):
    pred_fs = model.predict(X_fs)
    pred_test = model.predict(X_test)
    out = []
    out.append({
        "kind": kind,
        "params": json.dumps(params, ensure_ascii=False),
        "train_strategy": strategy,
        "threshold_strategy": "default",
        "threshold": "",
        "fs_accuracy": accuracy_score(y_fs, pred_fs),
        "fs_balanced_accuracy": balanced_accuracy_score(y_fs, pred_fs),
        "test_accuracy": accuracy_score(y_test, pred_test),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, pred_test),
        "test_confusion_matrix": json.dumps(confusion_matrix(y_test, pred_test).tolist()),
    })
    fs_score = scores(model, X_fs)
    test_score = scores(model, X_test)
    for metric in ["accuracy", "balanced"]:
        t, _ = best_threshold_from_fs(fs_score, y_fs, metric)
        pred_fs_t = (fs_score >= t).astype(np.int64)
        pred_test_t = (test_score >= t).astype(np.int64)
        out.append({
            "kind": kind,
            "params": json.dumps(params, ensure_ascii=False),
            "train_strategy": strategy,
            "threshold_strategy": "fs_%s" % metric,
            "threshold": t,
            "fs_accuracy": accuracy_score(y_fs, pred_fs_t),
            "fs_balanced_accuracy": balanced_accuracy_score(y_fs, pred_fs_t),
            "test_accuracy": accuracy_score(y_test, pred_test_t),
            "test_balanced_accuracy": balanced_accuracy_score(y_test, pred_test_t),
            "test_confusion_matrix": json.dumps(confusion_matrix(y_test, pred_test_t).tolist()),
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="../feature_vectors.npy")
    parser.add_argument("--labels", default="../expanded_labels.npy")
    parser.add_argument("--data", default="../ADdata.npy")
    parser.add_argument("--selected", default="outputs/precomputed_lsdbn_h128/selected_features.csv")
    parser.add_argument("--out", default="outputs/tune_bcfs_124_advanced.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X_train, y_train, X_fs, y_fs, X_test, y_test, selected, _ = load_bcfs_data(args)
    print("Shapes", X_train.shape, X_fs.shape, X_test.shape)

    class_weights = [
        "balanced",
        {0: 1.0, 1: 0.45},
        {0: 1.0, 1: 0.60},
        {0: 1.0, 1: 0.80},
        {0: 1.0, 1: 1.00},
        {0: 1.4, 1: 1.00},
        {0: 1.8, 1: 1.00},
    ]
    specs = []
    for cw in class_weights:
        for leaf in [1, 3, 6]:
            specs.append(("extra_trees", {"class_weight": cw, "min_samples_leaf": leaf, "max_features": "sqrt", "n_estimators": 700}))
    for cw in class_weights[:5]:
        specs.append(("random_forest", {"class_weight": cw, "min_samples_leaf": 1, "max_features": "sqrt", "n_estimators": 500}))
    for cw in class_weights:
        for C in [0.003, 0.01, 0.03, 0.1]:
            specs.append(("rbf_svc", {"class_weight": cw, "C": C, "gamma": "scale"}))
    for cw in class_weights[:5]:
        for C in [0.001, 0.003, 0.01, 0.03]:
            specs.append(("logistic_l1", {"class_weight": cw, "C": C}))
    for pw in [0.45, 0.6, 0.8, 1.0]:
        for lr in [0.03, 0.05]:
            specs.append(("gradient_boosting", {"pos_weight": pw, "n_estimators": 220, "learning_rate": lr, "max_depth": 2, "subsample": 0.85}))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "kind", "params", "train_strategy", "threshold_strategy", "threshold",
        "fs_accuracy", "fs_balanced_accuracy", "test_accuracy", "test_balanced_accuracy",
        "test_confusion_matrix",
    ]
    rows = []
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, (kind, params) in enumerate(specs, start=1):
            for strategy in ["train_only", "train_plus_feature_selection"]:
                if strategy == "train_only":
                    xt, yt = X_train, y_train
                else:
                    xt, yt = np.vstack([X_train, X_fs]), np.concatenate([y_train, y_fs])
                model = make_model(kind, args.seed, params)
                fit_kwargs = {}
                if kind == "gradient_boosting":
                    fit_kwargs["sample_weight"] = sample_weight(yt, float(params.get("pos_weight", 1.0)))
                model.fit(xt, yt, **fit_kwargs)
                eval_rows = evaluate(kind, params, model, X_fs, y_fs, X_test, y_test, strategy)
                for row in eval_rows:
                    rows.append(row)
                    writer.writerow(row)
                f.flush()
            if idx % 5 == 0:
                best = sorted(rows, key=lambda r: (r["fs_accuracy"], r["test_accuracy"]), reverse=True)[0]
                print("progress %d/%d best_by_fs=%s" % (idx, len(specs), json.dumps(best, ensure_ascii=False)))

    df = pd.DataFrame(rows)
    print("Best by fs:")
    print(df.sort_values(["fs_accuracy", "fs_balanced_accuracy", "test_accuracy"], ascending=False).head(15).to_string(index=False))
    print("Best by test diagnostic:")
    print(df.sort_values(["test_accuracy", "test_balanced_accuracy"], ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()

