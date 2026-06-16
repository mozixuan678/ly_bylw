from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def no_self_indices(n_roi=90):
    return np.asarray([i * n_roi + j for i in range(n_roi) for j in range(n_roi) if i != j], dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="../feature_vectors.npy")
    parser.add_argument("--labels", default="../expanded_labels.npy")
    parser.add_argument("--out", default="outputs/window_leakage_probe.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X_all = np.load(args.features, mmap_mode="r")
    y_all = np.load(args.labels, mmap_mode="r").astype(np.int64)
    cols = no_self_indices()
    rng = np.random.RandomState(args.seed)
    idx = np.arange(X_all.shape[0])
    rng.shuffle(idx)
    n_train = int(round(0.60 * len(idx)))
    n_fs = int(round(0.30 * len(idx)))
    tr, fs, te = idx[:n_train], idx[n_train:n_train + n_fs], idx[n_train + n_fs:]
    X_train = np.asarray(X_all[tr][:, cols], dtype=np.float32)
    X_fs = np.asarray(X_all[fs][:, cols], dtype=np.float32)
    X_test = np.asarray(X_all[te][:, cols], dtype=np.float32)
    y_train, y_fs, y_test = y_all[tr], y_all[fs], y_all[te]

    rank = np.argsort(np.nan_to_num(f_classif(StandardScaler().fit_transform(X_train), y_train)[0]))[::-1]
    rows = []
    for k in [124, 256, 512]:
        sel = rank[:k]
        models = [
            ("extra_trees", ExtraTreesClassifier(n_estimators=700, max_features="sqrt", min_samples_leaf=3, class_weight={0: 1.0, 1: 0.6}, random_state=args.seed, n_jobs=1)),
            ("random_forest", RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=1, class_weight={0: 1.0, 1: 0.6}, random_state=args.seed, n_jobs=1)),
            ("rbf_svc", make_pipeline(StandardScaler(), SVC(C=0.03, gamma="scale", class_weight="balanced"))),
        ]
        for name, model in models:
            model.fit(np.vstack([X_train[:, sel], X_fs[:, sel]]), np.concatenate([y_train, y_fs]))
            pred = model.predict(X_test[:, sel])
            rows.append({
                "warning": "window-level random split; subject leakage likely; diagnostic only",
                "k": k,
                "model": name,
                "test_accuracy": accuracy_score(y_test, pred),
                "test_balanced_accuracy": balanced_accuracy_score(y_test, pred),
                "confusion_matrix": json.dumps(confusion_matrix(y_test, pred).tolist()),
            })
    df = pd.DataFrame(rows).sort_values(["test_accuracy", "test_balanced_accuracy"], ascending=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

