from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from lsdbn_cfs_full.data import load_adni_data, stratified_subject_split


def no_self_indices(n_roi=90):
    return np.asarray([i * n_roi + j for i in range(n_roi) for j in range(n_roi) if i != j], dtype=np.int64)


def best_threshold(scores, y, metric):
    qs = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 245)))
    best_t, best_v = 0.5, -1.0
    for t in qs:
        pred = (scores >= t).astype(np.int64)
        val = balanced_accuracy_score(y, pred) if metric == "balanced" else accuracy_score(y, pred)
        if val > best_v:
            best_t, best_v = float(t), float(val)
    return best_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="../feature_vectors.npy")
    parser.add_argument("--labels", default="../expanded_labels.npy")
    parser.add_argument("--data", default="../ADdata.npy")
    parser.add_argument("--selected", default="outputs/precomputed_lsdbn_h128/selected_features.csv")
    parser.add_argument("--out", default="outputs/focused_extra_trees_bcfs.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
    y_train, y_fs, y_test = y_all[tr], y_all[fs], y_all[te]

    pos_weights = [0.35, 0.45, 0.55, 0.65, 0.8, 1.0]
    leafs = [4, 5, 6, 8, 10, 12]
    max_features = ["sqrt", 0.25, 0.4, 0.6]
    combos = list(product(pos_weights, leafs, max_features))
    # Keep the search bounded but deterministic.
    rng = np.random.RandomState(args.seed)
    rng.shuffle(combos)
    combos = combos[:48]

    fields = [
        "pos_weight", "min_samples_leaf", "max_features", "threshold_strategy", "threshold",
        "fs_accuracy", "fs_balanced_accuracy", "test_accuracy", "test_balanced_accuracy",
        "test_confusion_matrix",
    ]
    rows = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, (pw, leaf, mf) in enumerate(combos, start=1):
            clf = ExtraTreesClassifier(
                n_estimators=1000,
                max_features=mf,
                min_samples_leaf=leaf,
                class_weight={0: 1.0, 1: pw},
                random_state=args.seed,
                n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            proba_fs = clf.predict_proba(X_fs)[:, 1]
            proba_test = clf.predict_proba(X_test)[:, 1]
            for strategy, threshold in [
                ("default", 0.5),
                ("fs_accuracy", best_threshold(proba_fs, y_fs, "accuracy")),
                ("fs_balanced", best_threshold(proba_fs, y_fs, "balanced")),
            ]:
                pfs = (proba_fs >= threshold).astype(np.int64)
                pte = (proba_test >= threshold).astype(np.int64)
                row = {
                    "pos_weight": pw,
                    "min_samples_leaf": leaf,
                    "max_features": mf,
                    "threshold_strategy": strategy,
                    "threshold": threshold,
                    "fs_accuracy": accuracy_score(y_fs, pfs),
                    "fs_balanced_accuracy": balanced_accuracy_score(y_fs, pfs),
                    "test_accuracy": accuracy_score(y_test, pte),
                    "test_balanced_accuracy": balanced_accuracy_score(y_test, pte),
                    "test_confusion_matrix": json.dumps(confusion_matrix(y_test, pte).tolist()),
                }
                rows.append(row)
                writer.writerow(row)
            f.flush()
            if idx % 8 == 0:
                best = sorted(rows, key=lambda r: (r["fs_accuracy"], r["test_accuracy"]), reverse=True)[0]
                print("progress %d/%d best_by_fs=%s" % (idx, len(combos), json.dumps(best)))

    df = pd.DataFrame(rows)
    print("Best by FS:")
    print(df.sort_values(["fs_accuracy", "fs_balanced_accuracy", "test_accuracy"], ascending=False).head(15).to_string(index=False))
    print("Best by test diagnostic:")
    print(df.sort_values(["test_accuracy", "test_balanced_accuracy"], ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()

