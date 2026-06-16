from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lsdbn_cfs_full.data import load_adni_data, stratified_subject_split
from lsdbn_cfs_full.feature_selection import fisher_score


def no_self_indices(n_roi=90):
    keep = []
    for i in range(n_roi):
        for j in range(n_roi):
            if i != j:
                keep.append(i * n_roi + j)
    return np.asarray(keep, dtype=np.int64)


def make_classifier(name, seed, C=1.0):
    if name == "logistic_l2":
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000, solver="liblinear", class_weight="balanced", random_state=seed))
    if name == "logistic_l1":
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, penalty="l1", max_iter=4000, solver="liblinear", class_weight="balanced", random_state=seed))
    if name == "rbf_svc":
        return make_pipeline(StandardScaler(), SVC(C=C, gamma="scale", class_weight="balanced"))
    if name == "extra":
        return ExtraTreesClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=1)
    if name == "rf":
        return RandomForestClassifier(n_estimators=400, max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=1)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="../feature_vectors.npy")
    parser.add_argument("--labels", default="../expanded_labels.npy")
    parser.add_argument("--data", default="../ADdata.npy")
    parser.add_argument("--out", default="outputs/probe_feature_vectors.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-window-leakage", action="store_true")
    args = parser.parse_args()

    X_all = np.load(args.features, mmap_mode="r")
    y_all = np.load(args.labels, mmap_mode="r").astype(np.int64)
    n_subjects = 132
    n_windows = X_all.shape[0] // n_subjects
    print("feature_vectors shape", X_all.shape, "windows/subject", n_windows)
    keep_no_self = no_self_indices()

    adni = load_adni_data(args.data)
    train_s, fs_s, test_s = stratified_subject_split(adni.labels, random_state=args.seed)
    if args.allow_window_leakage:
        rng = np.random.RandomState(args.seed)
        idx = np.arange(X_all.shape[0])
        rng.shuffle(idx)
        n_train = int(round(0.60 * len(idx)))
        n_fs = int(round(0.30 * len(idx)))
        train_idx = idx[:n_train]
        fs_idx = idx[n_train:n_train + n_fs]
        test_idx = idx[n_train + n_fs:]
    else:
        train_idx = np.concatenate([np.arange(s * n_windows, (s + 1) * n_windows) for s in train_s])
        fs_idx = np.concatenate([np.arange(s * n_windows, (s + 1) * n_windows) for s in fs_s])
        test_idx = np.concatenate([np.arange(s * n_windows, (s + 1) * n_windows) for s in test_s])

    X_train = np.asarray(X_all[train_idx][:, keep_no_self], dtype=np.float32)
    X_fs = np.asarray(X_all[fs_idx][:, keep_no_self], dtype=np.float32)
    X_test = np.asarray(X_all[test_idx][:, keep_no_self], dtype=np.float32)
    y_train = np.asarray(y_all[train_idx], dtype=np.int64)
    y_fs = np.asarray(y_all[fs_idx], dtype=np.int64)
    y_test = np.asarray(y_all[test_idx], dtype=np.int64)

    scaler_rank = StandardScaler()
    Xr = scaler_rank.fit_transform(X_train)
    rank = np.argsort(fisher_score(Xr, y_train))[::-1]
    rows = []
    for k in [30, 60, 90, 124, 180, 256, 384, 512, 768, 1024]:
        sel = rank[:k]
        for clf_name in ["logistic_l2", "logistic_l1", "rbf_svc", "extra"]:
            Cs = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0] if clf_name != "extra" else [1.0]
            for C in Cs:
                clf = make_classifier(clf_name, args.seed, C)
                clf.fit(X_train[:, sel], y_train)
                pred_fs = clf.predict(X_fs[:, sel])
                pred_test = clf.predict(X_test[:, sel])
                row = {
                    "subject_level_split": not args.allow_window_leakage,
                    "k": int(k),
                    "classifier": clf_name,
                    "C": C,
                    "fs_accuracy": accuracy_score(y_fs, pred_fs),
                    "fs_balanced_accuracy": balanced_accuracy_score(y_fs, pred_fs),
                    "test_accuracy": accuracy_score(y_test, pred_test),
                    "test_balanced_accuracy": balanced_accuracy_score(y_test, pred_test),
                    "test_confusion_matrix": json.dumps(confusion_matrix(y_test, pred_test).tolist()),
                }
                rows.append(row)
                print(json.dumps(row))
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["fs_accuracy", "test_accuracy"], ascending=False).to_csv(args.out, index=False, encoding="utf-8-sig")
    print("Top by fs")
    print(df.sort_values(["fs_accuracy", "fs_balanced_accuracy"], ascending=False).head(10).to_string(index=False))
    print("Top by test")
    print(df.sort_values(["test_accuracy", "test_balanced_accuracy"], ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()

