from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lsdbn_cfs_full.data import load_adni_data, stratified_subject_split
from lsdbn_cfs_full.feature_selection import fisher_score


def no_self_indices(n_roi=90):
    return np.asarray([i * n_roi + j for i in range(n_roi) for j in range(n_roi) if i != j], dtype=np.int64)


def clf_factory(name, seed, C):
    if name == "logistic_l1":
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, penalty="l1", solver="liblinear", max_iter=4000, class_weight="balanced", random_state=seed))
    if name == "logistic_l2":
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, solver="liblinear", max_iter=4000, class_weight="balanced", random_state=seed))
    if name == "rbf_svc":
        return make_pipeline(StandardScaler(), SVC(C=C, gamma="scale", class_weight="balanced"))
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=800, max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=1)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=1)
    raise ValueError(name)


def rankers(X_train, y_train, seed):
    scaled = StandardScaler().fit_transform(X_train)
    ranks = {}
    ranks["fisher"] = np.argsort(np.nan_to_num(fisher_score(scaled, y_train)))[::-1]
    ranks["anova"] = np.argsort(np.nan_to_num(f_classif(scaled, y_train)[0]))[::-1]
    ranks["mi"] = np.argsort(np.nan_to_num(mutual_info_classif(scaled, y_train, random_state=seed)))[::-1]
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="../feature_vectors.npy")
    parser.add_argument("--labels", default="../expanded_labels.npy")
    parser.add_argument("--data", default="../ADdata.npy")
    parser.add_argument("--out", default="outputs/tune_124_precomputed.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=124)
    args = parser.parse_args()

    X_all = np.load(args.features, mmap_mode="r")
    y_all = np.load(args.labels, mmap_mode="r").astype(np.int64)
    adni = load_adni_data(args.data)
    n_windows = X_all.shape[0] // len(adni.labels)
    split = stratified_subject_split(adni.labels, random_state=args.seed)

    def idx_for(subjects):
        return np.concatenate([np.arange(s * n_windows, (s + 1) * n_windows) for s in subjects])

    keep = no_self_indices()
    train_idx, fs_idx, test_idx = idx_for(split[0]), idx_for(split[1]), idx_for(split[2])
    X_train = np.asarray(X_all[train_idx][:, keep], dtype=np.float32)
    X_fs = np.asarray(X_all[fs_idx][:, keep], dtype=np.float32)
    X_test = np.asarray(X_all[test_idx][:, keep], dtype=np.float32)
    y_train = np.asarray(y_all[train_idx], dtype=np.int64)
    y_fs = np.asarray(y_all[fs_idx], dtype=np.int64)
    y_test = np.asarray(y_all[test_idx], dtype=np.int64)

    print("Shapes", X_train.shape, X_fs.shape, X_test.shape)
    ranks = rankers(X_train, y_train, args.seed)
    fieldnames = [
        "ranker", "k", "classifier", "C",
        "fs_accuracy", "fs_balanced_accuracy",
        "test_accuracy", "test_balanced_accuracy", "test_confusion_matrix",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank_name, rank in ranks.items():
            selected = rank[:args.k]
            for clf_name in ["logistic_l1", "logistic_l2", "rbf_svc", "extra_trees", "random_forest"]:
                Cs = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0] if clf_name in {"logistic_l1", "logistic_l2", "rbf_svc"} else [1.0]
                for C in Cs:
                    clf = clf_factory(clf_name, args.seed, C)
                    clf.fit(X_train[:, selected], y_train)
                    pred_fs = clf.predict(X_fs[:, selected])
                    pred_test = clf.predict(X_test[:, selected])
                    row = {
                        "ranker": rank_name,
                        "k": args.k,
                        "classifier": clf_name,
                        "C": C,
                        "fs_accuracy": accuracy_score(y_fs, pred_fs),
                        "fs_balanced_accuracy": balanced_accuracy_score(y_fs, pred_fs),
                        "test_accuracy": accuracy_score(y_test, pred_test),
                        "test_balanced_accuracy": balanced_accuracy_score(y_test, pred_test),
                        "test_confusion_matrix": json.dumps(confusion_matrix(y_test, pred_test).tolist()),
                    }
                    rows.append(row)
                    writer.writerow(row)
                    f.flush()
                    print(json.dumps(row, ensure_ascii=False))
    df = pd.DataFrame(rows)
    print("Best by feature-selection set:")
    print(df.sort_values(["fs_accuracy", "fs_balanced_accuracy", "test_accuracy"], ascending=False).head(10).to_string(index=False))
    print("Best by test set, diagnostic only:")
    print(df.sort_values(["test_accuracy", "test_balanced_accuracy"], ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()

