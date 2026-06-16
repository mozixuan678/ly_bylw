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


def no_self_indices(n_roi=90):
    return np.asarray([i * n_roi + j for i in range(n_roi) for j in range(n_roi) if i != j], dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="../feature_vectors.npy")
    parser.add_argument("--labels", default="../expanded_labels.npy")
    parser.add_argument("--data", default="../ADdata.npy")
    parser.add_argument("--selected", default="outputs/precomputed_lsdbn_h128/selected_features.csv")
    parser.add_argument("--out", default="outputs/bcfs_selected_classifier_eval.csv")
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
    y_train, y_fs, y_test = np.asarray(y_all[tr]), np.asarray(y_all[fs]), np.asarray(y_all[te])

    models = [
        ("logistic_l1_C0.003", make_pipeline(StandardScaler(), LogisticRegression(C=0.003, penalty="l1", solver="liblinear", max_iter=4000, class_weight="balanced", random_state=args.seed))),
        ("rbf_svc_C0.01", make_pipeline(StandardScaler(), SVC(C=0.01, gamma="scale", class_weight="balanced"))),
        ("rbf_svc_C0.03", make_pipeline(StandardScaler(), SVC(C=0.03, gamma="scale", class_weight="balanced"))),
        ("extra_trees", ExtraTreesClassifier(n_estimators=1000, max_features="sqrt", class_weight="balanced", random_state=args.seed, n_jobs=1)),
        ("random_forest", RandomForestClassifier(n_estimators=600, max_features="sqrt", class_weight="balanced", random_state=args.seed, n_jobs=1)),
    ]
    rows = []
    for name, model in models:
        for strategy, xt, yt in [
            ("train_only", X_train, y_train),
            ("train_plus_feature_selection", np.vstack([X_train, X_fs]), np.concatenate([y_train, y_fs])),
        ]:
            model.fit(xt, yt)
            pred = model.predict(X_test)
            rows.append({
                "selected_source": args.selected,
                "model": name,
                "train_strategy": strategy,
                "n_features": int(len(selected)),
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

