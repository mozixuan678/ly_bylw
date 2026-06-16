from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_classif
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
    parser.add_argument("--out", default="outputs/ensemble_124_eval.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X_all = np.load(args.features, mmap_mode="r")
    y_all = np.load(args.labels, mmap_mode="r").astype(np.int64)
    adni = load_adni_data(args.data)
    n_windows = X_all.shape[0] // len(adni.labels)
    train_s, fs_s, test_s = stratified_subject_split(adni.labels, random_state=args.seed)
    cols = no_self_indices()

    def rows_for(subjects):
        return np.concatenate([np.arange(s * n_windows, (s + 1) * n_windows) for s in subjects])

    tr, fs, te = rows_for(train_s), rows_for(fs_s), rows_for(test_s)
    X_train = np.asarray(X_all[tr][:, cols], dtype=np.float32)
    X_fs = np.asarray(X_all[fs][:, cols], dtype=np.float32)
    X_test = np.asarray(X_all[te][:, cols], dtype=np.float32)
    y_train, y_fs, y_test = np.asarray(y_all[tr]), np.asarray(y_all[fs]), np.asarray(y_all[te])

    rank = np.argsort(np.nan_to_num(f_classif(StandardScaler().fit_transform(X_train), y_train)[0]))[::-1]
    sel = rank[:124]
    models = [
        ("rbf_003_train", make_pipeline(StandardScaler(), SVC(C=0.03, gamma="scale", class_weight="balanced")), X_train, y_train),
        ("rbf_001_train", make_pipeline(StandardScaler(), SVC(C=0.01, gamma="scale", class_weight="balanced")), X_train, y_train),
        ("extra_trainfs", ExtraTreesClassifier(n_estimators=1000, max_features="sqrt", class_weight="balanced", random_state=args.seed, n_jobs=1), np.vstack([X_train, X_fs]), np.concatenate([y_train, y_fs])),
        ("log_l1_train", make_pipeline(StandardScaler(), LogisticRegression(C=0.003, penalty="l1", solver="liblinear", max_iter=4000, class_weight="balanced", random_state=args.seed)), X_train, y_train),
    ]
    fs_preds, test_preds = [], []
    rows = []
    for name, model, xt, yt in models:
        model.fit(xt[:, sel], yt)
        pfs = model.predict(X_fs[:, sel])
        pte = model.predict(X_test[:, sel])
        fs_preds.append(pfs)
        test_preds.append(pte)
        rows.append({
            "method": name,
            "fs_accuracy": accuracy_score(y_fs, pfs),
            "fs_balanced_accuracy": balanced_accuracy_score(y_fs, pfs),
            "test_accuracy": accuracy_score(y_test, pte),
            "test_balanced_accuracy": balanced_accuracy_score(y_test, pte),
            "confusion_matrix": json.dumps(confusion_matrix(y_test, pte).tolist()),
        })

    fs_mat = np.vstack(fs_preds)
    te_mat = np.vstack(test_preds)
    for threshold in [1, 2, 3, 4]:
        pfs = (fs_mat.sum(axis=0) >= threshold).astype(np.int64)
        pte = (te_mat.sum(axis=0) >= threshold).astype(np.int64)
        rows.append({
            "method": "hard_vote_at_least_%d" % threshold,
            "fs_accuracy": accuracy_score(y_fs, pfs),
            "fs_balanced_accuracy": balanced_accuracy_score(y_fs, pfs),
            "test_accuracy": accuracy_score(y_test, pte),
            "test_balanced_accuracy": balanced_accuracy_score(y_test, pte),
            "confusion_matrix": json.dumps(confusion_matrix(y_test, pte).tolist()),
        })

    df = pd.DataFrame(rows).sort_values(["fs_accuracy", "fs_balanced_accuracy", "test_accuracy"], ascending=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

