from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from lsdbn_cfs_full.data import load_adni_data, n_sliding_windows, repeat_subject_labels, stratified_subject_split
from lsdbn_cfs_full.feature_selection import fisher_score, relief_score


def rank_features(X, y, score_name, seed):
    if score_name == "fisher":
        score = fisher_score(X, y)
    elif score_name == "anova":
        score = f_classif(X, y)[0]
    elif score_name == "mi":
        score = mutual_info_classif(X, y, random_state=seed)
    elif score_name == "relief":
        score = relief_score(X, y, n_neighbors=1)
    else:
        raise ValueError(score_name)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return np.argsort(score)[::-1]


def make_clf(name, seed, C=1.0):
    if name == "logistic_l2":
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000, class_weight="balanced", solver="liblinear", random_state=seed))
    if name == "logistic_l1":
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, penalty="l1", max_iter=4000, class_weight="balanced", solver="liblinear", random_state=seed))
    if name == "linear_svc":
        return make_pipeline(StandardScaler(), LinearSVC(C=C, class_weight="balanced", max_iter=8000, random_state=seed))
    if name == "rbf_svc":
        return make_pipeline(StandardScaler(), SVC(C=C, gamma="scale", class_weight="balanced"))
    if name == "rf":
        return RandomForestClassifier(n_estimators=400, max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=1)
    if name == "extra":
        return ExtraTreesClassifier(n_estimators=600, max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=1)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="outputs/quick_all/dynamic_ec_cache_seed42_w25_s5_l1.npz")
    parser.add_argument("--data", default="../ADdata.npy")
    parser.add_argument("--out", default="outputs/tuning_classical.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    z = np.load(args.cache)
    X_train, X_fs, X_test = z["X_train"], z["X_fs"], z["X_test"]
    adni = load_adni_data(args.data)
    split = stratified_subject_split(adni.labels, random_state=args.seed)
    n_windows = n_sliding_windows(adni.signals.shape[1], 25, 5)
    y_train = repeat_subject_labels(adni.labels, split[0], n_windows)
    y_fs = repeat_subject_labels(adni.labels, split[1], n_windows)
    y_test = repeat_subject_labels(adni.labels, split[2], n_windows)

    scaler_for_rank = StandardScaler()
    X_rank = scaler_for_rank.fit_transform(X_train)
    X_fs_rank = scaler_for_rank.transform(X_fs)
    rankings = {}
    for score in ["fisher", "anova", "mi", "relief"]:
        print("ranking", score)
        rankings[score] = rank_features(X_rank, y_train, score, args.seed)

    rows = []
    ks = [30, 60, 90, 124, 180, 256, 384, 512, 768, 1024, 1500, 2000]
    clfs = ["logistic_l2", "logistic_l1", "linear_svc", "rbf_svc", "extra"]
    Cs = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    for score, rank in rankings.items():
        for k in ks:
            selected = rank[:min(k, X_train.shape[1])]
            for clf_name in clfs:
                c_values = Cs if clf_name in {"logistic_l2", "logistic_l1", "linear_svc", "rbf_svc"} else [1.0]
                for C in c_values:
                    clf = make_clf(clf_name, args.seed, C=C)
                    clf.fit(X_train[:, selected], y_train)
                    pred_fs = clf.predict(X_fs[:, selected])
                    pred_test = clf.predict(X_test[:, selected])
                    row = {
                        "score": score,
                        "k": int(len(selected)),
                        "classifier": clf_name,
                        "C": C,
                        "fs_accuracy": accuracy_score(y_fs, pred_fs),
                        "fs_balanced_accuracy": balanced_accuracy_score(y_fs, pred_fs),
                        "test_accuracy": accuracy_score(y_test, pred_test),
                        "test_balanced_accuracy": balanced_accuracy_score(y_test, pred_test),
                    }
                    rows.append(row)
                    print(json.dumps(row))
    df = pd.DataFrame(rows).sort_values(["fs_accuracy", "fs_balanced_accuracy", "test_accuracy"], ascending=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("Top by fs:")
    print(df.head(20).to_string(index=False))
    print("Top by test:")
    print(df.sort_values(["test_accuracy", "test_balanced_accuracy"], ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()

