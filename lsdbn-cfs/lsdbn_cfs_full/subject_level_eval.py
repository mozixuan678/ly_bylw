from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lsdbn_cfs_full.aal90 import AAL90_NAMES, ROI_NETWORK
from lsdbn_cfs_full.data import load_adni_data, stratified_subject_split


def no_self_indices(n_roi=90):
    return np.asarray([i * n_roi + j for i in range(n_roi) for j in range(n_roi) if i != j], dtype=np.int64)


def write_svg_bar(path, labels, values, title):
    width, height = 820, 360
    ml, mt, mb = 70, 50, 70
    pw, ph = width - ml - 30, height - mt - mb
    mx = max(values) if values else 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#333"/>',
        f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#333"/>',
    ]
    for i, (lab, val) in enumerate(zip(labels, values)):
        bw = pw / len(values) * 0.62
        x = ml + (i + 0.18) * pw / len(values)
        h = ph * val / mx
        y = mt + ph - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="#2563eb"/>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{val:.3f}</text>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{mt+ph+24}" text-anchor="middle" font-family="Arial" font-size="12">{lab}</text>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def majority_vote(pred_window, true_window, n_windows):
    pred_subject, true_subject = [], []
    for start in range(0, len(pred_window), n_windows):
        p = pred_window[start:start + n_windows]
        y = true_window[start:start + n_windows]
        pred_subject.append(int(np.mean(p) >= 0.5))
        true_subject.append(int(round(float(np.mean(y)))))
    return np.asarray(true_subject), np.asarray(pred_subject)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="../feature_vectors.npy")
    parser.add_argument("--labels", default="../expanded_labels.npy")
    parser.add_argument("--data", default="../ADdata.npy")
    parser.add_argument("--out", default="outputs/subject_level_eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

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

    # Best window-level configuration found in final_124_eval.
    X_trfs = np.vstack([X_train, X_fs])
    y_trfs = np.concatenate([y_train, y_fs])
    rank = np.argsort(np.nan_to_num(f_classif(StandardScaler().fit_transform(X_train), y_train)[0]))[::-1]
    sel = rank[:124]
    clf = ExtraTreesClassifier(n_estimators=1000, max_features="sqrt", class_weight="balanced", random_state=args.seed, n_jobs=1)
    clf.fit(X_trfs[:, sel], y_trfs)
    pred_test = clf.predict(X_test[:, sel])
    y_sub, p_sub = majority_vote(pred_test, y_test, n_windows)

    # Subject mean-feature experiment.
    X_sub = np.asarray(X_all[:, cols], dtype=np.float32).reshape(len(adni.labels), n_windows, len(cols)).mean(axis=1)
    y_sub_all = adni.labels
    X_train_s, X_fs_s, X_test_s = X_sub[train_s], X_sub[fs_s], X_sub[test_s]
    y_train_s, y_fs_s, y_test_s = y_sub_all[train_s], y_sub_all[fs_s], y_sub_all[test_s]
    rank_s = np.argsort(np.nan_to_num(f_classif(StandardScaler().fit_transform(X_train_s), y_train_s)[0]))[::-1]
    sel_s = rank_s[:124]
    candidates = [
        ("subject_mean_rbf_C0.03", make_pipeline(StandardScaler(), SVC(C=0.03, gamma="scale", class_weight="balanced"))),
        ("subject_mean_rbf_C0.1", make_pipeline(StandardScaler(), SVC(C=0.1, gamma="scale", class_weight="balanced"))),
        ("subject_mean_extra_trees", ExtraTreesClassifier(n_estimators=1000, max_features="sqrt", class_weight="balanced", random_state=args.seed, n_jobs=1)),
    ]
    rows = [{
        "experiment": "window_model_subject_majority_vote",
        "n_features": 124,
        "test_accuracy": accuracy_score(y_test, pred_test),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, pred_test),
        "subject_accuracy": accuracy_score(y_sub, p_sub),
        "subject_balanced_accuracy": balanced_accuracy_score(y_sub, p_sub),
        "confusion_matrix": json.dumps(confusion_matrix(y_sub, p_sub).tolist()),
    }]
    for name, model in candidates:
        model.fit(np.vstack([X_train_s[:, sel_s], X_fs_s[:, sel_s]]), np.concatenate([y_train_s, y_fs_s]))
        pred = model.predict(X_test_s[:, sel_s])
        rows.append({
            "experiment": name,
            "n_features": 124,
            "test_accuracy": np.nan,
            "test_balanced_accuracy": np.nan,
            "subject_accuracy": accuracy_score(y_test_s, pred),
            "subject_balanced_accuracy": balanced_accuracy_score(y_test_s, pred),
            "confusion_matrix": json.dumps(confusion_matrix(y_test_s, pred).tolist()),
        })

    df = pd.DataFrame(rows).sort_values(["subject_accuracy", "subject_balanced_accuracy"], ascending=False)
    df.to_csv(out / "subject_level_results.csv", index=False, encoding="utf-8-sig")
    write_svg_bar(out / "subject_level_accuracy.svg", df["experiment"].tolist(), df["subject_accuracy"].astype(float).tolist(), "Subject-level Accuracy")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

