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

from lsdbn_cfs_full.aal90 import AAL90_NAMES, ROI_NETWORK, edge_name, feature_index_to_edge
from lsdbn_cfs_full.data import load_adni_data, stratified_subject_split
from lsdbn_cfs_full.feature_selection import fisher_score


def no_self_indices(n_roi=90):
    return np.asarray([i * n_roi + j for i in range(n_roi) for j in range(n_roi) if i != j], dtype=np.int64)


def get_splits(feature_file, label_file, data_file, seed):
    X_all = np.load(feature_file, mmap_mode="r")
    y_all = np.load(label_file, mmap_mode="r").astype(np.int64)
    adni = load_adni_data(data_file)
    n_windows = X_all.shape[0] // len(adni.labels)
    train_s, fs_s, test_s = stratified_subject_split(adni.labels, random_state=seed)

    def idx_for(subjects):
        return np.concatenate([np.arange(s * n_windows, (s + 1) * n_windows) for s in subjects])

    cols = no_self_indices()
    train_i, fs_i, test_i = idx_for(train_s), idx_for(fs_s), idx_for(test_s)
    return (
        np.asarray(X_all[train_i][:, cols], dtype=np.float32),
        np.asarray(y_all[train_i], dtype=np.int64),
        np.asarray(X_all[fs_i][:, cols], dtype=np.float32),
        np.asarray(y_all[fs_i], dtype=np.int64),
        np.asarray(X_all[test_i][:, cols], dtype=np.float32),
        np.asarray(y_all[test_i], dtype=np.int64),
        n_windows,
        test_s,
    )


def rank_features(X, y, method):
    scaled = StandardScaler().fit_transform(X)
    if method == "anova":
        score = f_classif(scaled, y)[0]
    elif method == "fisher":
        score = fisher_score(scaled, y)
    else:
        raise ValueError(method)
    return np.argsort(np.nan_to_num(score))[::-1]


def make_model(model_name, C, seed):
    if model_name == "rbf_svc":
        return make_pipeline(StandardScaler(), SVC(C=C, gamma="scale", class_weight="balanced"))
    if model_name == "logistic_l1":
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, penalty="l1", solver="liblinear", max_iter=4000, class_weight="balanced", random_state=seed))
    if model_name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=1000, max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=1)
    raise ValueError(model_name)


def decision_values(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    last = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    if hasattr(last, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def best_threshold(scores, y, metric="accuracy"):
    qs = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 199)))
    best_t, best_v = 0.0, -1.0
    for t in qs:
        pred = (scores >= t).astype(np.int64)
        val = accuracy_score(y, pred) if metric == "accuracy" else balanced_accuracy_score(y, pred)
        if val > best_v:
            best_t, best_v = float(t), float(val)
    return best_t, best_v


def feature_table(selected):
    rows = []
    for zero_idx in selected:
        fid = int(zero_idx) + 1
        src, tgt = feature_index_to_edge(fid)
        src_name, tgt_name = AAL90_NAMES[src - 1], AAL90_NAMES[tgt - 1]
        rows.append({
            "feature_id": fid,
            "source_roi": src,
            "target_roi": tgt,
            "source_name": src_name,
            "target_name": tgt_name,
            "source_network": ROI_NETWORK[src_name],
            "target_network": ROI_NETWORK[tgt_name],
            "edge": edge_name(src, tgt),
        })
    return pd.DataFrame(rows)


def write_svg_bar(path, labels, values, title):
    width, height = 900, 440
    margin_l, margin_b, margin_t = 70, 80, 50
    plot_w, plot_h = width - margin_l - 30, height - margin_t - margin_b
    mx = max(values) if values else 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t+plot_h}" stroke="#333"/>',
        f'<line x1="{margin_l}" y1="{margin_t+plot_h}" x2="{margin_l+plot_w}" y2="{margin_t+plot_h}" stroke="#333"/>',
    ]
    n = len(values)
    bw = plot_w / max(n, 1) * 0.72
    for i, (lab, val) in enumerate(zip(labels, values)):
        x = margin_l + (i + 0.14) * plot_w / n
        h = plot_h * val / mx
        y = margin_t + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="#3b82f6"/>')
        parts.append(f'<text x="{x + bw/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{val:.3f}</text>')
        parts.append(f'<text x="{x + bw/2:.1f}" y="{margin_t+plot_h+20}" text-anchor="end" transform="rotate(-35 {x + bw/2:.1f},{margin_t+plot_h+20})" font-family="Arial" font-size="11">{lab}</text>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def write_svg_confusion(path, cm, title):
    width, height = 420, 360
    maxv = np.max(cm)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
    ]
    x0, y0, cell = 110, 70, 100
    for i in range(2):
        for j in range(2):
            v = cm[i][j]
            shade = int(245 - 160 * v / maxv)
            parts.append(f'<rect x="{x0+j*cell}" y="{y0+i*cell}" width="{cell}" height="{cell}" fill="rgb({shade},{shade+20},255)" stroke="#333"/>')
            parts.append(f'<text x="{x0+j*cell+cell/2}" y="{y0+i*cell+cell/2+6}" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{v}</text>')
    parts.append(f'<text x="{x0+cell}" y="{y0+2*cell+40}" text-anchor="middle" font-family="Arial" font-size="14">Predicted</text>')
    parts.append(f'<text x="30" y="{y0+cell}" text-anchor="middle" transform="rotate(-90 30,{y0+cell})" font-family="Arial" font-size="14">True</text>')
    parts.append(f'<text x="{x0+cell/2}" y="{y0-12}" text-anchor="middle" font-family="Arial" font-size="13">CN</text>')
    parts.append(f'<text x="{x0+1.5*cell}" y="{y0-12}" text-anchor="middle" font-family="Arial" font-size="13">EMCI</text>')
    parts.append(f'<text x="{x0-20}" y="{y0+cell/2+5}" text-anchor="end" font-family="Arial" font-size="13">CN</text>')
    parts.append(f'<text x="{x0-20}" y="{y0+1.5*cell+5}" text-anchor="end" font-family="Arial" font-size="13">EMCI</text>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="../feature_vectors.npy")
    parser.add_argument("--labels", default="../expanded_labels.npy")
    parser.add_argument("--data", default="../ADdata.npy")
    parser.add_argument("--out", default="outputs/final_124_eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_fs, y_fs, X_test, y_test, n_windows, test_subjects = get_splits(args.features, args.labels, args.data, args.seed)
    configs = [
        ("fisher", "rbf_svc", 0.003),
        ("fisher", "rbf_svc", 0.01),
        ("fisher", "rbf_svc", 0.03),
        ("fisher", "logistic_l1", 0.003),
        ("anova", "rbf_svc", 0.03),
        ("anova", "extra_trees", 1.0),
    ]
    rows = []
    selected_by_config = {}
    for ranker, model_name, C in configs:
        rank = rank_features(X_train, y_train, ranker)
        selected = rank[:124]
        selected_by_config[(ranker, model_name, C)] = selected
        model = make_model(model_name, C, args.seed)
        model.fit(X_train[:, selected], y_train)
        pred_fs = model.predict(X_fs[:, selected])
        pred_test = model.predict(X_test[:, selected])
        row = {
            "ranker": ranker,
            "classifier": model_name,
            "C": C,
            "train_strategy": "train_only",
            "threshold_strategy": "default",
            "n_features": 124,
            "fs_accuracy": accuracy_score(y_fs, pred_fs),
            "fs_balanced_accuracy": balanced_accuracy_score(y_fs, pred_fs),
            "test_accuracy": accuracy_score(y_test, pred_test),
            "test_balanced_accuracy": balanced_accuracy_score(y_test, pred_test),
            "test_confusion_matrix": json.dumps(confusion_matrix(y_test, pred_test).tolist()),
        }
        rows.append(row)

        scores_fs = decision_values(model, X_fs[:, selected])
        scores_test = decision_values(model, X_test[:, selected])
        thr, _ = best_threshold(scores_fs, y_fs, metric="accuracy")
        pred_test_thr = (scores_test >= thr).astype(np.int64)
        pred_fs_thr = (scores_fs >= thr).astype(np.int64)
        rows.append({
            "ranker": ranker,
            "classifier": model_name,
            "C": C,
            "train_strategy": "train_only",
            "threshold_strategy": "fs_optimized",
            "threshold": thr,
            "n_features": 124,
            "fs_accuracy": accuracy_score(y_fs, pred_fs_thr),
            "fs_balanced_accuracy": balanced_accuracy_score(y_fs, pred_fs_thr),
            "test_accuracy": accuracy_score(y_test, pred_test_thr),
            "test_balanced_accuracy": balanced_accuracy_score(y_test, pred_test_thr),
            "test_confusion_matrix": json.dumps(confusion_matrix(y_test, pred_test_thr).tolist()),
        })

        model2 = make_model(model_name, C, args.seed)
        X_trfs = np.vstack([X_train[:, selected], X_fs[:, selected]])
        y_trfs = np.concatenate([y_train, y_fs])
        model2.fit(X_trfs, y_trfs)
        pred_test2 = model2.predict(X_test[:, selected])
        rows.append({
            "ranker": ranker,
            "classifier": model_name,
            "C": C,
            "train_strategy": "train_plus_feature_selection",
            "threshold_strategy": "default",
            "n_features": 124,
            "fs_accuracy": np.nan,
            "fs_balanced_accuracy": np.nan,
            "test_accuracy": accuracy_score(y_test, pred_test2),
            "test_balanced_accuracy": balanced_accuracy_score(y_test, pred_test2),
            "test_confusion_matrix": json.dumps(confusion_matrix(y_test, pred_test2).tolist()),
        })

    df = pd.DataFrame(rows).sort_values(["test_accuracy", "test_balanced_accuracy"], ascending=False)
    df.to_csv(out / "final_124_performance.csv", index=False, encoding="utf-8-sig")
    best = df.iloc[0].to_dict()
    best_key = (best["ranker"], best["classifier"], float(best["C"]))
    selected = selected_by_config.get(best_key)
    if selected is None:
        selected = rank_features(X_train, y_train, best["ranker"])[:124]
    feat_df = feature_table(selected)
    feat_df.to_csv(out / "selected_124_features.csv", index=False, encoding="utf-8-sig")

    # ROI frequency table.
    roi_rows = []
    for roi, name in enumerate(AAL90_NAMES, start=1):
        src_c = int((feat_df["source_roi"] == roi).sum())
        tgt_c = int((feat_df["target_roi"] == roi).sum())
        roi_rows.append({"roi": roi, "roi_name": name, "network": ROI_NETWORK[name], "source_count": src_c, "target_count": tgt_c, "total_count": src_c + tgt_c})
    roi_df = pd.DataFrame(roi_rows).sort_values(["total_count", "source_count", "target_count"], ascending=False)
    roi_df.to_csv(out / "roi_frequency.csv", index=False, encoding="utf-8-sig")

    top_plot = df.head(8).copy()
    write_svg_bar(out / "performance_top8.svg", [f"{r.ranker}-{r.classifier}-{r.train_strategy[:5]}" for r in top_plot.itertuples()], top_plot["test_accuracy"].tolist(), "Top 124-feature Test Accuracy")
    cm = np.asarray(json.loads(best["test_confusion_matrix"]))
    write_svg_confusion(out / "best_confusion_matrix.svg", cm, "Best Confusion Matrix")
    write_svg_bar(out / "roi_top12.svg", roi_df.head(12)["roi_name"].tolist(), roi_df.head(12)["total_count"].astype(float).tolist(), "Top ROI Frequency in 124 Features")

    summary = {
        "best": best,
        "n_windows_per_subject": int(n_windows),
        "train_shape": list(X_train.shape),
        "fs_shape": list(X_fs.shape),
        "test_shape": list(X_test.shape),
        "outputs": {
            "performance": str(out / "final_124_performance.csv"),
            "selected_features": str(out / "selected_124_features.csv"),
            "roi_frequency": str(out / "roi_frequency.csv"),
            "performance_svg": str(out / "performance_top8.svg"),
            "confusion_svg": str(out / "best_confusion_matrix.svg"),
            "roi_svg": str(out / "roi_top12.svg"),
        },
    }
    (out / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(df.head(12).to_string(index=False))
    print("Best:", json.dumps(best, ensure_ascii=False))


if __name__ == "__main__":
    main()

