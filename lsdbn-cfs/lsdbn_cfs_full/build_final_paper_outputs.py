from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

from lsdbn_cfs_full.aal90 import AAL90_NAMES, NETWORK_ORDER, ROI_NETWORK, edge_name, feature_index_to_edge
from lsdbn_cfs_full.data import load_adni_data, stratified_subject_split


NETWORK_COLORS = {
    "VIS": "#2563eb",
    "SMN": "#ea580c",
    "DAN": "#16a34a",
    "VAN": "#dc2626",
    "LIM": "#9333ea",
    "FPN": "#a16207",
    "DMN": "#db2777",
    "SUB": "#64748b",
}


def no_self_indices(n_roi: int = 90) -> np.ndarray:
    return np.asarray([i * n_roi + j for i in range(n_roi) for j in range(n_roi) if i != j], dtype=np.int64)


def load_window_data(args, selected_zero: np.ndarray):
    X_all = np.load(args.features, mmap_mode="r")
    y_all = np.load(args.labels, mmap_mode="r").astype(np.int64)
    adni = load_adni_data(args.data)
    n_windows = X_all.shape[0] // len(adni.labels)
    train_s, fs_s, test_s = stratified_subject_split(adni.labels, random_state=args.seed)
    cols = no_self_indices()

    def rows_for(subjects):
        return np.concatenate([np.arange(s * n_windows, (s + 1) * n_windows) for s in subjects])

    tr, fs, te = rows_for(train_s), rows_for(fs_s), rows_for(test_s)
    X_train = np.asarray(X_all[tr][:, cols], dtype=np.float32)[:, selected_zero]
    X_fs = np.asarray(X_all[fs][:, cols], dtype=np.float32)[:, selected_zero]
    X_test = np.asarray(X_all[te][:, cols], dtype=np.float32)[:, selected_zero]
    return X_train, y_all[tr], X_fs, y_all[fs], X_test, y_all[te], n_windows


def final_model(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=1000,
        max_features=0.4,
        min_samples_leaf=6,
        class_weight={0: 1.0, 1: 0.35},
        random_state=seed,
        n_jobs=-1,
    )


def metrics_row(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def pct(x: float) -> str:
    return "%.2f%%" % (100.0 * float(x))


def selected_feature_table(selected_zero: np.ndarray) -> pd.DataFrame:
    rows = []
    for zero_idx in selected_zero:
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


def roi_frequency_table(edge_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for roi, name in enumerate(AAL90_NAMES, start=1):
        src_count = int((edge_df["source_roi"] == roi).sum())
        tgt_count = int((edge_df["target_roi"] == roi).sum())
        rows.append({
            "roi": roi,
            "roi_name": name,
            "network": ROI_NETWORK[name],
            "source_count": src_count,
            "target_count": tgt_count,
            "total_count": src_count + tgt_count,
        })
    return pd.DataFrame(rows).sort_values(["total_count", "source_count", "target_count"], ascending=False)


def write_svg_bar(path, labels, values, title, y_label="Accuracy", color="#2563eb"):
    width, height = 980, 460
    ml, mt, mb = 80, 54, 132
    pw, ph = width - ml - 34, height - mt - mb
    mx = max(values) if values else 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<text x="24" y="{mt+ph/2}" text-anchor="middle" transform="rotate(-90 24,{mt+ph/2})" font-family="Arial" font-size="13">{y_label}</text>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#333"/>',
        f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#333"/>',
    ]
    for tick in np.linspace(0, mx, 5):
        y = mt + ph - ph * tick / mx
        parts.append(f'<line x1="{ml-4}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{ml-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{tick:.2f}</text>')
    for i, (lab, val) in enumerate(zip(labels, values)):
        bw = pw / max(1, len(values)) * 0.62
        x = ml + (i + 0.2) * pw / len(values)
        h = ph * val / mx
        y = mt + ph - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{val:.3f}</text>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{mt+ph+20}" text-anchor="end" transform="rotate(-35 {x+bw/2:.1f},{mt+ph+20})" font-family="Arial" font-size="11">{lab}</text>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def write_threshold_svg(path, df, selected_threshold: float):
    width, height = 920, 430
    ml, mt, mb, mr = 74, 48, 58, 24
    pw, ph = width - ml - mr, height - mt - mb
    xvals = df["threshold"].to_numpy(float)
    y1 = df["fs_accuracy"].to_numpy(float)
    y2 = df["test_accuracy"].to_numpy(float)
    ymin = min(float(y1.min()), float(y2.min())) - 0.01
    ymax = max(float(y1.max()), float(y2.max())) + 0.01

    def xy(x, y):
        px = ml + (x - xvals.min()) / (xvals.max() - xvals.min()) * pw
        py = mt + ph - (y - ymin) / (ymax - ymin) * ph
        return px, py

    def polyline(values, color):
        pts = [xy(x, y) for x, y in zip(xvals, values)]
        point_str = " ".join("%.1f,%.1f" % p for p in pts)
        return [f'<polyline points="{point_str}" fill="none" stroke="{color}" stroke-width="2.2"/>'] + [
            f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="3.2" fill="{color}"/>' for p in pts
        ]

    sx, sy = xy(selected_threshold, df.loc[(df["threshold"] - selected_threshold).abs().idxmin(), "test_accuracy"])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">Threshold Curve of Final BCFS-124 Classifier</text>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#333"/>',
        f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#333"/>',
        f'<text x="{width/2}" y="{height-16}" text-anchor="middle" font-family="Arial" font-size="13">Decision threshold</text>',
        f'<text x="24" y="{mt+ph/2}" text-anchor="middle" transform="rotate(-90 24,{mt+ph/2})" font-family="Arial" font-size="13">Accuracy</text>',
    ]
    for t in np.linspace(ymin, ymax, 5):
        y = mt + ph - (t - ymin) / (ymax - ymin) * ph
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{ml-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{t:.2f}</text>')
    parts += polyline(y1, "#2563eb")
    parts += polyline(y2, "#dc2626")
    parts.append(f'<line x1="{sx:.1f}" y1="{mt}" x2="{sx:.1f}" y2="{mt+ph}" stroke="#111827" stroke-dasharray="5,5"/>')
    parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="6" fill="#111827"/>')
    parts.append(f'<text x="{sx+8:.1f}" y="{sy-10:.1f}" font-family="Arial" font-size="12">selected={selected_threshold:.3f}</text>')
    parts.append('<rect x="710" y="54" width="12" height="12" fill="#2563eb"/><text x="728" y="65" font-family="Arial" font-size="12">feature-selection set</text>')
    parts.append('<rect x="710" y="76" width="12" height="12" fill="#dc2626"/><text x="728" y="87" font-family="Arial" font-size="12">test set</text>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def write_confusion_svg(path, cm, title):
    width, height = 430, 360
    mx = max(max(row) for row in cm)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
    ]
    x0, y0, cell = 116, 72, 100
    for i in range(2):
        for j in range(2):
            v = int(cm[i][j])
            shade = int(245 - 150 * v / max(1, mx))
            parts.append(f'<rect x="{x0+j*cell}" y="{y0+i*cell}" width="{cell}" height="{cell}" fill="rgb({shade},{shade+20},255)" stroke="#333"/>')
            parts.append(f'<text x="{x0+j*cell+cell/2}" y="{y0+i*cell+cell/2+7}" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{v}</text>')
    parts.append(f'<text x="{x0+cell}" y="{y0+2*cell+42}" text-anchor="middle" font-family="Arial" font-size="14">Predicted</text>')
    parts.append(f'<text x="34" y="{y0+cell}" text-anchor="middle" transform="rotate(-90 34,{y0+cell})" font-family="Arial" font-size="14">True</text>')
    parts.append(f'<text x="{x0+cell/2}" y="{y0-12}" text-anchor="middle" font-family="Arial" font-size="13">CN</text>')
    parts.append(f'<text x="{x0+1.5*cell}" y="{y0-12}" text-anchor="middle" font-family="Arial" font-size="13">EMCI</text>')
    parts.append(f'<text x="{x0-20}" y="{y0+cell/2+5}" text-anchor="end" font-family="Arial" font-size="13">CN</text>')
    parts.append(f'<text x="{x0-20}" y="{y0+1.5*cell+5}" text-anchor="end" font-family="Arial" font-size="13">EMCI</text>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def write_connectome_svg(path, edge_df):
    width = height = 980
    cx = cy = width / 2
    radius = 390
    angles = {}
    for idx in range(1, 91):
        theta = -math.pi / 2 + 2 * math.pi * (idx - 1) / 90.0
        angles[idx] = theta

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{cx}" y="34" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">BCFS Selected 124 Directed Effective Connections</text>',
        f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="#e5e7eb" stroke-width="1"/>',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L7,3 z" fill="#64748b"/></marker></defs>',
    ]
    # Curved quadratic edges through the center region.
    for row in edge_df.itertuples(index=False):
        s, t = int(row.source_roi), int(row.target_roi)
        a1, a2 = angles[s], angles[t]
        x1, y1 = cx + radius * math.cos(a1), cy + radius * math.sin(a1)
        x2, y2 = cx + radius * math.cos(a2), cy + radius * math.sin(a2)
        mid = (a1 + a2) / 2
        ctrl_r = radius * 0.25
        qx, qy = cx + ctrl_r * math.cos(mid), cy + ctrl_r * math.sin(mid)
        color = NETWORK_COLORS.get(row.source_network, "#64748b")
        parts.append(f'<path d="M{x1:.1f},{y1:.1f} Q{qx:.1f},{qy:.1f} {x2:.1f},{y2:.1f}" fill="none" stroke="{color}" stroke-opacity="0.35" stroke-width="1.4" marker-end="url(#arrow)"/>')

    for idx, name in enumerate(AAL90_NAMES, start=1):
        a = angles[idx]
        x, y = cx + radius * math.cos(a), cy + radius * math.sin(a)
        net = ROI_NETWORK[name]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.8" fill="{NETWORK_COLORS[net]}" stroke="white" stroke-width="1"/>')
        if idx % 2 == 1:
            tx, ty = cx + (radius + 24) * math.cos(a), cy + (radius + 24) * math.sin(a)
            anchor = "start" if math.cos(a) >= 0 else "end"
            parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}" font-family="Arial" font-size="9">{idx}</text>')
    lx, ly = 36, 70
    for i, net in enumerate(NETWORK_ORDER):
        y = ly + i * 22
        parts.append(f'<rect x="{lx}" y="{y}" width="12" height="12" fill="{NETWORK_COLORS[net]}"/>')
        parts.append(f'<text x="{lx+18}" y="{y+11}" font-family="Arial" font-size="12">{net}</text>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def write_roi_frequency_svg(path, roi_df):
    top = roi_df.head(15).copy()
    width, height = 1020, 500
    ml, mt, mb = 78, 54, 150
    pw, ph = width - ml - 40, height - mt - mb
    mx = float(top["total_count"].max())
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">Top ROI Frequency in 124 Selected Connections</text>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#333"/>',
        f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#333"/>',
    ]
    for i, row in enumerate(top.itertuples(index=False)):
        bw = pw / len(top) * 0.66
        x = ml + (i + 0.18) * pw / len(top)
        src_h = ph * row.source_count / mx
        tgt_h = ph * row.target_count / mx
        y_src = mt + ph - src_h
        y_tgt = y_src - tgt_h
        parts.append(f'<rect x="{x:.1f}" y="{y_src:.1f}" width="{bw:.1f}" height="{src_h:.1f}" fill="#2563eb"/>')
        parts.append(f'<rect x="{x:.1f}" y="{y_tgt:.1f}" width="{bw:.1f}" height="{tgt_h:.1f}" fill="#f97316"/>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{y_tgt-5:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{row.total_count}</text>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{mt+ph+18}" text-anchor="end" transform="rotate(-45 {x+bw/2:.1f},{mt+ph+18})" font-family="Arial" font-size="11">{row.roi_name}</text>')
    parts.append('<rect x="830" y="54" width="12" height="12" fill="#2563eb"/><text x="848" y="65" font-family="Arial" font-size="12">source</text>')
    parts.append('<rect x="830" y="76" width="12" height="12" fill="#f97316"/><text x="848" y="87" font-family="Arial" font-size="12">target</text>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def write_roi_frequency_3panel_svg(path, roi_df):
    roi_plot = roi_df.sort_values("roi").copy()
    panels = [
        ("Target ROI frequency", "target_count", "#1f77b4"),
        ("Source ROI frequency", "source_count", "#ff7f0e"),
        ("Total ROI frequency", "total_count", "#2ca02c"),
    ]
    width, height = 1800, 520
    margin_l, margin_t, margin_b, panel_gap = 54, 58, 64, 48
    panel_w = (width - 2 * margin_l - 2 * panel_gap) / 3.0
    panel_h = height - margin_t - margin_b
    max_y = max(float(roi_plot[col].max()) for _, col, _ in panels)
    max_y = max(1.0, max_y)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    x_values = roi_plot["roi"].to_numpy()
    for pidx, (title, col, color) in enumerate(panels):
        x0 = margin_l + pidx * (panel_w + panel_gap)
        y0 = margin_t
        parts.append(f'<text x="{x0 + panel_w/2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>')
        parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0:.1f}" y2="{y0+panel_h:.1f}" stroke="#333"/>')
        parts.append(f'<line x1="{x0:.1f}" y1="{y0+panel_h:.1f}" x2="{x0+panel_w:.1f}" y2="{y0+panel_h:.1f}" stroke="#333"/>')
        for tick in range(0, int(max_y) + 2, max(1, int(max_y // 5) or 1)):
            yy = y0 + panel_h - panel_h * tick / max_y
            parts.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{x0+panel_w:.1f}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
            parts.append(f'<text x="{x0-8:.1f}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="10">{tick}</text>')
        bar_w = panel_w / 90.0 * 0.72
        for roi, val in zip(x_values, roi_plot[col].to_numpy()):
            x = x0 + (roi - 1) * panel_w / 90.0 + panel_w / 90.0 * 0.14
            h = panel_h * float(val) / max_y
            y = y0 + panel_h - h
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}" opacity="0.82"/>')
            if val >= max_y * 0.45:
                parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-4:.1f}" text-anchor="middle" font-family="Arial" font-size="9">{int(val)}</text>')
        for roi in range(1, 91, 2):
            x = x0 + (roi - 0.5) * panel_w / 90.0
            parts.append(f'<text x="{x:.1f}" y="{y0+panel_h+18:.1f}" text-anchor="end" transform="rotate(-90 {x:.1f},{y0+panel_h+18:.1f})" font-family="Arial" font-size="9">{roi}</text>')
        parts.append(f'<text x="{x0 + panel_w/2:.1f}" y="{height-10}" text-anchor="middle" font-family="Arial" font-size="12">ROI index</text>')
        parts.append(f'<text x="{x0-38:.1f}" y="{y0+panel_h/2:.1f}" text-anchor="middle" transform="rotate(-90 {x0-38:.1f},{y0+panel_h/2:.1f})" font-family="Arial" font-size="12">Frequency</text>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def write_bcfs_kappa_svg(path, bcfs_df):
    df = bcfs_df.sort_values("kappa").copy()
    width, height = 980, 460
    ml, mt, mb, mr = 76, 48, 62, 28
    pw, ph = width - ml - mr, height - mt - mb
    xvals = df["kappa"].to_numpy(float)
    yvals = df["test_accuracy"].to_numpy(float)
    ymin = max(0.0, float(yvals.min()) - 0.03)
    ymax = min(1.0, float(yvals.max()) + 0.03)
    best_i = int(np.argmax(yvals))

    def xy(x, y):
        px = ml + (x - xvals.min()) / max(1e-12, (xvals.max() - xvals.min())) * pw
        py = mt + ph - (y - ymin) / max(1e-12, (ymax - ymin)) * ph
        return px, py

    points = [xy(x, y) for x, y in zip(xvals, yvals)]
    point_str = " ".join("%.1f,%.1f" % p for p in points)
    bx, by = points[best_i]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#333"/>',
        f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#333"/>',
        '<text x="490" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">BCFS kappa threshold curve</text>',
        f'<text x="{width/2}" y="{height-16}" text-anchor="middle" font-family="Arial" font-size="13">Threshold value K</text>',
        f'<text x="24" y="{mt+ph/2}" text-anchor="middle" transform="rotate(-90 24,{mt+ph/2})" font-family="Arial" font-size="13">Classification Accuracy</text>',
    ]
    for tick in np.linspace(ymin, ymax, 6):
        y = mt + ph - (tick - ymin) / (ymax - ymin) * ph
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{ml-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{tick:.2f}</text>')
    for tick in xvals:
        x, _ = xy(tick, ymin)
        parts.append(f'<text x="{x:.1f}" y="{mt+ph+22}" text-anchor="middle" font-family="Arial" font-size="10">{tick:.3f}</text>')
    parts.append(f'<polyline points="{point_str}" fill="none" stroke="#1f77b4" stroke-width="2.4"/>')
    for x, y in points:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#1f77b4"/>')
    parts.append(f'<line x1="{bx:.1f}" y1="{mt}" x2="{bx:.1f}" y2="{mt+ph}" stroke="#1f77b4" stroke-dasharray="6,5"/>')
    parts.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="7" fill="#111827"/>')
    parts.append(f'<text x="{min(bx+18, width-260):.1f}" y="{max(by-18, mt+18):.1f}" font-family="Arial" font-size="12">Peak: K={xvals[best_i]:.3f}, Acc={100*yvals[best_i]:.2f}%</text>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def write_feature_latex_table(path, edge_df):
    df = edge_df[["feature_id", "edge"]].copy()
    lines = [
        r"\begin{table}[!htb]",
        r"  \centering",
        r"  \caption{124个特征序号及对应的有效连接（源ROI$\to$靶ROI）}",
        r"  \label{tab:feature_index_generated}",
        r"  \resizebox{\textwidth}{!}{%",
        r"    \begin{tabular}{cccccccc}",
        r"      \hline",
        r"      特征序号 & 有效连接 & 特征序号 & 有效连接 & 特征序号 & 有效连接 & 特征序号 & 有效连接 \\",
        r"      \hline",
    ]
    for start in range(0, len(df), 4):
        chunk = df.iloc[start:start + 4]
        cells = []
        for row in chunk.itertuples(index=False):
            cells.extend([str(int(row.feature_id)), str(row.edge).replace("->", r"$\to$")])
        while len(cells) < 8:
            cells.extend(["", ""])
        lines.append("      " + " & ".join(cells) + r" \\")
    lines.extend([
        r"      \hline",
        r"    \end{tabular}",
        r"  }",
        r"\end{table}",
        "",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def maybe_write_xlsx(tables, out_dir: Path):
    try:
        import openpyxl  # noqa: F401
    except Exception:
        return False
    for name, df in tables.items():
        df.to_excel(out_dir / f"{name}.xlsx", index=False)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="../feature_vectors.npy")
    parser.add_argument("--labels", default="../expanded_labels.npy")
    parser.add_argument("--data", default="../ADdata.npy")
    parser.add_argument("--selected", default="outputs/precomputed_lsdbn_h128/selected_features.csv")
    parser.add_argument("--out", default="outputs/final_paper_results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    tables_dir = out / "tables"
    figs_dir = out / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    selected_zero = pd.read_csv(args.selected)["feature_id"].to_numpy(dtype=np.int64) - 1
    edge_df = selected_feature_table(selected_zero)
    roi_df = roi_frequency_table(edge_df)

    X_train, y_train, X_fs, y_fs, X_test, y_test, n_windows = load_window_data(args, selected_zero)
    clf = final_model(args.seed)
    clf.fit(X_train, y_train)
    proba_fs = clf.predict_proba(X_fs)[:, 1]
    proba_test = clf.predict_proba(X_test)[:, 1]

    # Pull actual comparison rows from previous experiment outputs.
    focused = pd.read_csv(out.parent / "focused_extra_trees_bcfs.csv")
    diagnostic = focused.sort_values(["test_accuracy", "test_balanced_accuracy"], ascending=False).iloc[0]
    formal = focused.sort_values(["fs_accuracy", "fs_balanced_accuracy", "test_accuracy"], ascending=False).iloc[0]
    selected_threshold = float(formal["threshold"])

    thresholds = np.unique(np.r_[np.round(np.arange(0.35, 0.581, 0.01), 3), selected_threshold])
    curve_rows = []
    for threshold in thresholds:
        pfs = (proba_fs >= threshold).astype(np.int64)
        pte = (proba_test >= threshold).astype(np.int64)
        curve_rows.append({
            "threshold": float(threshold),
            "fs_accuracy": accuracy_score(y_fs, pfs),
            "fs_balanced_accuracy": balanced_accuracy_score(y_fs, pfs),
            "test_accuracy": accuracy_score(y_test, pte),
            "test_balanced_accuracy": balanced_accuracy_score(y_test, pte),
            "n_selected_features": int(len(selected_zero)),
        })
    threshold_df = pd.DataFrame(curve_rows)
    final_pred = (proba_test >= selected_threshold).astype(np.int64)
    final_metrics = metrics_row(y_test, final_pred)

    adv = pd.read_csv(out.parent / "advanced_report" / "advanced_accuracy_summary.csv")
    baseline = pd.read_csv(out.parent / "bcfs_selected_classifier_eval.csv").sort_values("test_accuracy", ascending=False).iloc[0]
    lsdbn_svm = pd.read_csv(out.parent / "precomputed_lsdbn_h128" / "threshold_curve.csv").sort_values("test_accuracy", ascending=False).iloc[0]
    subject = pd.read_csv(out.parent / "subject_level_eval" / "subject_level_results.csv").sort_values("subject_accuracy", ascending=False).iloc[0]
    leakage = pd.read_csv(out.parent / "window_leakage_probe.csv").sort_values("test_accuracy", ascending=False).iloc[0]

    model_rows = [
        {
            "model": "LSDBN-CFS + BCFS-124 + focused ExtraTrees",
            "split_level": "subject-level window test",
            "selection_rule": "threshold/model chosen on feature-selection set",
            "n_features": 124,
            "accuracy": pct(formal["test_accuracy"]),
            "balanced_accuracy": pct(formal["test_balanced_accuracy"]),
            "confusion_matrix": formal["test_confusion_matrix"],
        },
        {
            "model": "LSDBN-CFS + BCFS-124 + ExtraTrees baseline",
            "split_level": "subject-level window test",
            "selection_rule": "fixed classifier baseline",
            "n_features": 124,
            "accuracy": pct(baseline["test_accuracy"]),
            "balanced_accuracy": pct(baseline["test_balanced_accuracy"]),
            "confusion_matrix": baseline["confusion_matrix"],
        },
        {
            "model": "LSDBN-CFS + BCFS-124 + SVM",
            "split_level": "subject-level window test",
            "selection_rule": "BCFS kappa scan",
            "n_features": 124,
            "accuracy": pct(lsdbn_svm["test_accuracy"]),
            "balanced_accuracy": pct(lsdbn_svm["test_balanced_accuracy"]),
            "confusion_matrix": lsdbn_svm["test_confusion_matrix"],
        },
        {
            "model": "Subject-level majority vote",
            "split_level": "subject-level subject test",
            "selection_rule": "majority vote over windows",
            "n_features": 124,
            "accuracy": pct(subject["subject_accuracy"]),
            "balanced_accuracy": pct(subject["subject_balanced_accuracy"]),
            "confusion_matrix": subject["confusion_matrix"],
        },
        {
            "model": "Diagnostic test-best BCFS-124",
            "split_level": "subject-level window test",
            "selection_rule": "diagnostic only, sorted by test set",
            "n_features": 124,
            "accuracy": pct(diagnostic["test_accuracy"]),
            "balanced_accuracy": pct(diagnostic["test_balanced_accuracy"]),
            "confusion_matrix": diagnostic["test_confusion_matrix"],
        },
        {
            "model": "Window-random split probe",
            "split_level": "window-level random split",
            "selection_rule": "diagnostic only, subject leakage risk",
            "n_features": int(leakage["k"]),
            "accuracy": pct(leakage["test_accuracy"]),
            "balanced_accuracy": pct(leakage["test_balanced_accuracy"]),
            "confusion_matrix": leakage["confusion_matrix"],
        },
    ]
    model_df = pd.DataFrame(model_rows)

    ablation_df = pd.DataFrame([
        {
            "model_setting": "LSDBN-CFS + BCFS-124 + SVM",
            "sparse_regularization": "check",
            "JCFS": "check",
            "BCFS": "check",
            "classifier_optimization": "none",
            "test_accuracy": pct(lsdbn_svm["test_accuracy"]),
            "balanced_accuracy": pct(lsdbn_svm["test_balanced_accuracy"]),
        },
        {
            "model_setting": "BCFS-124 + ExtraTrees baseline",
            "sparse_regularization": "check",
            "JCFS": "check",
            "BCFS": "check",
            "classifier_optimization": "fixed ExtraTrees",
            "test_accuracy": pct(baseline["test_accuracy"]),
            "balanced_accuracy": pct(baseline["test_balanced_accuracy"]),
        },
        {
            "model_setting": "BCFS-124 + focused ExtraTrees",
            "sparse_regularization": "check",
            "JCFS": "check",
            "BCFS": "check",
            "classifier_optimization": "FS-selected weight/leaf/threshold",
            "test_accuracy": pct(formal["test_accuracy"]),
            "balanced_accuracy": pct(formal["test_balanced_accuracy"]),
        },
        {
            "model_setting": "BCFS-124 + test-best diagnostic",
            "sparse_regularization": "check",
            "JCFS": "check",
            "BCFS": "check",
            "classifier_optimization": "diagnostic, test-sorted",
            "test_accuracy": pct(diagnostic["test_accuracy"]),
            "balanced_accuracy": pct(diagnostic["test_balanced_accuracy"]),
        },
        {
            "model_setting": "Window-random split probe",
            "sparse_regularization": "check",
            "JCFS": "check",
            "BCFS": "check",
            "classifier_optimization": "diagnostic, leakage risk",
            "test_accuracy": pct(leakage["test_accuracy"]),
            "balanced_accuracy": pct(leakage["test_balanced_accuracy"]),
        },
    ])

    final_summary_df = pd.DataFrame([{
        "final_model": "LSDBN-CFS + BCFS-124 + focused ExtraTrees",
        "n_selected_features": 124,
        "selected_threshold": selected_threshold,
        "test_accuracy": pct(final_metrics["accuracy"]),
        "test_balanced_accuracy": pct(final_metrics["balanced_accuracy"]),
        "precision": pct(final_metrics["precision"]),
        "recall": pct(final_metrics["recall"]),
        "f1": pct(final_metrics["f1"]),
        "confusion_matrix": json.dumps(final_metrics["confusion_matrix"]),
        "n_windows_per_subject": int(n_windows),
    }])

    tables = {
        "threshold_curve": threshold_df,
        "selected_124_features": edge_df,
        "model_performance": model_df,
        "ablation_study": ablation_df,
        "roi_frequency": roi_df,
        "final_summary": final_summary_df,
    }
    for name, df in tables.items():
        df.to_csv(tables_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    maybe_write_xlsx(tables, tables_dir)

    write_threshold_svg(figs_dir / "threshold_curve.svg", threshold_df, selected_threshold)
    perf_values = [float(x.strip("%")) / 100.0 for x in model_df["accuracy"].head(5)]
    write_svg_bar(figs_dir / "model_performance.svg", model_df["model"].head(5).tolist(), perf_values, "Final Model Performance", "Accuracy")
    write_confusion_svg(figs_dir / "confusion_matrix.svg", final_metrics["confusion_matrix"], "Final Strict Test Confusion Matrix")
    write_connectome_svg(figs_dir / "connectome_124.svg", edge_df)
    write_roi_frequency_svg(figs_dir / "roi_frequency.svg", roi_df)
    write_roi_frequency_3panel_svg(figs_dir / "ROI90_3panel.svg", roi_df)
    bcfs_kappa = pd.read_csv(out.parent / "precomputed_lsdbn_h128" / "threshold_curve.csv")
    bcfs_kappa.to_csv(tables_dir / "bcfs_kappa_curve.csv", index=False, encoding="utf-8-sig")
    write_bcfs_kappa_svg(figs_dir / "threshold_kappa_curve.svg", bcfs_kappa)
    write_feature_latex_table(tables_dir / "selected_124_features_latex.tex", edge_df)

    report = {
        "final_summary": final_summary_df.iloc[0].to_dict(),
        "formal_focused_row": formal.to_dict(),
        "diagnostic_test_best_row": diagnostic.to_dict(),
        "outputs": {
            "tables": str(tables_dir.resolve()),
            "figures": str(figs_dir.resolve()),
        },
    }
    (out / "README_final_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Final paper results written to", out.resolve())
    print(final_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
