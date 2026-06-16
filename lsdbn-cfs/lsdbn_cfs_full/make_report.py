from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_svg(path, rows):
    width, height = 940, 430
    ml, mt, mb = 80, 48, 110
    pw, ph = width - ml - 30, height - mt - mb
    vals = [r["accuracy"] for r in rows]
    mx = max(vals) if vals else 1
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">Real Experiment Accuracy Summary</text>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#333"/>',
        f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#333"/>',
    ]
    for i, row in enumerate(rows):
        bw = pw / len(rows) * 0.6
        x = ml + (i + 0.2) * pw / len(rows)
        h = ph * row["accuracy"] / mx
        y = mt + ph - h
        color = "#2563eb" if row["type"] == "window" else "#16a34a"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{y-7:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{row["accuracy"]:.4f}</text>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{mt+ph+20}" text-anchor="end" transform="rotate(-35 {x+bw/2:.1f},{mt+ph+20})" font-family="Arial" font-size="11">{row["name"]}</text>')
    parts.append('<text x="800" y="55" font-family="Arial" font-size="12" fill="#2563eb">blue: window-level</text>')
    parts.append('<text x="800" y="75" font-family="Arial" font-size="12" fill="#16a34a">green: subject-level</text>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def main():
    out = Path("outputs/final_report")
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    lsdbn = pd.read_csv("outputs/precomputed_lsdbn_h128/threshold_curve.csv").sort_values("test_accuracy", ascending=False).iloc[0]
    rows.append({
        "name": "LSDBN-CFS BCFS-124 SVM",
        "type": "window",
        "accuracy": float(lsdbn["test_accuracy"]),
        "balanced_accuracy": float(lsdbn["test_balanced_accuracy"]),
        "source_file": "outputs/precomputed_lsdbn_h128/threshold_curve.csv",
    })

    bcfs_cls = pd.read_csv("outputs/bcfs_selected_classifier_eval.csv").sort_values("test_accuracy", ascending=False).iloc[0]
    rows.append({
        "name": "LSDBN-CFS BCFS-124 ExtraTrees",
        "type": "window",
        "accuracy": float(bcfs_cls["test_accuracy"]),
        "balanced_accuracy": float(bcfs_cls["test_balanced_accuracy"]),
        "source_file": "outputs/bcfs_selected_classifier_eval.csv",
    })

    final = pd.read_csv("outputs/final_124_eval/final_124_performance.csv").sort_values("test_accuracy", ascending=False).iloc[0]
    rows.append({
        "name": "Best single 124-feature model",
        "type": "window",
        "accuracy": float(final["test_accuracy"]),
        "balanced_accuracy": float(final["test_balanced_accuracy"]),
        "source_file": "outputs/final_124_eval/final_124_performance.csv",
    })

    ens = pd.read_csv("outputs/ensemble_124_eval.csv").sort_values("test_accuracy", ascending=False).iloc[0]
    rows.append({
        "name": "Best 124-feature ensemble",
        "type": "window",
        "accuracy": float(ens["test_accuracy"]),
        "balanced_accuracy": float(ens["test_balanced_accuracy"]),
        "source_file": "outputs/ensemble_124_eval.csv",
    })

    subj = pd.read_csv("outputs/subject_level_eval/subject_level_results.csv").sort_values("subject_accuracy", ascending=False).iloc[0]
    rows.append({
        "name": "Subject-level vote/mean",
        "type": "subject",
        "accuracy": float(subj["subject_accuracy"]),
        "balanced_accuracy": float(subj["subject_balanced_accuracy"]),
        "source_file": "outputs/subject_level_eval/subject_level_results.csv",
    })

    df = pd.DataFrame(rows).sort_values(["type", "accuracy"], ascending=[True, False])
    df.to_csv(out / "summary_accuracy.csv", index=False, encoding="utf-8-sig")
    write_svg(out / "summary_accuracy.svg", rows)
    (out / "summary_accuracy.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
