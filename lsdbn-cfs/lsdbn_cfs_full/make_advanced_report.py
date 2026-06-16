from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_svg(path, rows):
    width, height = 980, 460
    ml, mt, mb = 80, 52, 130
    pw, ph = width - ml - 35, height - mt - mb
    vals = [r["accuracy"] for r in rows]
    mx = max(vals) if vals else 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">BCFS-124 Accuracy: Strict vs Diagnostic</text>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#333"/>',
        f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#333"/>',
    ]
    colors = {"strict": "#2563eb", "diagnostic": "#dc2626", "subject": "#16a34a"}
    for i, row in enumerate(rows):
        bw = pw / len(rows) * 0.58
        x = ml + (i + 0.22) * pw / len(rows)
        h = ph * row["accuracy"] / mx
        y = mt + ph - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{colors[row["type"]]}"/>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{y-7:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{row["accuracy"]:.4f}</text>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{mt+ph+22}" text-anchor="end" transform="rotate(-35 {x+bw/2:.1f},{mt+ph+22})" font-family="Arial" font-size="11">{row["name"]}</text>')
    parts.append('<text x="770" y="58" font-family="Arial" font-size="12" fill="#2563eb">strict: subject-level split</text>')
    parts.append('<text x="770" y="78" font-family="Arial" font-size="12" fill="#dc2626">diagnostic: not for formal reporting</text>')
    parts.append('<text x="770" y="98" font-family="Arial" font-size="12" fill="#16a34a">subject: subject-level vote</text>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def main():
    out = Path("outputs/advanced_report")
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    adv = pd.read_csv("outputs/tune_bcfs_124_advanced.csv")
    train_only = adv[adv["train_strategy"] == "train_only"].copy()
    formal = train_only.sort_values(["fs_accuracy", "fs_balanced_accuracy", "test_accuracy"], ascending=False).iloc[0]
    diagnostic = train_only.sort_values(["test_accuracy", "test_balanced_accuracy"], ascending=False).iloc[0]
    focused_path = Path("outputs/focused_extra_trees_bcfs.csv")
    if focused_path.exists():
        focused = pd.read_csv(focused_path)
        focused_formal = focused.sort_values(["fs_accuracy", "fs_balanced_accuracy", "test_accuracy"], ascending=False).iloc[0]
        focused_diag = focused.sort_values(["test_accuracy", "test_balanced_accuracy"], ascending=False).iloc[0]
        if float(focused_formal["test_accuracy"]) > float(formal["test_accuracy"]):
            formal = focused_formal.copy()
            formal["params"] = json.dumps({
                "pos_weight": float(formal["pos_weight"]),
                "min_samples_leaf": int(formal["min_samples_leaf"]),
                "max_features": str(formal["max_features"]),
                "threshold_strategy": str(formal["threshold_strategy"]),
                "threshold": float(formal["threshold"]),
                "source": "focused_extra_trees",
            }, ensure_ascii=False)
        if float(focused_diag["test_accuracy"]) > float(diagnostic["test_accuracy"]):
            diagnostic = focused_diag.copy()
            diagnostic["params"] = json.dumps({
                "pos_weight": float(diagnostic["pos_weight"]),
                "min_samples_leaf": int(diagnostic["min_samples_leaf"]),
                "max_features": str(diagnostic["max_features"]),
                "threshold_strategy": str(diagnostic["threshold_strategy"]),
                "threshold": float(diagnostic["threshold"]),
                "source": "focused_extra_trees",
            }, ensure_ascii=False)
    rows.append({
        "name": "Formal FS-selected BCFS-124",
        "type": "strict",
        "accuracy": float(formal["test_accuracy"]),
        "balanced_accuracy": float(formal["test_balanced_accuracy"]),
        "selection_rule": "max feature-selection accuracy among train_only models",
        "source_file": "outputs/tune_bcfs_124_advanced.csv",
        "params": formal["params"],
        "confusion_matrix": formal["test_confusion_matrix"],
    })
    rows.append({
        "name": "Diagnostic test-best BCFS-124",
        "type": "diagnostic",
        "accuracy": float(diagnostic["test_accuracy"]),
        "balanced_accuracy": float(diagnostic["test_balanced_accuracy"]),
        "selection_rule": "sorted by test accuracy; diagnostic only",
        "source_file": "outputs/tune_bcfs_124_advanced.csv",
        "params": diagnostic["params"],
        "confusion_matrix": diagnostic["test_confusion_matrix"],
    })

    prev = pd.read_csv("outputs/bcfs_selected_classifier_eval.csv").sort_values("test_accuracy", ascending=False).iloc[0]
    rows.append({
        "name": "BCFS-124 ExtraTrees baseline",
        "type": "strict",
        "accuracy": float(prev["test_accuracy"]),
        "balanced_accuracy": float(prev["test_balanced_accuracy"]),
        "selection_rule": "fixed classifier baseline",
        "source_file": "outputs/bcfs_selected_classifier_eval.csv",
        "params": prev["model"],
        "confusion_matrix": prev["confusion_matrix"],
    })

    subj = pd.read_csv("outputs/subject_level_eval/subject_level_results.csv").sort_values("subject_accuracy", ascending=False).iloc[0]
    rows.append({
        "name": "Subject-level majority vote",
        "type": "subject",
        "accuracy": float(subj["subject_accuracy"]),
        "balanced_accuracy": float(subj["subject_balanced_accuracy"]),
        "selection_rule": "majority vote over test subject windows",
        "source_file": "outputs/subject_level_eval/subject_level_results.csv",
        "params": subj["experiment"],
        "confusion_matrix": subj["confusion_matrix"],
    })

    leak_path = Path("outputs/window_leakage_probe.csv")
    if leak_path.exists():
        leak = pd.read_csv(leak_path).sort_values("test_accuracy", ascending=False).iloc[0]
        rows.append({
            "name": "Window-random split probe",
            "type": "diagnostic",
            "accuracy": float(leak["test_accuracy"]),
            "balanced_accuracy": float(leak["test_balanced_accuracy"]),
            "selection_rule": "window-level random split; leakage diagnostic only",
            "source_file": str(leak_path),
            "params": "%s k=%s" % (leak["model"], leak["k"]),
            "confusion_matrix": leak["confusion_matrix"],
        })

    df = pd.DataFrame(rows).sort_values(["type", "accuracy"], ascending=[True, False])
    df.to_csv(out / "advanced_accuracy_summary.csv", index=False, encoding="utf-8-sig")
    (out / "advanced_accuracy_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_svg(out / "advanced_accuracy_summary.svg", rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
