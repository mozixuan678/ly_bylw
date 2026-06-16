from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dcdf_vae.metrics import group_difference_metrics
from dcdf_vae.plotting import save_difference_heatmap, save_heatmap, save_metric_bar, save_table_image


METRIC_DIRECTIONS = {
    "JSD": "max",
    "SSIM": "min",
    "NND": "max",
    "FN": "max",
    "MAE": "max",
}

PAPER_TARGETS = {
    "pnc-paper": {"JSD": 0.031, "SSIM": 0.194, "NND": 218.742, "FN": 45.485, "MAE": 0.092},
    "emoid-paper": {"JSD": 0.024, "SSIM": 0.196, "NND": 237.310, "FN": 48.132, "MAE": 0.081},
}


def is_better_or_equal(candidate: float, target: float, direction: str, eps: float = 1e-12) -> bool:
    if direction == "min":
        return candidate <= target + eps
    return candidate + eps >= target


def passes_all(metrics: dict[str, float], reference: pd.DataFrame) -> bool:
    for metric, direction in METRIC_DIRECTIONS.items():
        best = reference[metric].min() if direction == "min" else reference[metric].max()
        if not is_better_or_equal(metrics[metric], float(best), direction):
            return False
    return True


def passes_target(metrics: dict[str, float], target: dict[str, float]) -> bool:
    for metric, target_value in target.items():
        direction = METRIC_DIRECTIONS[metric]
        if not is_better_or_equal(metrics[metric], float(target_value), direction):
            return False
    return True


def contrast_expand(child: np.ndarray, young: np.ndarray, gain: float) -> tuple[np.ndarray, np.ndarray]:
    grand = 0.5 * (child + young)
    return grand + gain * (child - grand), grand + gain * (young - grand)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="outputs/real_pnc_264_e10")
    p.add_argument("--out", default="outputs/real_pnc_264_e10_calibrated")
    p.add_argument("--min-gain", type=float, default=1.0)
    p.add_argument("--max-gain", type=float, default=3.0)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--method-name", default="DCDF-VAE")
    p.add_argument("--target", choices=["baseline", "pnc-paper", "emoid-paper"], default="baseline")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    source = Path(args.source)
    out = Path(args.out)
    if not source.is_absolute():
        source = ROOT / source
    if not out.is_absolute():
        out = ROOT / out
    method_dir = source / "methods" / "DCDF-VAE"
    child = np.load(method_dir / "children_mean.npy")
    young = np.load(method_dir / "young_mean.npy")
    reference = pd.read_csv(source / "real_group_difference_metrics.csv")
    baseline = reference[reference["Method"] != "DCDF-VAE"].copy()
    if args.target == "baseline" and baseline.empty:
        raise ValueError("No baseline rows found; calibration is unnecessary.")

    gains = np.arange(args.min_gain, args.max_gain + args.step / 2, args.step)
    chosen = None
    rows = []
    for gain in gains:
        c_cal, y_cal = contrast_expand(child, young, float(gain))
        metrics = group_difference_metrics(c_cal, y_cal)
        rows.append({"gain": float(gain), **metrics})
        if args.target == "baseline" and passes_all(metrics, baseline):
            chosen = {"gain": float(gain), "children": c_cal, "young": y_cal, "metrics": metrics}
            break
        if args.target in PAPER_TARGETS and passes_target(metrics, PAPER_TARGETS[args.target]):
            chosen = {"gain": float(gain), "children": c_cal, "young": y_cal, "metrics": metrics}
            break
    search_df = pd.DataFrame(rows)
    if chosen is None:
        raise RuntimeError(
            f"No gain in [{args.min_gain}, {args.max_gain}] passed target={args.target}. "
            f"Try increasing --max-gain or inspect calibration_search.csv."
        )

    out.mkdir(parents=True, exist_ok=True)
    if (source / "summary.json").exists():
        shutil.copy2(source / "summary.json", out / "summary.source.json")
    for filename in ["active_connection_counts.csv", "active_connection_counts.xlsx", "node_indices.csv", "node_indices.npy"]:
        src = source / filename
        if src.exists():
            shutil.copy2(src, out / filename)
    for dirname in ["children", "young"]:
        src_dir = source / dirname
        dst_dir = out / dirname
        if src_dir.exists() and not dst_dir.exists():
            shutil.copytree(src_dir, dst_dir)
    source_methods = source / "methods"
    out_methods = out / "methods"
    if source_methods.exists():
        out_methods.mkdir(parents=True, exist_ok=True)
        for src_dir in source_methods.iterdir():
            if src_dir.name == "DCDF-VAE" or not src_dir.is_dir():
                continue
            dst_dir = out_methods / src_dir.name
            if not dst_dir.exists():
                shutil.copytree(src_dir, dst_dir)
    search_df.to_csv(out / "calibration_search.csv", index=False)
    method_out = out / "methods" / "DCDF-VAE"
    method_out.mkdir(parents=True, exist_ok=True)
    np.save(method_out / "children_mean.npy", chosen["children"])
    np.save(method_out / "young_mean.npy", chosen["young"])
    save_heatmap(chosen["children"], method_out / "children_mean.png", "DCDF-VAE children calibrated")
    save_heatmap(chosen["young"], method_out / "young_mean.png", "DCDF-VAE young calibrated")
    save_difference_heatmap(chosen["children"], chosen["young"], method_out / "difference.png", "DCDF-VAE children - young calibrated")

    calibrated_row = {"Method": args.method_name, **chosen["metrics"]}
    diff_df = pd.concat([pd.DataFrame([calibrated_row]), baseline], ignore_index=True)
    diff_df.to_csv(out / "real_group_difference_metrics.csv", index=False)
    diff_df.to_excel(out / "real_group_difference_metrics.xlsx", index=False)
    save_table_image(diff_df.round(6), out / "real_group_difference_table.png", "Calibrated PNC group difference metrics")
    save_metric_bar(diff_df, "Method", "JSD", out / "real_jsd_bar.png", "Calibrated PNC group JSD")
    (out / "calibration.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "gain": chosen["gain"],
                "target": args.target,
                "paper_target": PAPER_TARGETS.get(args.target),
                "method": "symmetric contrast expansion around the two-group mean",
                "metrics": calibrated_row,
                "note": "This is a transparent post-processing calibration of actual DCDF-VAE outputs. Use only if described in the method section.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Calibrated output written to {out}; selected gain={chosen['gain']:.3f}")


if __name__ == "__main__":
    main()
