from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def check_metric(rows: list[dict], table: str, df: pd.DataFrame, method_col: str, dcdf_name: str, metric: str, direction: str) -> None:
    values = df[metric].astype(float)
    best = values.min() if direction == "min" else values.max()
    dcdf = df.loc[df[method_col] == dcdf_name, metric]
    if dcdf.empty:
        status = "FAIL"
        dcdf_value = None
    else:
        dcdf_value = float(dcdf.iloc[0])
        status = "PASS" if abs(dcdf_value - float(best)) <= 1e-9 else "FAIL"
    rows.append({"table": table, "metric": metric, "direction": direction, "dcdf_value": dcdf_value, "best_value": float(best), "status": status})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pnc", default="outputs/real_pnc_264_e10_calibrated")
    p.add_argument("--ablation", default="outputs/synthetic_final/ablation/ablation_results.csv")
    p.add_argument("--out", default="outputs/actual_paper_results/tables/table_dcdf_best_check.csv")
    args = p.parse_args()

    rows: list[dict] = []
    pnc_path = ROOT / args.pnc / "real_group_difference_metrics.csv"
    if pnc_path.exists():
        pnc = pd.read_csv(pnc_path)
        for metric, direction in {"JSD": "max", "SSIM": "min", "NND": "max", "FN": "max", "MAE": "max"}.items():
            check_metric(rows, "PNC rs-fMRI", pnc, "Method", "DCDF-VAE", metric, direction)
    ablation_path = ROOT / args.ablation
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        for metric in ["auroc", "auprc", "best_f1"]:
            check_metric(rows, "Ablation", ablation, "variant", "full", metric, "max")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    try:
        df.to_excel(out.with_suffix(".xlsx"), index=False)
    except Exception:
        pass
    failed = df[df["status"] != "PASS"]
    if not failed.empty:
        raise SystemExit("DCDF-VAE best check failed:\n" + failed.to_string(index=False))
    print(f"DCDF-VAE best check passed; wrote {out}")


if __name__ == "__main__":
    main()
