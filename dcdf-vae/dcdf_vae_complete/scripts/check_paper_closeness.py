from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


PAPER_VAR = {
    "VAR-1_T250": 92.7,
    "VAR-1_T500": 95.1,
    "VAR-1_T1000": 98.9,
    "VAR-2_T250": 85.8,
    "VAR-2_T500": 93.2,
    "VAR-2_T1000": 98.4,
}

PAPER_LORENZ = {
    "Lorenz_N30_T500": 92.98,
    "Lorenz_N30_T1000": 96.21,
    "Lorenz_N30_T1500": 99.28,
    "Lorenz_N40_T500": 91.91,
    "Lorenz_N40_T1000": 96.70,
    "Lorenz_N40_T1500": 99.12,
    "Lorenz_N50_T500": 91.45,
    "Lorenz_N50_T1000": 96.09,
    "Lorenz_N50_T1500": 98.87,
}

PAPER_PNC = {"JSD": 0.031, "SSIM": 0.194, "NND": 218.742, "FN": 45.485, "MAE": 0.092}
PAPER_EMOID = {"JSD": 0.024, "SSIM": 0.196, "NND": 237.310, "FN": 48.132, "MAE": 0.081}


def rel_err(actual: float, target: float) -> float:
    return abs(actual - target) / max(abs(target), 1e-12)


def add_check(
    rows: list[dict],
    section: str,
    metric: str,
    actual: float,
    target: float,
    abs_tol: float,
    rel_tol: float | None = None,
) -> None:
    abs_error = abs(actual - target)
    relative_error = rel_err(actual, target)
    passed = abs_error <= abs_tol if rel_tol is None else (abs_error <= abs_tol or relative_error <= rel_tol)
    rows.append(
        {
            "section": section,
            "metric": metric,
            "actual": actual,
            "paper_target": target,
            "abs_error": abs_error,
            "rel_error": relative_error,
            "abs_tol": abs_tol,
            "rel_tol": "" if rel_tol is None else rel_tol,
            "status": "PASS" if passed else "FAIL",
        }
    )


def add_performance_floor_check(
    rows: list[dict],
    section: str,
    metric: str,
    actual: float,
    target: float,
    abs_tol: float,
) -> None:
    shortfall = max(0.0, target - actual)
    passed = shortfall <= abs_tol
    rows.append(
        {
            "section": section,
            "metric": metric,
            "actual": actual,
            "paper_target": target,
            "abs_error": abs(actual - target),
            "rel_error": rel_err(actual, target),
            "abs_tol": abs_tol,
            "rel_tol": "",
            "status": "PASS" if passed else "FAIL",
        }
    )


def add_directional_paper_check(
    rows: list[dict],
    section: str,
    metric: str,
    actual: float,
    target: float,
    direction: str,
    abs_tol: float,
    rel_tol: float,
) -> None:
    if direction == "min":
        miss = max(0.0, actual - target)
    else:
        miss = max(0.0, target - actual)
    passed = miss <= abs_tol or miss / max(abs(target), 1e-12) <= rel_tol
    rows.append(
        {
            "section": section,
            "metric": metric,
            "actual": actual,
            "paper_target": target,
            "abs_error": abs(actual - target),
            "rel_error": rel_err(actual, target),
            "abs_tol": abs_tol,
            "rel_tol": rel_tol,
            "status": "PASS" if passed else "FAIL",
        }
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", default="outputs/synthetic_full15/synthetic_best_results.csv")
    p.add_argument("--pnc", default="outputs/real_pnc_264_e10_calibrated/real_group_difference_metrics.csv")
    p.add_argument("--emoid", default="outputs/emoid_pnc_264_e3/real_group_difference_metrics.csv")
    p.add_argument("--synthetic-abs-tol", type=float, default=3.0, help="AUROC percentage-point tolerance.")
    p.add_argument("--real-abs-tol", type=float, default=0.02)
    p.add_argument("--real-rel-tol", type=float, default=0.20)
    p.add_argument("--norm-abs-tol", type=float, default=12.0, help="Tolerance for NND/FN scale metrics.")
    p.add_argument("--out", default="outputs/actual_paper_results/tables/table_paper_closeness_check.csv")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    rows: list[dict] = []

    synthetic_path = ROOT / args.synthetic
    if synthetic_path.exists():
        synthetic = pd.read_csv(synthetic_path)
        for _, row in synthetic.iterrows():
            dataset = str(row["dataset"])
            actual = 100.0 * float(row["auroc"])
            if dataset in PAPER_VAR:
                add_performance_floor_check(rows, "VAR", dataset, actual, PAPER_VAR[dataset], args.synthetic_abs_tol)
            elif dataset in PAPER_LORENZ:
                add_performance_floor_check(rows, "Lorenz-96", dataset, actual, PAPER_LORENZ[dataset], args.synthetic_abs_tol)

    for label, rel_path, target in [
        ("PNC rs-fMRI", args.pnc, PAPER_PNC),
        ("PNC emoid-fMRI", args.emoid, PAPER_EMOID),
    ]:
        path = ROOT / rel_path
        if not path.exists():
            continue
        df = pd.read_csv(path)
        row = df[df["Method"] == "DCDF-VAE"]
        if row.empty:
            continue
        row = row.iloc[0]
        directions = {"JSD": "max", "SSIM": "min", "NND": "max", "FN": "max", "MAE": "max"}
        for metric, target_value in target.items():
            abs_tol = args.norm_abs_tol if metric in {"NND", "FN"} else args.real_abs_tol
            add_directional_paper_check(rows, label, metric, float(row[metric]), target_value, directions[metric], abs_tol, args.real_rel_tol)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(out, index=False)
    try:
        result.to_excel(out.with_suffix(".xlsx"), index=False)
    except Exception:
        pass
    failed = result[result["status"] != "PASS"]
    if not failed.empty:
        raise SystemExit("Paper closeness check failed:\n" + failed.to_string(index=False))
    print(f"Paper closeness check passed; wrote {out}")


if __name__ == "__main__":
    main()
