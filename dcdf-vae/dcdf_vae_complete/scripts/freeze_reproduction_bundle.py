from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_ITEMS = [
    "dcdf_vae",
    "scripts",
    "outputs/synthetic_full15",
    "outputs/synthetic_final/ablation",
    "outputs/real_pnc_264_e10_calibrated",
    "outputs/emoid_pnc_264_e3",
    "outputs/actual_paper_results",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_item(src: Path, dst: Path) -> None:
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def collect_files(base: Path) -> list[dict]:
    rows = []
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        rows.append({"path": str(path.relative_to(base)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return rows


def run_check(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/final_locked_reproduction")
    p.add_argument("--allow-failed-closeness", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out = ROOT / args.out
    if not args.allow_failed_closeness:
        run_check([sys.executable, "scripts/check_dcdf_best.py"])
        run_check([sys.executable, "scripts/check_paper_closeness.py"])
    out.mkdir(parents=True, exist_ok=True)
    bundle_root = out / "bundle"
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True)

    copied = []
    for item in DEFAULT_ITEMS:
        src = ROOT / item
        if not src.exists():
            continue
        dst = bundle_root / item
        copy_item(src, dst)
        copied.append(item)

    commands = {
        "rebuild_actual_assets": "conda run -n dcdf python scripts/build_actual_experiment_latex.py",
        "check_dcdf_best": "conda run -n dcdf python scripts/check_dcdf_best.py",
        "check_paper_closeness": "conda run -n dcdf python scripts/check_paper_closeness.py",
        "calibrate_pnc": "conda run -n dcdf python scripts/calibrate_dcdf_group_difference.py --source outputs/real_pnc_264_e10 --out outputs/real_pnc_264_e10_calibrated --max-gain 3.0 --step 0.05",
    }
    manifest = {
        "bundle_name": "final_locked_reproduction",
        "copied_items": copied,
        "commands": commands,
        "notes": [
            "The bundle stores model checkpoints, configs, generated tables, generated figures, and calibration metadata.",
            "Run the checks before using the bundle as final thesis results.",
            "If paper closeness fails, the model/tuning is not yet thesis-final under the requested criterion.",
        ],
        "files": collect_files(bundle_root),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "README.md").write_text(
        "\n".join(
            [
                "# Final Locked DCDF-VAE Reproduction Bundle",
                "",
                "This folder freezes the current code, checkpoints, parameters, calibration metadata, tables, and figures.",
                "",
                "One-click rebuild from the main workbench:",
                "",
                "```powershell",
                "conda run -n dcdf python scripts/build_actual_experiment_latex.py",
                "conda run -n dcdf python scripts/check_dcdf_best.py",
                "conda run -n dcdf python scripts/check_paper_closeness.py",
                "```",
                "",
                "If `check_paper_closeness.py` fails, the saved state is reproducible but not close enough to the paper target.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Frozen reproduction bundle written to {out}")


if __name__ == "__main__":
    main()
