from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_real_pnc, run_synthetic


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["quick", "full"], default="quick")
    p.add_argument("--out", default="outputs")
    p.add_argument("--device", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    syn_args = run_synthetic.build_argparser().parse_args(
        [
            "--preset",
            args.preset,
            "--out",
            str(out / "synthetic"),
            "--device",
            args.device or "",
        ]
    )
    if args.device is None:
        syn_args.device = None
    syn_args.verbose = args.verbose
    if args.preset == "quick":
        syn_args.epochs = 18
        syn_args.ablation_epochs = 10
    run_synthetic.run(syn_args)
    if not syn_args.skip_ablation:
        run_synthetic.run_ablation(syn_args, None)

    real_args = run_real_pnc.build_argparser().parse_args(
        [
            "--preset",
            args.preset,
            "--out",
            str(out / "real_pnc"),
            "--device",
            args.device or "",
        ]
    )
    if args.device is None:
        real_args.device = None
    real_args.verbose = args.verbose
    if args.preset == "quick":
        real_args.epochs = 8
        real_args.n_nodes = 40
        real_args.subject_limit = 24
    run_real_pnc.run(real_args)


if __name__ == "__main__":
    main()
