from __future__ import annotations

import argparse
from pathlib import Path

from lsdbn_cfs_full.config import ExperimentConfig, preset_config
from lsdbn_cfs_full.experiments import run_experiment_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LSDBN-CFS experiments.")
    parser.add_argument("--data", default="../ADdata.npy", help="Path to ADdata.npy or ADdata_arrays.npz")
    parser.add_argument("--out", default="outputs/quick_run", help="Output directory")
    parser.add_argument("--preset", choices=["quick", "paper"], default="quick")
    parser.add_argument("--methods", choices=["core", "all"], default="all")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--input-prefilter-k", type=int, default=None)
    parser.add_argument("--selected-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force-recompute-dec", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = preset_config(args.preset)
    cfg.data_path = str(Path(args.data))
    cfg.output_dir = str(Path(args.out))
    cfg.methods = args.methods
    if args.repeats is not None:
        cfg.repeats = args.repeats
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.input_prefilter_k is not None:
        cfg.input_prefilter_k = args.input_prefilter_k
    if args.selected_k is not None:
        cfg.selected_k = args.selected_k
    if args.seed is not None:
        cfg.random_state = args.seed
    cfg.force_recompute_dec = bool(args.force_recompute_dec)
    run_experiment_suite(cfg)


if __name__ == "__main__":
    main()

