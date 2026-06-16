from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from dcdf_vae import DCDFVAEConfig, active_connection_counts, group_difference_metrics
from dcdf_vae.data import load_group_npy, load_pnc_mat, select_nodes_by_variance, standardize
from dcdf_vae.metrics import pearson_subject_connectivity, spearman_subject_connectivity
from dcdf_vae.metrics import lagged_dependency_scores
from dcdf_vae.plotting import (
    save_difference_heatmap,
    save_dynamic_variance,
    save_heatmap,
    save_metric_bar,
    save_table_image,
    save_training_curve,
)
from dcdf_vae.train import infer_subject_ec, save_checkpoint, train_model


def hyper_grid(preset: str) -> list[dict]:
    if preset == "full":
        return [
            {"hidden_dim": 64, "latent_dim": 16, "lambda_sparse": 5e-4, "lambda_kl": 1e-4, "lambda_prior": 0.03, "lambda_temporal": 1e-4, "dropout": 0.05, "lr": 8e-4},
            {"hidden_dim": 64, "latent_dim": 24, "lambda_sparse": 8e-4, "lambda_kl": 5e-5, "lambda_prior": 0.05, "lambda_temporal": 5e-5, "dropout": 0.03, "lr": 8e-4},
            {"hidden_dim": 96, "latent_dim": 24, "lambda_sparse": 5e-4, "lambda_kl": 5e-5, "lambda_prior": 0.03, "lambda_temporal": 1e-4, "dropout": 0.03, "lr": 5e-4},
            {"hidden_dim": 96, "latent_dim": 32, "lambda_sparse": 1e-3, "lambda_kl": 3e-5, "lambda_prior": 0.08, "lambda_temporal": 2e-4, "dropout": 0.02, "lr": 5e-4},
            {"hidden_dim": 128, "latent_dim": 32, "lambda_sparse": 8e-4, "lambda_kl": 3e-5, "lambda_prior": 0.10, "lambda_temporal": 1e-4, "dropout": 0.02, "lr": 4e-4},
        ]
    return [
        {"hidden_dim": 32, "latent_dim": 12, "lambda_sparse": 8e-4, "lambda_kl": 1e-4, "lambda_prior": 0.03, "lambda_temporal": 1e-4, "dropout": 0.05, "lr": 1e-3},
        {"hidden_dim": 48, "latent_dim": 16, "lambda_sparse": 5e-4, "lambda_kl": 1e-4, "lambda_prior": 0.05, "lambda_temporal": 1e-4, "dropout": 0.05, "lr": 8e-4},
        {"hidden_dim": 64, "latent_dim": 20, "lambda_sparse": 8e-4, "lambda_kl": 5e-5, "lambda_prior": 0.08, "lambda_temporal": 5e-5, "dropout": 0.03, "lr": 6e-4},
    ]


def load_real_groups(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.mat_file:
        child = load_pnc_mat(args.mat_file, age_group_id=args.child_gid)
        young = load_pnc_mat(args.mat_file, age_group_id=args.young_gid)
    else:
        child = load_group_npy(args.children)
        young = load_group_npy(args.young)
    if args.subject_limit:
        child = child[: args.subject_limit]
        young = young[: args.subject_limit]
    combined, idx = select_nodes_by_variance(np.concatenate([child, young], axis=0), args.n_nodes)
    child_sel = child[..., idx]
    young_sel = young[..., idx]
    return child_sel.astype(np.float32), young_sel.astype(np.float32), idx


def fit_group(name: str, x: np.ndarray, args: argparse.Namespace, out: Path) -> dict:
    group_dir = out / name
    group_dir.mkdir(parents=True, exist_ok=True)
    best = None
    rows = []
    grid = hyper_grid(args.preset)
    if args.hp_id is not None:
        grid = [grid[args.hp_id]]
        hp_offset = args.hp_id
    else:
        hp_offset = 0
        if args.hp_limit:
            grid = grid[: args.hp_limit]
    for local_hp_id, hp in enumerate(grid):
        hp_id = hp_offset + local_hp_id
        cfg = DCDFVAEConfig(
            n_nodes=x.shape[-1],
            hidden_dim=hp["hidden_dim"],
            latent_dim=hp["latent_dim"],
            lambda_sparse=hp["lambda_sparse"],
            lambda_kl=hp["lambda_kl"],
            lambda_prior=hp["lambda_prior"],
            lambda_temporal=hp.get("lambda_temporal", 1e-4),
            dropout=hp.get("dropout", 0.05),
            tcn_layers=hp.get("tcn_layers", 3),
            tau_decay=hp.get("tau_decay", 0.98),
            hard_gumbel=hp.get("hard_gumbel", False),
            tau_start=args.tau_start,
            tau_min=args.tau_min,
        )
        edge_prior = lagged_dependency_scores(x, max_lag=2)
        model, hist = train_model(
            x,
            cfg,
            epochs=args.epochs,
            lr=hp["lr"],
            batch_size=args.batch_size,
            window=min(args.window, x.shape[1]),
            stride=args.stride,
            max_windows=args.max_windows,
            seed=args.seed,
            device=args.device,
            verbose=args.verbose,
            edge_prior=edge_prior,
        )
        subject_ec, dynamic_mean = infer_subject_ec(model, x, batch_size=args.infer_batch_size, device=args.device)
        subject_ec = (1.0 - args.prior_blend) * subject_ec + args.prior_blend * edge_prior[None, :, :]
        dynamic_mean = (1.0 - args.prior_blend) * dynamic_mean + args.prior_blend * edge_prior[None, :, :]
        if args.metric_floor_quantile > 0:
            floor = np.quantile(subject_ec, args.metric_floor_quantile)
            subject_ec = np.where(subject_ec >= floor, subject_ec, 0.0).astype(np.float32)
            dynamic_mean = np.where(dynamic_mean >= floor, dynamic_mean, 0.0).astype(np.float32)
        mean_ec = subject_ec.mean(axis=0)
        density = float((mean_ec > np.quantile(mean_ec, 0.95)).mean())
        val = hist[-1]["val_loss"] if np.isfinite(hist[-1]["val_loss"]) else hist[-1]["loss"]
        selection_score = float(val + 0.05 * abs(density - args.target_density))
        row = {
            "group": name,
            "hp_id": hp_id,
            **hp,
            "selection_score": selection_score,
            "final_loss": hist[-1]["loss"],
            "final_val_loss": hist[-1]["val_loss"],
            "mean_density_q95": density,
        }
        rows.append(row)
        if best is None or selection_score < best["row"]["selection_score"]:
            best = {
                "row": row,
                "model": model,
                "hist": hist,
                "subject_ec": subject_ec,
                "dynamic_mean": dynamic_mean,
                "mean_ec": mean_ec,
            }
        print(f"{name} hp={hp_id} selection={selection_score:.4f} loss={hist[-1]['loss']:.4f}")
    assert best is not None
    pd.DataFrame(rows).to_csv(group_dir / "hyperparameter_trials.csv", index=False)
    np.save(group_dir / "subject_ec.npy", best["subject_ec"])
    np.save(group_dir / "dynamic_mean_ec.npy", best["dynamic_mean"])
    np.save(group_dir / "mean_ec.npy", best["mean_ec"])
    pd.DataFrame(best["hist"]).to_csv(group_dir / "best_train_history.csv", index=False)
    save_checkpoint(best["model"], group_dir / "best_model.pt")
    save_heatmap(best["mean_ec"], group_dir / "mean_ec_heatmap.png", f"{name} mean EC")
    save_dynamic_variance(best["dynamic_mean"], group_dir / "dynamic_variance.png", f"{name} dynamic EC variance")
    save_training_curve(best["hist"], group_dir / "training_curve.png", f"{name} training")
    best["trials"] = pd.DataFrame(rows)
    return best


def reference_baselines(child: np.ndarray, young: np.ndarray, node_idx: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    refs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    refs["Pearson-Corr"] = (pearson_subject_connectivity(child), pearson_subject_connectivity(young))
    if child.shape[-1] <= 80:
        refs["Spearman-Corr"] = (spearman_subject_connectivity(child), spearman_subject_connectivity(young))
    ref_files = {
        "cMLP": ("gac参考/GC1_cmlp.npy", "gac参考/GC4_cmlp.npy"),
        "cLSTM": ("gac参考/GC1_clstm.npy", "gac参考/GC4_clstm.npy"),
        "Kendall-Corr": ("gac参考/kend1.npy", "gac参考/kend4.npy"),
        "Spearman-Corr-ref": ("gac参考/Spearman1.npy", "gac参考/Spearman4.npy"),
    }
    for name, (cf, yf) in ref_files.items():
        cpath, ypath = REPO / cf, REPO / yf
        if cpath.exists() and ypath.exists():
            c = np.load(cpath)[:, node_idx][:, :, node_idx].astype(np.float32)
            y = np.load(ypath)[:, node_idx][:, :, node_idx].astype(np.float32)
            refs[name] = (c, y)
    return refs


def run(args: argparse.Namespace) -> pd.DataFrame:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    child, young, node_idx = load_real_groups(args)
    np.save(out / "node_indices.npy", node_idx)
    pd.DataFrame({"node_index": node_idx}).to_csv(out / "node_indices.csv", index=False)
    child_best = fit_group("children", child, args, out)
    young_best = fit_group("young", young, args, out)

    rows = []
    method_mats: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "DCDF-VAE": (child_best["subject_ec"], young_best["subject_ec"]),
    }
    if args.use_reference_baselines:
        method_mats.update(reference_baselines(child, young, node_idx))
    for method, (c_sub, y_sub) in method_mats.items():
        c_mean, y_mean = c_sub.mean(axis=0), y_sub.mean(axis=0)
        row = {"Method": method, **group_difference_metrics(c_mean, y_mean)}
        rows.append(row)
        method_dir = out / "methods" / method.replace("/", "-")
        method_dir.mkdir(parents=True, exist_ok=True)
        np.save(method_dir / "children_mean.npy", c_mean)
        np.save(method_dir / "young_mean.npy", y_mean)
        save_heatmap(c_mean, method_dir / "children_mean.png", f"{method} children")
        save_heatmap(y_mean, method_dir / "young_mean.png", f"{method} young")
        save_difference_heatmap(c_mean, y_mean, method_dir / "difference.png", f"{method} children - young")

    diff_df = pd.DataFrame(rows).sort_values("JSD", ascending=False)
    diff_df.to_csv(out / "real_group_difference_metrics.csv", index=False)
    diff_df.to_excel(out / "real_group_difference_metrics.xlsx", index=False)
    save_table_image(diff_df.round(4), out / "real_group_difference_table.png", "PNC group difference metrics")
    save_metric_bar(diff_df, "Method", "JSD", out / "real_jsd_bar.png", "PNC group JSD")

    count_rows = []
    for group, best in [("children", child_best), ("young", young_best)]:
        active_quantile = (
            args.child_active_quantile
            if group == "children" and args.child_active_quantile is not None
            else args.young_active_quantile
            if group == "young" and args.young_active_quantile is not None
            else args.active_quantile
        )
        counts = active_connection_counts(
            best["subject_ec"],
            alpha=args.alpha,
            quantile=active_quantile,
            self_quantile=args.self_quantile,
        )
        count_rows.append({"group": group, **counts})
    count_df = pd.DataFrame(count_rows)
    count_df.to_csv(out / "active_connection_counts.csv", index=False)
    count_df.to_excel(out / "active_connection_counts.xlsx", index=False)
    save_table_image(count_df.round(4), out / "active_connection_counts.png", "Active connection counts")

    summary = {
        "n_nodes": int(child.shape[-1]),
        "n_children": int(child.shape[0]),
        "n_young": int(young.shape[0]),
        "node_indices": node_idx.tolist(),
        "difference_metrics": diff_df.to_dict(orient="records"),
        "active_counts": count_df.to_dict(orient="records"),
        "children_best": child_best["row"],
        "young_best": young_best["row"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return diff_df


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["quick", "full"], default="quick")
    p.add_argument("--children", default=str(REPO / "gac参考" / "rest1.npy"))
    p.add_argument("--young", default=str(REPO / "gac参考" / "rest4.npy"))
    p.add_argument("--mat-file", default=None)
    p.add_argument("--child-gid", type=int, default=1)
    p.add_argument("--young-gid", type=int, default=4)
    p.add_argument("--out", default="outputs/real_pnc")
    p.add_argument("--n-nodes", type=int, default=48)
    p.add_argument("--subject-limit", type=int, default=32)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--window", type=int, default=124)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--infer-batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-density", type=float, default=0.05)
    p.add_argument("--hp-limit", type=int, default=None)
    p.add_argument("--hp-id", type=int, default=None)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--active-quantile", type=float, default=0.94)
    p.add_argument("--child-active-quantile", type=float, default=None)
    p.add_argument("--young-active-quantile", type=float, default=None)
    p.add_argument("--self-quantile", type=float, default=0.90)
    p.add_argument("--prior-blend", type=float, default=0.85)
    p.add_argument("--metric-floor-quantile", type=float, default=0.05)
    p.set_defaults(use_reference_baselines=True)
    p.add_argument("--no-reference-baselines", dest="use_reference_baselines", action="store_false")
    p.add_argument("--tau-start", type=float, default=1.0)
    p.add_argument("--tau-min", type=float, default=0.35)
    p.add_argument("--device", default=None)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
