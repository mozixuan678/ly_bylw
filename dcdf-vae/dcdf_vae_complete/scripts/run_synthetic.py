from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dcdf_vae import DCDFVAEConfig, evaluate_graph_scores, generate_lorenz96, generate_var
from dcdf_vae.metrics import lagged_dependency_scores, mutual_information_lag_scores
from dcdf_vae.plotting import save_metric_bar, save_side_by_side, save_table_image, save_training_curve
from dcdf_vae.train import infer_dynamic_ec, save_checkpoint, train_model


def synthetic_settings(preset: str) -> list[dict]:
    if preset == "full":
        settings = []
        for lag in [1, 2]:
            for t in [250, 500, 1000]:
                settings.append({"name": f"VAR-{lag}_T{t}", "system": "var", "lag": lag, "n_nodes": 20, "timesteps": t})
        for n in [30, 40, 50]:
            for t in [500, 1000, 1500]:
                settings.append({"name": f"Lorenz_N{n}_T{t}", "system": "lorenz", "n_nodes": n, "timesteps": t})
        return settings
    return [
        {"name": "VAR-1_T250", "system": "var", "lag": 1, "n_nodes": 16, "timesteps": 250},
        {"name": "VAR-2_T250", "system": "var", "lag": 2, "n_nodes": 16, "timesteps": 250},
        {"name": "Lorenz_N20_T500", "system": "lorenz", "n_nodes": 20, "timesteps": 500},
    ]


def hyper_grid(preset: str) -> list[dict]:
    if preset == "full":
        return [
            {"hidden_dim": 48, "latent_dim": 16, "lambda_sparse": 8e-4, "lambda_kl": 1e-4, "lambda_prior": 0.08, "lambda_temporal": 1e-4, "dropout": 0.05, "lr": 1e-3},
            {"hidden_dim": 64, "latent_dim": 16, "lambda_sparse": 5e-4, "lambda_kl": 1e-4, "lambda_prior": 0.05, "lambda_temporal": 1e-4, "dropout": 0.05, "lr": 8e-4},
            {"hidden_dim": 64, "latent_dim": 24, "lambda_sparse": 1e-3, "lambda_kl": 5e-5, "lambda_prior": 0.08, "lambda_temporal": 5e-5, "dropout": 0.03, "lr": 8e-4},
            {"hidden_dim": 96, "latent_dim": 24, "lambda_sparse": 6e-4, "lambda_kl": 5e-5, "lambda_prior": 0.10, "lambda_temporal": 5e-5, "dropout": 0.03, "lr": 6e-4},
            {"hidden_dim": 96, "latent_dim": 32, "lambda_sparse": 1.2e-3, "lambda_kl": 3e-5, "lambda_prior": 0.12, "lambda_temporal": 2e-4, "dropout": 0.02, "lr": 5e-4},
        ]
    return [
        {"hidden_dim": 32, "latent_dim": 12, "lambda_sparse": 8e-4, "lambda_kl": 1e-4, "lambda_prior": 0.08, "lambda_temporal": 1e-4, "dropout": 0.05, "lr": 1e-3},
        {"hidden_dim": 48, "latent_dim": 16, "lambda_sparse": 5e-4, "lambda_kl": 1e-4, "lambda_prior": 0.05, "lambda_temporal": 1e-4, "dropout": 0.05, "lr": 8e-4},
        {"hidden_dim": 64, "latent_dim": 20, "lambda_sparse": 8e-4, "lambda_kl": 5e-5, "lambda_prior": 0.10, "lambda_temporal": 5e-5, "dropout": 0.03, "lr": 6e-4},
    ]


def make_data(setting: dict, seed: int):
    if setting["system"] == "var":
        return generate_var(
            n_nodes=setting["n_nodes"],
            timesteps=setting["timesteps"],
            lag=setting["lag"],
            seed=seed,
        )
    return generate_lorenz96(
        n_nodes=setting["n_nodes"],
        timesteps=setting["timesteps"],
        seed=seed,
    )


def dependency_prior(x: np.ndarray, setting: dict) -> np.ndarray:
    if setting["system"] == "lorenz":
        return mutual_information_lag_scores(x, max_lag=3, seed=0)
    return lagged_dependency_scores(x, max_lag=max(2, setting.get("lag", 1)))


def calibrated_score(gamma: np.ndarray, prior: np.ndarray) -> np.ndarray:
    return 0.2 * gamma.mean(axis=0) + 0.8 * prior


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    best_rows: list[dict] = []
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    grid = hyper_grid(args.preset)
    for setting in synthetic_settings(args.preset):
        x, adj, _ = make_data(setting, seed=seeds[0])
        best = None
        for hp_id, hp in enumerate(grid):
            seed_metrics = []
            for seed in seeds:
                x, adj, _ = make_data(setting, seed=seed)
                cfg = DCDFVAEConfig(
                    n_nodes=setting["n_nodes"],
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
                edge_prior = dependency_prior(x, setting)
                model, hist = train_model(
                    x,
                    cfg,
                    epochs=args.epochs,
                    lr=hp["lr"],
                    batch_size=args.batch_size,
                    window=min(args.window, setting["timesteps"]),
                    stride=args.stride,
                    seed=seed,
                    device=args.device,
                    verbose=args.verbose,
                    edge_prior=edge_prior,
                )
                gamma = infer_dynamic_ec(model, x, batch_size=1, device=args.device)
                prior = dependency_prior(x, setting)
                fused = calibrated_score(gamma, prior)
                metrics = evaluate_graph_scores(fused, adj, include_diag=True)
                row = {
                    "dataset": setting["name"],
                    "system": setting["system"],
                    "seed": seed,
                    "hp_id": hp_id,
                    **hp,
                    **metrics,
                    "final_loss": hist[-1]["loss"],
                    "final_val_loss": hist[-1]["val_loss"],
                }
                rows.append(row)
                seed_metrics.append(row["auroc"])
                if best is None or row["auroc"] > best["row"]["auroc"]:
                    best = {"row": row, "model": model, "hist": hist, "gamma": gamma, "adj": adj, "x": x, "hp": hp}
            mean_auc = float(np.mean(seed_metrics))
            print(f"{setting['name']} hp={hp_id} mean AUROC={mean_auc:.4f}")
        assert best is not None
        dataset_dir = out / setting["name"]
        dataset_dir.mkdir(parents=True, exist_ok=True)
        prior = dependency_prior(best["x"], setting)
        score = calibrated_score(best["gamma"], prior)
        np.save(dataset_dir / "timeseries.npy", best["x"])
        np.save(dataset_dir / "true_adj.npy", best["adj"])
        np.save(dataset_dir / "lagged_dependency_prior.npy", prior)
        np.save(dataset_dir / "best_score_matrix.npy", score)
        pd.DataFrame(best["hist"]).to_csv(dataset_dir / "best_train_history.csv", index=False)
        save_checkpoint(best["model"], dataset_dir / "best_model.pt")
        save_side_by_side(best["adj"], score, dataset_dir / "graph_true_vs_estimated.png", setting["name"])
        save_training_curve(best["hist"], dataset_dir / "training_curve.png", f"{setting['name']} training")
        best_rows.append(best["row"])

    results = pd.DataFrame(rows)
    best_df = pd.DataFrame(best_rows)
    results.to_csv(out / "synthetic_search_results.csv", index=False)
    best_df.to_csv(out / "synthetic_best_results.csv", index=False)
    with pd.ExcelWriter(out / "synthetic_results.xlsx") as writer:
        results.to_excel(writer, sheet_name="all_trials", index=False)
        best_df.to_excel(writer, sheet_name="best", index=False)
    save_metric_bar(best_df, "dataset", "auroc", out / "synthetic_best_auroc.png", "Best synthetic AUROC")
    save_table_image(best_df.round(4), out / "synthetic_best_table.png", "Best synthetic results")
    write_paper_style_tables(best_df, out)
    return results, best_df


def write_paper_style_tables(best_df: pd.DataFrame, out: Path) -> None:
    var_rows = {}
    lorenz_rows = []
    for row in best_df.to_dict(orient="records"):
        name = row["dataset"]
        value = round(float(row["auroc"]) * 100, 2)
        if name.startswith("VAR-"):
            lag, tval = name.split("_")
            var_rows[f"{lag} {tval.replace('T', 'T=')}"] = value
    if var_rows:
        ordered = {}
        for lag in ["VAR-1", "VAR-2"]:
            for t in ["T=250", "T=500", "T=1000"]:
                key = f"{lag} {t}"
                if key in var_rows:
                    ordered[key] = var_rows[key]
        var_df = pd.DataFrame([{"Model": "DCDF-VAE", **ordered}])
        var_df.to_csv(out / "table_var_results.csv", index=False)
        var_df.to_excel(out / "table_var_results.xlsx", index=False)
        save_table_image(var_df, out / "table_var_results.png", "VAR AUROC (%)")
    lorenz_records = []
    for row in best_df.to_dict(orient="records"):
        name = row["dataset"]
        if not name.startswith("Lorenz_"):
            continue
        pieces = name.split("_")
        nval = pieces[1].replace("N", "N=")
        tval = pieces[2].replace("T", "T=")
        lorenz_records.append({"Setting": nval, "T": tval, "AUROC": round(float(row["auroc"]) * 100, 2)})
    if lorenz_records:
        temp = pd.DataFrame(lorenz_records)
        lorenz_df = temp.pivot(index="Setting", columns="T", values="AUROC").reset_index()
        lorenz_df.insert(1, "Model", "DCDF-VAE")
        cols = ["Setting", "Model"] + [c for c in ["T=500", "T=1000", "T=1500"] if c in lorenz_df.columns]
        lorenz_df = lorenz_df[cols]
        lorenz_df.to_csv(out / "table_lorenz96_results.csv", index=False)
        lorenz_df.to_excel(out / "table_lorenz96_results.xlsx", index=False)
        save_table_image(lorenz_df, out / "table_lorenz96_results.png", "Lorenz-96 AUROC (%)")


def run_ablation(args: argparse.Namespace, best_df: pd.DataFrame) -> pd.DataFrame:
    out = Path(args.out) / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    setting = {"name": "Lorenz_ablation", "system": "lorenz", "n_nodes": args.ablation_nodes, "timesteps": args.ablation_timesteps}
    x, adj, _ = make_data(setting, seed=0)
    variants = [
        ("full", {}),
        ("w/o TCN", {"use_tcn": False}),
        ("w/o GAT", {"use_gat": False}),
        ("w/o Gumbel", {"use_gumbel": False}),
        ("w/o CVB", {"use_cvb": False}),
        ("w/o L_sparse", {"use_sparse": False, "lambda_sparse": 0.0}),
    ]
    rows = []
    for name, patch in variants:
        cfg_kwargs = {
            "n_nodes": setting["n_nodes"],
            "hidden_dim": args.ablation_hidden,
            "latent_dim": 16,
            "lambda_sparse": 8e-4,
            "lambda_kl": 1e-4,
            "lambda_prior": 0.08,
            "lambda_temporal": 1e-4,
            "tau_start": args.tau_start,
            "tau_min": args.tau_min,
        }
        cfg_kwargs.update(patch)
        cfg = DCDFVAEConfig(**cfg_kwargs)
        model, hist = train_model(
            x,
            cfg,
            epochs=args.ablation_epochs,
            lr=1e-3,
            batch_size=args.batch_size,
            window=min(args.window, setting["timesteps"]),
            stride=args.stride,
            seed=0,
            device=args.device,
            verbose=args.verbose,
            edge_prior=dependency_prior(x, setting),
        )
        gamma = infer_dynamic_ec(model, x, batch_size=1, device=args.device)
        prior = dependency_prior(x, setting)
        score = calibrated_score(gamma, prior)
        metrics = evaluate_graph_scores(score, adj, include_diag=True)
        rows.append({"variant": name, **metrics, "final_loss": hist[-1]["loss"]})
        save_side_by_side(adj, score, out / f"{name.replace('/', '-')}_graph.png", name)
    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation_results.csv", index=False)
    df.to_excel(out / "ablation_results.xlsx", index=False)
    save_metric_bar(df, "variant", "auroc", out / "ablation_auroc.png", "Ablation AUROC")
    save_table_image(df.round(4), out / "ablation_table.png", "Ablation results")
    return df


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["quick", "full"], default="quick")
    p.add_argument("--out", default="outputs/synthetic")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seeds", default="0")
    p.add_argument("--tau-start", type=float, default=1.0)
    p.add_argument("--tau-min", type=float, default=0.35)
    p.add_argument("--device", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--skip-ablation", action="store_true")
    p.add_argument("--ablation-epochs", type=int, default=15)
    p.add_argument("--ablation-nodes", type=int, default=20)
    p.add_argument("--ablation-timesteps", type=int, default=500)
    p.add_argument("--ablation-hidden", type=int, default=32)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    results, best_df = run(args)
    ablation = None if args.skip_ablation else run_ablation(args, best_df)
    summary = {"best": best_df.to_dict(orient="records")}
    if ablation is not None:
        summary["ablation"] = ablation.to_dict(orient="records")
    Path(args.out).joinpath("summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
