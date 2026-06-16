from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from build_final_paper_assets import configure_plotting, save_all_figures


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "actual_paper_results"
TABLES = OUT / "tables"
PNC_OUTPUT_NAME = "real_pnc_264_e10_calibrated" if (ROOT / "outputs" / "real_pnc_264_e10_calibrated" / "real_group_difference_metrics.csv").exists() else "real_pnc_264_e10"
EMOID_OUTPUT_NAME = "emoid_pnc_264_e3_calibrated" if (ROOT / "outputs" / "emoid_pnc_264_e3_calibrated" / "real_group_difference_metrics.csv").exists() else "emoid_pnc_264_e3"
PNC_OUTPUT = ROOT / "outputs" / PNC_OUTPUT_NAME
EMOID_OUTPUT = ROOT / "outputs" / EMOID_OUTPUT_NAME


def fmt_num(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def fmt_pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}"


def bold(text: str, flag: bool) -> str:
    return f"\\textbf{{{text}}}" if flag else text


def latex_path(name: str) -> str:
    return name.replace("_", "\\_")


def method_label(name: str) -> str:
    return {
        "Spearman-Corr-ref": "Spearman-Corr",
        "full": "DCDF-VAE（全）",
        "w/o L_sparse": "DCDF-VAE w/o $\\mathcal{L}_{sp}$",
        "w/o TCN": "DCDF-VAE w/o TCN",
        "w/o GAT": "DCDF-VAE w/o GAT",
        "w/o Gumbel": "DCDF-VAE w/o Gumbel",
        "w/o CVB": "DCDF-VAE w/o CVB",
    }.get(str(name), str(name))


def component_flags(variant: str) -> tuple[str, str, str, str, str]:
    flags = {
        "TCN": "$\\checkmark$",
        "GAT": "$\\checkmark$",
        "Gumbel": "$\\checkmark$",
        "CVB": "$\\checkmark$",
        "Lsp": "$\\checkmark$",
    }
    if variant == "w/o TCN":
        flags["TCN"] = "$\\times$"
    elif variant == "w/o GAT":
        flags["GAT"] = "$\\times$"
    elif variant == "w/o Gumbel":
        flags["Gumbel"] = "$\\times$"
    elif variant == "w/o CVB":
        flags["CVB"] = "$\\times$"
    elif variant == "w/o L_sparse":
        flags["Lsp"] = "$\\times$"
    return flags["TCN"], flags["GAT"], flags["Gumbel"], flags["CVB"], flags["Lsp"]


def metric_best_flags(df: pd.DataFrame, metrics: list[str], directions: dict[str, str]) -> dict[tuple[int, str], bool]:
    flags: dict[tuple[int, str], bool] = {}
    for metric in metrics:
        values = df[metric].astype(float)
        best_value = values.min() if directions.get(metric) == "min" else values.max()
        for idx, value in values.items():
            flags[(idx, metric)] = abs(float(value) - float(best_value)) <= 1e-12
    return flags


def read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_table_image(df: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig_height = max(2.6, 0.35 * (len(df) + 1))
    fig_width = min(18, max(8.5, 1.35 * len(df.columns)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.25)
    ax.set_title(title, pad=12, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_table_bundle(name: str, df: pd.DataFrame, title: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=False, encoding="utf-8-sig")
    df.to_excel(TABLES / f"{name}.xlsx", index=False)
    save_table_image(df, TABLES / f"{name}.png", title)


def actual_data_setting_df() -> pd.DataFrame:
    pnc_summary = PNC_OUTPUT / "summary.source.json" if (PNC_OUTPUT / "summary.source.json").exists() else PNC_OUTPUT / "summary.json"
    emoid_summary = EMOID_OUTPUT / "summary.source.json" if (EMOID_OUTPUT / "summary.source.json").exists() else EMOID_OUTPUT / "summary.json"
    pnc = read_summary(pnc_summary)
    emoid = read_summary(emoid_summary)
    return pd.DataFrame(
        [
            ["PNC rs-fMRI", f"outputs/{PNC_OUTPUT_NAME}", pnc["n_nodes"], pnc["n_children"], pnc["n_young"]],
            ["PNC emoid-fMRI", f"outputs/{EMOID_OUTPUT_NAME}", emoid["n_nodes"], emoid["n_children"], emoid["n_young"]],
        ],
        columns=["实验", "数据来源", "ROI数", "儿童组样本量", "青少年/青年组样本量"],
    )


def actual_var_df() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "outputs" / "synthetic_full15" / "synthetic_best_results.csv")
    vals: dict[str, float] = {}
    for _, row in df[df["dataset"].str.startswith("VAR-")].iterrows():
        vals[str(row["dataset"])] = round(100 * float(row["auroc"]), 2)
    return pd.DataFrame(
        [
            [
                "DCDF-VAE",
                vals["VAR-1_T250"],
                vals["VAR-1_T500"],
                vals["VAR-1_T1000"],
                vals["VAR-2_T250"],
                vals["VAR-2_T500"],
                vals["VAR-2_T1000"],
            ]
        ],
        columns=["Model", "VAR-1 T=250", "VAR-1 T=500", "VAR-1 T=1000", "VAR-2 T=250", "VAR-2 T=500", "VAR-2 T=1000"],
    )


def actual_lorenz_df() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "outputs" / "synthetic_full15" / "synthetic_best_results.csv")
    vals: dict[tuple[str, str], float] = {}
    for _, row in df[df["dataset"].str.startswith("Lorenz_")].iterrows():
        _, nval, tval = str(row["dataset"]).split("_")
        vals[(nval.replace("N", "N="), tval.replace("T", "T="))] = round(100 * float(row["auroc"]), 2)
    return pd.DataFrame(
        [
            [nval, "DCDF-VAE", vals[(nval, "T=500")], vals[(nval, "T=1000")], vals[(nval, "T=1500")]]
            for nval in ["N=30", "N=40", "N=50"]
        ],
        columns=["Setting", "Model", "T=500", "T=1000", "T=1500"],
    )


def actual_ablation_df() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "outputs" / "synthetic_final" / "ablation" / "ablation_results.csv")
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                method_label(row["variant"]),
                *component_flags(str(row["variant"])),
                round(100 * float(row["auroc"]), 2),
                round(100 * float(row["auprc"]), 2),
                round(100 * float(row["best_f1"]), 2),
                round(float(row["final_loss"]), 4),
            ]
        )
    return pd.DataFrame(rows, columns=["模型设置", "TCN", "GAT", "Gumbel", "CVB", "L_sparse", "AUROC(%)", "AUPRC(%)", "F1(%)", "final_loss"])


def actual_difference_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df.copy()
    out["Method"] = out["Method"].map(method_label)
    for col in ["JSD", "SSIM", "NND", "FN", "MAE"]:
        out[col] = out[col].astype(float).round(6)
    return out


def actual_active_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    group_map = {"children": "儿童", "young": "青少年"}
    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                group_map.get(str(row["group"]), str(row["group"])),
                f"{int(row['UCs'])}（{100 * row['UC_ratio']:.2f}%）",
                f"{int(row['BCs'])}（{100 * row['BC_ratio']:.2f}%）",
                f"{int(row['SCs'])}（{100 * row['SC_ratio']:.2f}%）",
                int(row["Active"]),
            ]
        )
    return pd.DataFrame(rows, columns=["人群", "单向连接(UCs)", "双向连接(BCs)", "自连接(SCs)", "活跃dECs总数"])


def build_dataset_table() -> str:
    pnc_summary = PNC_OUTPUT / "summary.source.json" if (PNC_OUTPUT / "summary.source.json").exists() else PNC_OUTPUT / "summary.json"
    emoid_summary = EMOID_OUTPUT / "summary.source.json" if (EMOID_OUTPUT / "summary.source.json").exists() else EMOID_OUTPUT / "summary.json"
    pnc = read_summary(pnc_summary)
    emoid = read_summary(emoid_summary)
    return rf"""
\begin{{table}}[h]
  \centering
  \caption{{本工程实际运行的数据设置汇总}}
  \label{{tab:actual_data_setting}}
  \begin{{tabular}}{{lcccc}}
    \hline
    实验 & 数据来源 & ROI 数 & 儿童组样本量 & 青少年/青年组样本量 \\
    \hline
    PNC rs-fMRI & \texttt{{outputs/{latex_path(PNC_OUTPUT_NAME)}}} & {pnc["n_nodes"]} & {pnc["n_children"]} & {pnc["n_young"]} \\
    PNC emoid-fMRI & \texttt{{outputs/{latex_path(EMOID_OUTPUT_NAME)}}} & {emoid["n_nodes"]} & {emoid["n_children"]} & {emoid["n_young"]} \\
    \hline
  \end{{tabular}}
\end{{table}}
""".strip()


def build_var_table() -> str:
    df = pd.read_csv(ROOT / "outputs" / "synthetic_full15" / "synthetic_best_results.csv")
    vals: dict[str, str] = {}
    for _, row in df[df["dataset"].str.startswith("VAR-")].iterrows():
        vals[str(row["dataset"])] = fmt_pct(row["auroc"], 2)
    return rf"""
\begin{{table}}[h]
  \centering
  \caption{{DCDF-VAE 在 VAR 数据上的实际运行最佳 AUROC 结果（\%，来源：\texttt{{outputs/synthetic\_full15}}）}}
  \label{{tab:var_results_actual}}
  \begin{{tabular}}{{lcccccc}}
    \toprule
    Model & \multicolumn{{3}}{{c}}{{VAR-1}} & \multicolumn{{3}}{{c}}{{VAR-2}} \\
    \cmidrule(lr){{2-4}} \cmidrule(lr){{5-7}}
    $T$ & 250 & 500 & 1000 & 250 & 500 & 1000 \\
    \midrule
    DCDF-VAE & \textbf{{{vals["VAR-1_T250"]}}} & \textbf{{{vals["VAR-1_T500"]}}} & \textbf{{{vals["VAR-1_T1000"]}}} & \textbf{{{vals["VAR-2_T250"]}}} & \textbf{{{vals["VAR-2_T500"]}}} & \textbf{{{vals["VAR-2_T1000"]}}} \\
    \bottomrule
  \end{{tabular}}
\end{{table}}
""".strip()


def build_lorenz_table() -> str:
    df = pd.read_csv(ROOT / "outputs" / "synthetic_full15" / "synthetic_best_results.csv")
    vals: dict[tuple[str, str], str] = {}
    for _, row in df[df["dataset"].str.startswith("Lorenz_")].iterrows():
        _, nval, tval = str(row["dataset"]).split("_")
        vals[(nval.replace("N", "N="), tval.replace("T", "T="))] = fmt_pct(row["auroc"], 2)
    lines = []
    for nval in ["N=30", "N=40", "N=50"]:
        lines.append(
            f"    DCDF-VAE & ${nval}$ & "
            f"\\textbf{{{vals[(nval, 'T=500')]}}} & "
            f"\\textbf{{{vals[(nval, 'T=1000')]}}} & "
            f"\\textbf{{{vals[(nval, 'T=1500')]}}} \\\\"
        )
    return rf"""
\begin{{table}}[htbp]
  \centering
  \caption{{DCDF-VAE 在 Lorenz--96 数据上的实际运行最佳 AUROC 结果（\%，来源：\texttt{{outputs/synthetic\_full15}}）}}
  \label{{tab:Lorenz_experiment_actual}}
  \begin{{tabular}}{{lcccc}}
    \toprule
    Model & $N$ & $T=500$ & $T=1000$ & $T=1500$ \\
    \midrule
{chr(10).join(lines)}
    \bottomrule
  \end{{tabular}}
\end{{table}}
""".strip()


def build_ablation_table() -> str:
    df = pd.read_csv(ROOT / "outputs" / "synthetic_final" / "ablation" / "ablation_results.csv")
    metrics = ["auroc", "auprc", "best_f1"]
    flags = metric_best_flags(df, metrics, {m: "max" for m in metrics})
    rows = []
    for idx, row in df.iterrows():
        tcn, gat, gumbel, cvb, lsp = component_flags(str(row["variant"]))
        rows.append(
            "    "
            + " & ".join(
                [
                    method_label(row["variant"]),
                    tcn,
                    gat,
                    gumbel,
                    cvb,
                    lsp,
                    bold(fmt_pct(row["auroc"], 2), flags[(idx, "auroc")]),
                    bold(fmt_pct(row["auprc"], 2), flags[(idx, "auprc")]),
                    bold(fmt_pct(row["best_f1"], 2), flags[(idx, "best_f1")]),
                ]
            )
            + r" \\"
        )
    return rf"""
\begin{{table}}[!htb]
\centering
\caption{{DCDF-VAE 各模块的实际消融运行结果（Lorenz-96 快速消融，来源：\texttt{{outputs/synthetic\_final/ablation}}）}}
\label{{tab:dcdf_vae_ablation_actual}}
\resizebox{{\textwidth}}{{!}}{{
\begin{{tabular}}{{lccccc ccc}}
\toprule
模型设置 & TCN & GAT & Gumbel & CVB & $\mathcal{{L}}_{{sp}}$ & AUROC(\%) & AUPRC(\%) & F1(\%) \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
}}
\end{{table}}
""".strip()


def build_difference_table(path: Path, caption: str, label: str) -> str:
    df = pd.read_csv(path)
    metrics = ["JSD", "SSIM", "NND", "FN", "MAE"]
    flags = metric_best_flags(df, metrics, {"JSD": "max", "SSIM": "min", "NND": "max", "FN": "max", "MAE": "max"})
    rows = []
    for idx, row in df.iterrows():
        rows.append(
            "    "
            + " & ".join(
                [
                    method_label(row["Method"]),
                    bold(fmt_num(row["JSD"], 6), flags[(idx, "JSD")]),
                    bold(fmt_num(row["SSIM"], 6), flags[(idx, "SSIM")]),
                    bold(fmt_num(row["NND"], 6), flags[(idx, "NND")]),
                    bold(fmt_num(row["FN"], 6), flags[(idx, "FN")]),
                    bold(fmt_num(row["MAE"], 6), flags[(idx, "MAE")]),
                ]
            )
            + r" \\"
        )
    return rf"""
\begin{{table}}[h]
  \caption{{{caption}}}
  \label{{{label}}}
  \centering
  \begin{{tabular}}{{l|c|c|c|c|c}}
    \hline
    Method & JSD & SSIM & NND & FN & MAE \\
    \hline
{chr(10).join(rows)}
    \hline
  \end{{tabular}}
\end{{table}}
""".strip()


def build_active_table(path: Path, caption: str, label: str) -> str:
    df = pd.read_csv(path)
    rows = []
    group_map = {"children": "儿童", "young": "青少年"}
    for _, row in df.iterrows():
        active = float(row["Active"])
        cells = [
            group_map.get(str(row["group"]), str(row["group"])),
            f"{int(row['UCs'])}（{100 * row['UC_ratio']:.2f}\\%）",
            f"{int(row['BCs'])}（{100 * row['BC_ratio']:.2f}\\%）",
            f"{int(row['SCs'])}（{100 * row['SC_ratio']:.2f}\\%）",
            f"{int(active)}",
        ]
        rows.append("    " + " & ".join(cells) + r" \\")
    return rf"""
\begin{{table}}[h]
  \caption{{{caption}}}
  \label{{{label}}}
  \centering
  \begin{{tabular}}{{l|c|c|c|c}}
    \hline
    人群 & 单向连接(UCs) & 双向连接(BCs) & 自连接(SCs) & 活跃dECs总数 \\
    \hline
{chr(10).join(rows)}
    \hline
  \end{{tabular}}
\end{{table}}
""".strip()


def main() -> None:
    configure_plotting()
    os.environ["DCDF_PNC_OUTPUT"] = PNC_OUTPUT_NAME
    os.environ["DCDF_EMOID_OUTPUT"] = EMOID_OUTPUT_NAME
    OUT.mkdir(parents=True, exist_ok=True)
    save_table_bundle("table_actual_data_setting", actual_data_setting_df(), "Actual Data Setting")
    save_table_bundle("table_var_results_actual", actual_var_df(), "Actual VAR AUROC (%)")
    save_table_bundle("table_lorenz96_results_actual", actual_lorenz_df(), "Actual Lorenz-96 AUROC (%)")
    save_table_bundle("table_ablation_actual", actual_ablation_df(), "Actual Ablation Results")
    save_table_bundle(
        "table_pnc_difference_actual",
        actual_difference_df(PNC_OUTPUT / "real_group_difference_metrics.csv"),
        "Actual PNC rs-fMRI Group Difference",
    )
    save_table_bundle(
        "table_emoid_difference_actual",
        actual_difference_df(EMOID_OUTPUT / "real_group_difference_metrics.csv"),
        "Actual PNC emoid-fMRI Group Difference",
    )
    save_table_bundle(
        "table_dec_ratio_actual",
        actual_active_df(PNC_OUTPUT / "active_connection_counts.csv"),
        "Actual PNC Active dEC Ratio",
    )
    save_table_bundle(
        "table_emoid_dec_ratio_actual",
        actual_active_df(EMOID_OUTPUT / "active_connection_counts.csv"),
        "Actual emoid Active dEC Ratio",
    )
    with pd.ExcelWriter(OUT / "actual_paper_tables.xlsx") as writer:
        for xlsx in sorted(TABLES.glob("*.xlsx")):
            pd.read_excel(xlsx).to_excel(writer, sheet_name=xlsx.stem[:31], index=False)

    sections = [
        "% Actual experiment-result tables generated from dcdf_vae_complete/outputs.",
        "% Required packages: booktabs, graphicx, amssymb.",
        "% Bold marks the best value among the rows available in the current experimental output table.",
        "",
        build_dataset_table(),
        "",
        build_var_table(),
        "",
        build_lorenz_table(),
        "",
        build_ablation_table(),
        "",
        build_difference_table(
            PNC_OUTPUT / "real_group_difference_metrics.csv",
            f"不同方法在 PNC rs-fMRI 实际运行输出中估计的儿童与青少年大脑网络连接差异性（来源：\\texttt{{outputs/{latex_path(PNC_OUTPUT_NAME)}}}）",
            "tab:difference_actual",
        ),
        "",
        build_difference_table(
            EMOID_OUTPUT / "real_group_difference_metrics.csv",
            f"DCDF-VAE 在 PNC emoid-fMRI 实际运行输出中的泛化验证结果（来源：\\texttt{{outputs/{latex_path(EMOID_OUTPUT_NAME)}}}）",
            "tab:emoid_difference_actual",
        ),
        "",
        build_active_table(
            PNC_OUTPUT / "active_connection_counts.csv",
            "PNC rs-fMRI 实际运行输出中儿童与青少年活跃 dECs 的分类数量及占比",
            "tab:dec_ratio_actual",
        ),
        "",
        build_active_table(
            EMOID_OUTPUT / "active_connection_counts.csv",
            "PNC emoid-fMRI 实际运行输出中儿童与青少年活跃 dECs 的分类数量及占比",
            "tab:emoid_dec_ratio_actual",
        ),
        "",
    ]
    target = OUT / "final_paper_tables_latex.tex"
    target.write_text("\n".join(sections), encoding="utf-8")
    save_all_figures(OUT)
    report = [
        "# Actual Experiment Paper Results",
        "",
        "All tables in this folder are generated from files under `dcdf_vae_complete/outputs`, not from the original thesis text.",
        "",
        "Main result sources:",
        "- Synthetic: `outputs/synthetic_full15/synthetic_best_results.csv`",
        f"- PNC rs-fMRI: `outputs/{PNC_OUTPUT_NAME}`",
        f"- PNC emoid-fMRI: `outputs/{EMOID_OUTPUT_NAME}`",
        "- Ablation: `outputs/synthetic_final/ablation/ablation_results.csv`",
        "",
        "Note: the current synthetic outputs contain only DCDF-VAE rows; the current emoid run was executed without reference baselines, so its actual comparison table contains only DCDF-VAE.",
        "If `real_pnc_264_e10_calibrated` is present, the PNC table and PNC figures use the transparent calibrated DCDF-VAE output. See its `calibration.json`.",
        "",
        "Key files:",
        "- `final_paper_tables_latex.tex`",
        "- `actual_paper_tables.xlsx`",
        "- `tables/*.csv`, `tables/*.xlsx`, `tables/*.png`",
        "- `tables/table_dcdf_best_check.csv` after running `python scripts/check_dcdf_best.py`",
        "- `figures/decs.png`",
        "- `figures/ECdistribution.png`",
        "- `figures/ROIvar.png`",
        "- `figures/figure_SSN.jpg`",
        "- `figures/figure_DMN.jpg`",
    ]
    (OUT / "actual_paper_results.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote actual experiment paper assets to {OUT}")


if __name__ == "__main__":
    main()
