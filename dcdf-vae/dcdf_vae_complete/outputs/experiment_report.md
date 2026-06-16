# DCDF-VAE 实验报告

运行环境：`conda activate dcdf`，PyTorch `2.11.0+cu128`，GPU `NVIDIA GeForce GTX 1650`。

## 1. 合成数据实验

论文尺度合成实验输出目录：`outputs/synthetic_full15`

### VAR AUROC (%)

| Model | VAR-1 T=250 | VAR-1 T=500 | VAR-1 T=1000 | VAR-2 T=250 | VAR-2 T=500 | VAR-2 T=1000 |
|---|---:|---:|---:|---:|---:|---:|
| DCDF-VAE | 89.90 | 92.94 | 92.22 | 93.92 | 97.43 | 98.77 |

### Lorenz-96 AUROC (%)

| Setting | Model | T=500 | T=1000 | T=1500 |
|---|---|---:|---:|---:|
| N=30 | DCDF-VAE | 89.65 | 95.37 | 96.86 |
| N=40 | DCDF-VAE | 87.53 | 93.34 | 97.70 |
| N=50 | DCDF-VAE | 91.22 | 97.82 | 98.68 |

主要文件：

- `outputs/synthetic_full15/table_var_results.xlsx`
- `outputs/synthetic_full15/table_lorenz96_results.xlsx`
- `outputs/synthetic_full15/synthetic_best_results.csv`
- 每个数据集子目录下有 `graph_true_vs_estimated.png`、`training_curve.png`、`best_model.pt`

## 2. 快速消融实验

输出目录：`outputs/synthetic_final/ablation`

该快速消融使用 Lorenz-96 `N=20, T=500` 和 12 epoch。短训练下原始门控矩阵较不稳定，因此消融结果主要用于检查模块开关与流水线是否可运行；论文式消融复现建议提高 epoch 后重跑。

## 3. 真实 PNC rs-fMRI 实验

输出目录：`outputs/real_pnc_final`

实际运行设置：儿童组/青少年组各 32 名被试，选取 48 个高方差 ROI，12 epoch，使用模型门控与时滞一致性先验进行稀疏校准。

### 组间差异指标

| Method | JSD | SSIM | NND | FN | MAE |
|---|---:|---:|---:|---:|---:|
| DCDF-VAE | 0.026917 | 0.775280 | 20.499796 | 4.246265 | 0.069310 |
| cMLP | 0.011656 | 0.998078 | 0.108382 | 0.021154 | 0.000352 |
| Spearman-Corr | 0.007832 | 0.908046 | 13.276697 | 3.889303 | 0.067437 |
| Pearson-Corr | 0.007757 | 0.898747 | 14.290331 | 4.343069 | 0.076275 |
| Kendall-Corr | 0.006813 | 0.952196 | 7.544845 | 2.341362 | 0.039470 |
| cLSTM | 0.000003 | 0.740574 | 1.646525 | 0.284316 | 0.004785 |

### 活跃连接计数

| Group | UCs | BCs | SCs | Active | UC ratio | BC ratio | SC ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| children | 45 | 46 | 5 | 96 | 0.4688 | 0.4792 | 0.0521 |
| young | 79 | 12 | 5 | 96 | 0.8229 | 0.1250 | 0.0521 |

主要文件：

- `outputs/real_pnc_final/real_group_difference_metrics.xlsx`
- `outputs/real_pnc_final/active_connection_counts.xlsx`
- `outputs/real_pnc_final/methods/DCDF-VAE/difference.png`
- `outputs/real_pnc_final/children/mean_ec_heatmap.png`
- `outputs/real_pnc_final/young/mean_ec_heatmap.png`
- `outputs/real_pnc_final/children/dynamic_variance.png`
- `outputs/real_pnc_final/young/dynamic_variance.png`

## 4. 复现命令

```powershell
conda activate dcdf
cd D:\bylw_code\dcdf-vae\dcdf_vae_complete

# 快速全流程
python scripts\run_all.py --preset quick

# 论文尺度合成实验
python scripts\run_synthetic.py --preset full --epochs 15 --skip-ablation --out outputs\synthetic_full15

# 真实 PNC 快速实验
python scripts\run_real_pnc.py --preset quick --n-nodes 48 --subject-limit 32 --epochs 12 --out outputs\real_pnc_final
```
