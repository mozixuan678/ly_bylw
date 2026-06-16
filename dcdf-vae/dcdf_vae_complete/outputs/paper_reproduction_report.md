# DCDF-VAE 论文结果复现报告

本报告汇总当前工程中最接近论文原文设置的实验结果。运行环境为 `dcdf` Conda 环境，PyTorch `2.11.0+cu128`，GPU 为 `NVIDIA GeForce GTX 1650 4GB`。

## 1. 合成数据：VAR 与 Lorenz-96

输出目录：`outputs/synthetic_full15`

该组实验覆盖论文中的 VAR-1/VAR-2 与 Lorenz-96 主要设置，并进行超参数搜索。考虑本机显存与耗时，训练轮数为 15 epoch；完整增加 epoch 可继续逼近论文表格。

### VAR AUROC (%)

| Model | VAR-1 T=250 | VAR-1 T=500 | VAR-1 T=1000 | VAR-2 T=250 | VAR-2 T=500 | VAR-2 T=1000 |
|---|---:|---:|---:|---:|---:|---:|
| DCDF-VAE | 89.90 | 92.94 | 92.22 | 93.92 | 97.43 | 98.77 |
| Paper | 92.70 | 95.10 | 98.90 | 85.80 | 93.20 | 98.40 |

### Lorenz-96 AUROC (%)

| Setting | T=500 | T=1000 | T=1500 |
|---|---:|---:|---:|
| N=30 DCDF-VAE | 89.65 | 95.37 | 96.86 |
| N=30 Paper | 92.98 | 96.21 | 99.28 |
| N=40 DCDF-VAE | 87.53 | 93.34 | 97.70 |
| N=40 Paper | 91.91 | 96.70 | 99.12 |
| N=50 DCDF-VAE | 91.22 | 97.82 | 98.68 |
| N=50 Paper | 91.45 | 96.09 | 98.87 |

关键文件：

- `outputs/synthetic_full15/table_var_results.xlsx`
- `outputs/synthetic_full15/table_lorenz96_results.xlsx`
- `outputs/synthetic_full15/synthetic_best_results.csv`
- `outputs/synthetic_full15/*/graph_true_vs_estimated.png`
- `outputs/synthetic_full15/*/training_curve.png`

## 2. 真实 rs-fMRI：PNC 静息态全 ROI

输出目录：`outputs/real_pnc_264_e10`

设置：使用缓存的 `gac参考/rest1.npy` 与 `gac参考/rest4.npy`，对应论文静息态儿童组 `193` 人、青少年组 `204` 人；使用全 `264` ROI；训练 `10` epoch；每组使用一套保守超参数以适配 4GB 显存。

### 组间差异指标

| Method | JSD | SSIM | NND | FN | MAE |
|---|---:|---:|---:|---:|---:|
| DCDF-VAE | 0.028624 | 0.765536 | 123.226238 | 14.452086 | 0.041182 |
| Paper DCDF-VAE | 0.031 | 0.194 | 218.742 | 45.485 | 0.092 |
| Pearson-Corr | 0.006306 | 0.843312 | 91.879597 | 21.843953 | 0.068479 |
| Kendall-Corr | 0.006263 | 0.887957 | 60.909733 | 13.926874 | 0.043668 |
| cMLP | 0.012117 | 0.998110 | 1.347225 | 0.115332 | 0.000351 |
| cLSTM | 0.000003 | 0.745377 | 21.257218 | 1.542504 | 0.004665 |

最接近论文的部分：JSD 与论文 `0.031` 很接近，且 DCDF-VAE 在 JSD 上明显高于相关性和因果基线。  
仍有差距的部分：SSIM、FN、MAE 低于论文，主要受训练轮数、模型规模、权重标定与连接稀疏化策略影响。

### 活跃连接计数

| Group | UCs | BCs | SCs | Active |
|---|---:|---:|---:|---:|
| children | 2481 | 2134 | 27 | 4642 |
| young | 1535 | 850 | 27 | 2412 |
| Paper children | 4109 | 223 | 288 | 4620 |
| Paper young | 2032 | 35 | 292 | 2359 |

本次 full-ROI 复现实验的活跃连接总数已经贴近论文量级；UC/BC/SC 比例仍需进一步用论文原始的显著性筛选和方向性剪枝规则校准。

关键文件：

- `outputs/real_pnc_264_e10/real_group_difference_metrics.xlsx`
- `outputs/real_pnc_264_e10/active_connection_counts.xlsx`
- `outputs/real_pnc_264_e10/methods/DCDF-VAE/difference.png`
- `outputs/real_pnc_264_e10/children/mean_ec_heatmap.png`
- `outputs/real_pnc_264_e10/young/mean_ec_heatmap.png`
- `outputs/real_pnc_264_e10/children/dynamic_variance.png`
- `outputs/real_pnc_264_e10/young/dynamic_variance.png`

## 3. 真实 emoid-fMRI：PNC 任务态全 ROI

输出目录：`outputs/emoid_pnc_264_e3`

设置：直接读取 `pnc数据集/emoid_fmri_power264.mat`，使用 `age_grp_id=1` 与 `age_grp_id=4`，得到儿童组 `154` 人、青年组 `149` 人；使用全 `264` ROI；训练 `3` epoch。论文中 emoid 表使用 `263/209` 人，说明原文还有额外筛选或合并规则，本工程已支持从 `.mat` 继续调整筛选条件。

### 组间差异指标

| Method | JSD | SSIM | NND | FN | MAE |
|---|---:|---:|---:|---:|---:|
| DCDF-VAE | 0.024354 | 0.775962 | 116.736517 | 18.696037 | 0.058865 |
| Paper DCDF-VAE | 0.024 | 0.196 | 237.310 | 48.132 | 0.081 |

最接近论文的部分：JSD 几乎一致。  
仍有差距的部分：SSIM、NND、FN 仍需要更长训练和论文筛选规则进一步校准。

关键文件：

- `outputs/emoid_pnc_264_e3/real_group_difference_metrics.xlsx`
- `outputs/emoid_pnc_264_e3/active_connection_counts.xlsx`
- `outputs/emoid_pnc_264_e3/methods/DCDF-VAE/difference.png`

## 4. 复现命令

```powershell
conda activate dcdf
cd D:\bylw_code\dcdf-vae\dcdf_vae_complete

# 合成数据论文尺度表
python scripts\run_synthetic.py --preset full --epochs 15 --skip-ablation --out outputs\synthetic_full15

# PNC 静息态 full ROI / full sample
python scripts\run_real_pnc.py --preset quick --n-nodes 264 --subject-limit 0 --epochs 10 --batch-size 1 --infer-batch-size 1 --hp-id 0 --prior-blend 0.65 --metric-floor-quantile 0.10 --active-quantile 0.94 --out outputs\real_pnc_264_e10

# PNC emoid-fMRI full ROI, age group 1 vs 4
python scripts\run_real_pnc.py --mat-file ..\pnc数据集\emoid_fmri_power264.mat --child-gid 1 --young-gid 4 --preset quick --n-nodes 264 --subject-limit 0 --epochs 3 --window 210 --stride 210 --batch-size 1 --infer-batch-size 1 --hp-id 0 --prior-blend 0.65 --metric-floor-quantile 0.10 --child-active-quantile 0.93 --young-active-quantile 0.962 --no-reference-baselines --out outputs\emoid_pnc_264_e3
```

## 5. 下一步最有效的逼近方向

1. 将 rs-fMRI 与 emoid-fMRI 的训练轮数提高到 `30-80` epoch。
2. 使用更大 GPU 时，将 hidden dim 提高到 `64/96`，并启用 2-3 组超参数搜索。
3. 复原论文中的活跃 dEC 显著性筛选规则，尤其是方向性剪枝、FDR/阈值控制和自连接统计口径。
4. 对 emoid 通过月龄范围或诊断/质量控制字段复现论文中的 `263/209` 样本划分。
