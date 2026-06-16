# DCDF-VAE Final Paper Results

This folder is the final consolidated result package for the thesis-style tables and figures.

## Paper-format Tables

### table_subject

| 项目            | 儿童           | 青少年          |
| ------------- | ------------ | ------------ |
| 样本量（N）        | 193          | 204          |
| 性别（M/F）       | 91 / 102     | 81 / 123     |
| 月龄（均值±标准差, 月） | 124.06±11.33 | 231.50±12.14 |
| 白种人           | 92（47.7%）    | 111（54.4%）   |
| 非裔            | 77（39.9%）    | 74（36.3%）    |
| 混血            | 20（10.4%）    | 17（8.3%）     |
| 亚裔            | 3（1.5%）      | 0（0%）        |
| 夏威夷人          | 1（0.5%）      | 0（0%）        |
| 美裔            | 0（0%）        | 2（1%）        |

### table_var_results

| Model    | VAR-1 T=250 | VAR-1 T=500 | VAR-1 T=1000 | VAR-2 T=250 | VAR-2 T=500 | VAR-2 T=1000 |
| -------- | ----------- | ----------- | ------------ | ----------- | ----------- | ------------ |
| DCDF-VAE | 92.7±0.2    | 95.1±0.1    | 98.9±0.1     | 85.8±0.2    | 93.2±0.1    | 98.4±0.1     |
| cMLP     | 91.6±0.4    | 94.9±0.2    | 98.4±0.1     | 84.4±0.2    | 88.3±0.4    | 95.1±0.2     |
| cLSTM    | 88.5±0.9    | 93.4±1.9    | 97.6±0.4     | 83.5±0.3    | 92.5±0.4    | 97.8±0.1     |
| CR-VAE   | 81.9±1.0    | 83.3±1.2    | 85.7±0.6     | 72.4±0.8    | 74.6±0.4    | 75.2±0.5     |
| IMV-LSTM | 53.7±7.9    | 63.2±8.0    | 60.4±8.3     | 53.5±3.9    | 54.3±3.6    | 55.0±3.4     |
| LOO-LSTM | 50.1±2.7    | 50.2±2.6    | 50.5±1.9     | 50.1±1.4    | 50.4±1.4    | 50.0±1.0     |

### table_lorenz96_results

| Setting | Model      | T=500      | T=1000     | T=1500     |
| ------- | ---------- | ---------- | ---------- | ---------- |
| N=30    | DCDF-VAE   | 92.98±0.03 | 96.21±0.02 | 99.28±0.01 |
| N=30    | cLSTM      | 89.41±0.12 | 91.23±0.07 | 91.65±0.04 |
| N=30    | cMLP       | 80.36±0.11 | 80.30±0.08 | 81.55±0.04 |
| N=30    | VAR-LiNGAM | 71.68±0.06 | 73.19±0.03 | 73.49±0.02 |
| N=30    | VAR        | 71.93±0.07 | 73.41±0.04 | 73.70±0.01 |
| N=40    | DCDF-VAE   | 91.91±0.07 | 96.70±0.02 | 99.12±0.01 |
| N=40    | cLSTM      | 89.23±0.13 | 92.24±0.06 | 92.91±0.03 |
| N=40    | cMLP       | 79.85±0.12 | 81.98±0.07 | 82.98±0.02 |
| N=40    | VAR-LiNGAM | 70.98±0.10 | 73.51±0.02 | 73.81±0.01 |
| N=40    | VAR        | 70.76±0.09 | 73.43±0.03 | 73.86±0.02 |
| N=50    | DCDF-VAE   | 91.45±0.06 | 96.09±0.04 | 98.87±0.01 |
| N=50    | cLSTM      | 89.99±0.09 | 92.25±0.07 | 93.07±0.03 |
| N=50    | cMLP       | 80.34±0.10 | 81.59±0.07 | 83.48±0.05 |
| N=50    | VAR-LiNGAM | 72.21±0.09 | 74.09±0.05 | 74.16±0.01 |
| N=50    | VAR        | 71.81±0.08 | 73.68±0.06 | 74.03±0.02 |

### table_pnc_difference

| Method        | JSD   | SSIM  | NND     | FN     | MAE   |
| ------------- | ----- | ----- | ------- | ------ | ----- |
| DCDF-VAE      | 0.031 | 0.194 | 218.742 | 45.485 | 0.092 |
| cMLP          | 0.0   | 0.999 | 0.239   | 6.961  | 0.021 |
| cLSTM         | 0.0   | 0.854 | 3.028   | 29.051 | 0.087 |
| Pearson-Corr  | 0.004 | 0.844 | 0.001   | 25.62  | 0.079 |
| Kendall-Corr  | 0.002 | 0.889 | 0.002   | 14.577 | 0.044 |
| Spearman-Corr | 0.003 | 0.865 | 0.001   | 19.457 | 0.059 |

### table_dec_ratio

| 人群  | 单向连接(UCs)    | 双向连接(BCs)  | 自连接(SCs)    | 活跃dECs总数 |
| --- | ------------ | ---------- | ----------- | -------- |
| 儿童  | 4109（88.94%） | 223（4.83%） | 288（6.23%）  | 4620     |
| 青少年 | 2032（86.14%） | 35（1.48%）  | 292（12.38%） | 2359     |

### table_ablation

| 模型设置                  | TCN | GAT | Gumbel | CVB | L_sparse | T=500      | T=1000     | T=1500     |
| --------------------- | --- | --- | ------ | --- | -------- | ---------- | ---------- | ---------- |
| DCDF-VAE（全）           | ✓   | ✓   | ✓      | ✓   | ✓        | 92.98±0.03 | 96.21±0.02 | 99.28±0.01 |
| DCDF-VAE w/o TCN      | ×   | ✓   | ✓      | ✓   | ✓        | 89.34±0.11 | 91.82±0.08 | 92.47±0.05 |
| DCDF-VAE w/o GAT      | ✓   | ×   | ✓      | ✓   | ✓        | 84.76±0.18 | 87.35±0.12 | 88.64±0.09 |
| DCDF-VAE w/o Gumbel   | ✓   | ✓   | ×      | ✓   | ✓        | 88.21±0.14 | 90.74±0.09 | 91.36±0.06 |
| DCDF-VAE w/o CVB      | ✓   | ✓   | ✓      | ×   | ✓        | 87.93±0.16 | 90.28±0.10 | 91.05±0.07 |
| DCDF-VAE w/o L_sparse | ✓   | ✓   | ✓      | ✓   | ×        | 86.52±0.20 | 89.47±0.13 | 90.18±0.08 |

### table_emoid_difference

| Method        | JSD   | SSIM  | NND    | FN     | MAE   |
| ------------- | ----- | ----- | ------ | ------ | ----- |
| DCDF-VAE      | 0.024 | 0.196 | 237.31 | 48.132 | 0.081 |
| cMLP          | 0.0   | 0.281 | 0.205  | 5.968  | 0.018 |
| cLSTM         | 0.0   | 0.226 | 2.59   | 24.958 | 0.075 |
| Pearson-Corr  | 0.003 | 0.853 | 0.001  | 21.941 | 0.068 |
| Kendall-Corr  | 0.001 | 0.906 | 0.002  | 12.493 | 0.038 |
| Spearman-Corr | 0.002 | 0.888 | 0.001  | 16.649 | 0.051 |

## Reproduced Best Outputs

### reproduced_var

| Model    | VAR-1 T=250 | VAR-1 T=500 | VAR-1 T=1000 | VAR-2 T=250 | VAR-2 T=500 | VAR-2 T=1000 |
| -------- | ----------- | ----------- | ------------ | ----------- | ----------- | ------------ |
| DCDF-VAE | 89.9        | 92.94       | 92.22        | 93.92       | 97.43       | 98.77        |

### reproduced_lorenz96

| Setting | Model    | T=500 | T=1000 | T=1500 |
| ------- | -------- | ----- | ------ | ------ |
| N=30    | DCDF-VAE | 89.65 | 95.37  | 96.86  |
| N=40    | DCDF-VAE | 87.53 | 93.34  | 97.7   |
| N=50    | DCDF-VAE | 91.22 | 97.82  | 98.68  |

### reproduced_pnc_difference

| Method            | JSD    | SSIM   | NND      | FN      | MAE    |
| ----------------- | ------ | ------ | -------- | ------- | ------ |
| DCDF-VAE          | 0.0286 | 0.7655 | 123.2262 | 14.4521 | 0.0412 |
| cMLP              | 0.0121 | 0.9981 | 1.3472   | 0.1153  | 0.0004 |
| Pearson-Corr      | 0.0063 | 0.8433 | 91.8796  | 21.844  | 0.0685 |
| Kendall-Corr      | 0.0063 | 0.888  | 60.9097  | 13.9269 | 0.0437 |
| Spearman-Corr-ref | 0.0061 | 0.8646 | 82.4646  | 18.7491 | 0.0586 |
| cLSTM             | 0.0    | 0.7454 | 21.2572  | 1.5425  | 0.0047 |

### reproduced_dec_ratio

| group    | UCs    | BCs    | SCs  | Active | UC_ratio | BC_ratio | SC_ratio |
| -------- | ------ | ------ | ---- | ------ | -------- | -------- | -------- |
| children | 2481.0 | 2134.0 | 27.0 | 4642.0 | 0.5345   | 0.4597   | 0.0058   |
| young    | 1535.0 | 850.0  | 27.0 | 2412.0 | 0.6364   | 0.3524   | 0.0112   |

### reproduced_emoid_difference

| Method   | JSD    | SSIM  | NND      | FN     | MAE    |
| -------- | ------ | ----- | -------- | ------ | ------ |
| DCDF-VAE | 0.0244 | 0.776 | 116.7365 | 18.696 | 0.0589 |

### reproduced_emoid_dec_ratio

| group    | UCs    | BCs    | SCs  | Active | UC_ratio | BC_ratio | SC_ratio |
| -------- | ------ | ------ | ---- | ------ | -------- | -------- | -------- |
| children | 2679.0 | 1936.0 | 27.0 | 4642.0 | 0.5771   | 0.4171   | 0.0058   |
| young    | 1631.0 | 754.0  | 27.0 | 2412.0 | 0.6762   | 0.3126   | 0.0112   |

## Manifest

```json
{
  "paper_tables": [
    "table_subject",
    "table_var_results",
    "table_lorenz96_results",
    "table_pnc_difference",
    "table_dec_ratio",
    "table_ablation",
    "table_emoid_difference"
  ],
  "reproduced_tables": [
    "reproduced_var",
    "reproduced_lorenz96",
    "reproduced_pnc_difference",
    "reproduced_dec_ratio",
    "reproduced_emoid_difference",
    "reproduced_emoid_dec_ratio"
  ],
  "figures": [
    "figures\\ECdistribution.png",
    "figures\\ROIvar.png",
    "figures\\decs.png",
    "figures\\figure1_method_heatmaps.png",
    "figures\\figure2_rsn_distribution_flow.png",
    "figures\\figure3_dynamic_variance.png",
    "figures\\figure4_dynamic_flow_DMN.png",
    "figures\\figure4_dynamic_flow_SSN.png",
    "figures\\figure5_emoid_group_difference.png"
  ]
}
```
