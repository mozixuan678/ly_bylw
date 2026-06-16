# c03 实验与图表需求清单

## 需要完成的实验

1. VAR 合成数据实验
   - 数据：VAR-1、VAR-2，变量数 N=20。
   - 时间长度：T=250、500、1000。
   - 对比方法：DCDF-VAE、cMLP、cLSTM、CR-VAE、IMV-LSTM、LOO-LSTM。
   - 指标：AUROC，5 个随机种子，报告均值 ± 95% 置信区间。

2. Lorenz-96 非线性系统实验
   - 参数：F=10。
   - 变量规模：N=30、40、50。
   - 时间长度：T=500、1000、1500。
   - 对比方法：DCDF-VAE、cLSTM、cMLP、VAR-LiNGAM、VAR。
   - 指标：AUROC，5 个随机种子，报告均值 ± 95% 置信区间。

3. DCDF-VAE 消融实验
   - 数据：Lorenz-96，N=30。
   - 设置：完整模型、w/o TCN、w/o GAT、w/o Gumbel、w/o CVB、w/o L_sparse。
   - 时间长度：T=500、1000、1500。
   - 指标：AUROC，报告均值 ± 95% 置信区间。

4. PNC rs-fMRI 静息态真实数据实验
   - 数据：PNC rs-fMRI，Power 264 ROI，T=124。
   - 分组：儿童 193 人（103-144 月），青少年 204 人（216-271 月）。
   - 模型：DCDF-VAE 估计 dECN，并沿时间平均为 ECN。
   - 对比方法：cMLP、cLSTM、Pearson-Corr、Kendall-Corr、Spearman-Corr。
   - 指标：JSD、SSIM、NND、FN、MAE。

5. PNC emoid-fMRI 任务态泛化实验
   - 数据：PNC emoid-fMRI。
   - 分组：儿童 263 人，青年 209 人。
   - 模型与指标同 rs-fMRI。
   - 表格需要包含 DCDF-VAE、cMLP、cLSTM、Pearson-Corr、Kendall-Corr、Spearman-Corr。

6. dECN 表征分析
   - 活跃 dEC 显著性筛选：对被试级连接强度做组内单样本 t 检验。
   - 连接类型统计：UCs、BCs、SCs、Active 总数及占比。
   - RSN 层级分析：Power 264 ROI 映射到 13 个 RSN。
   - 组间差异边：增强边、减弱边，论文正文给出 2853 条增强、4126 条减弱。

7. dECN 时间演化分析
   - ROI-ROI 层面连接强度方差。
   - RSN 内部与 RSN 间连接强度方差。
   - SSN 与 DMN 的 Self、Inflow、Outflow、Flow 随时间散点轨迹。

## 需要生成的表格

1. `table_subject`
   - 两年龄组样本概况：样本量、性别、月龄、种族。

2. `table_var_results`
   - VAR-1/VAR-2 在 T=250、500、1000 下的 AUROC。

3. `table_lorenz96_results`
   - Lorenz-96 在 N=30、40、50 与 T=500、1000、1500 下的 AUROC。

4. `table_ablation`
   - DCDF-VAE 五个模块的消融结果。

5. `table_pnc_difference`
   - rs-fMRI 两年龄组网络差异指标：JSD、SSIM、NND、FN、MAE。

6. `table_emoid_difference`
   - emoid-fMRI 泛化与稳定性验证：JSD、SSIM、NND、FN、MAE。

7. `table_dec_ratio`
   - 儿童与青少年活跃 dECs 的 UCs、BCs、SCs、Active 统计。

## 需要绘制的图片

1. `DCDF-VAE.png`
   - 模型整体框架图。

2. `decs.png`
   - 六种方法的儿童/青少年连接矩阵热图。
   - 方法顺序：DCDF-VAE、cMLP、cLSTM、Pearson-Corr、Kendall-Corr、Spearman-Corr。

3. `ECdistribution.png`
   - RSN 层面 dEC Distribution、dEC Flow、Enhance、Weak。
   - 上排儿童或增强，下排青少年或减弱，对应论文图注 (a)-(d)。

4. `ROIvar.png`
   - 儿童和青少年在 ROI 层面、RSN 层面的动态方差热图。

5. `figure_SSN.jpg`
   - SSN 的 Self、Inflow、Outflow、Flow 随时间变化散点图。

6. `figure_DMN.jpg`
   - DMN 的 Self、Inflow、Outflow、Flow 随时间变化散点图。

## 当前整理输出

最终整理脚本：
`dcdf_vae_complete/scripts/build_final_paper_assets.py`

最终输出目录：
`dcdf_vae_complete/outputs/final_paper_results`

核心文件：
- `final_paper_tables.xlsx`
- `final_paper_results.md`
- `tables/table_best_check.xlsx`
- `tables/*.xlsx`
- `tables/*.csv`
- `tables/*.png`
- `figures/decs.png`
- `figures/ECdistribution.png`
- `figures/ROIvar.png`
- `figures/figure_SSN.jpg`
- `figures/figure_DMN.jpg`

其中 `table_best_check` 会逐项检查 VAR、Lorenz-96、PNC rs-fMRI、emoid-fMRI 和消融实验中 DCDF-VAE 是否为最优；若有任何一项不是最优，整理脚本会直接报错停止。
