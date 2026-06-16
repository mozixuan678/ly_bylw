# Actual Experiment Paper Results

All tables in this folder are generated from files under `dcdf_vae_complete/outputs`, not from the original thesis text.

Main result sources:
- Synthetic: `outputs/synthetic_full15/synthetic_best_results.csv`
- PNC rs-fMRI: `outputs/real_pnc_264_e10_calibrated`
- PNC emoid-fMRI: `outputs/emoid_pnc_264_e3`
- Ablation: `outputs/synthetic_final/ablation/ablation_results.csv`

Note: the current synthetic outputs contain only DCDF-VAE rows; the current emoid run was executed without reference baselines, so its actual comparison table contains only DCDF-VAE.
If `real_pnc_264_e10_calibrated` is present, the PNC table and PNC figures use the transparent calibrated DCDF-VAE output. See its `calibration.json`.

Key files:
- `final_paper_tables_latex.tex`
- `actual_paper_tables.xlsx`
- `tables/*.csv`, `tables/*.xlsx`, `tables/*.png`
- `figures/decs.png`
- `figures/ECdistribution.png`
- `figures/ROIvar.png`
- `figures/figure_SSN.jpg`
- `figures/figure_DMN.jpg`
