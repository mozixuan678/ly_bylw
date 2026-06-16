# LSDBN-CFS Complete Experiment Code

This folder contains a clean, runnable implementation of the LSDBN-CFS workflow:

- ADdata loading and subject-level train / feature-selection / test split
- sliding-window Gaussian transfer entropy dynamic effective connectivity
- log-sum sparse RBM and stacked LSDBN representation learning
- Relief, Inf-FS, ECFS and JCFS feature scoring
- backward causal feature selection (BCFS)
- model comparison, ablation, threshold scan, selected edge export and ROI frequency export

## Quick Run

The quick preset runs the full pipeline on the real `ADdata.npy` data with a small network so the code can be verified on CPU:

```powershell
py .\run_all.py --data ..\ADdata.npy --preset quick --out .\outputs\quick_run
```

If the local Python cannot unpickle `ADdata.npy` because of a pandas version mismatch, the loader will use `..\ADdata_arrays.npz` automatically. That compatibility cache can be regenerated with:

```powershell
D:\anaconda3\python.exe .\tools\convert_addata.py --data ..\ADdata.npy --out ..\ADdata_arrays.npz
```

## Paper-Scale Run

The paper preset keeps the architecture and key hyperparameters described in the chapter. This is CPU-expensive:

```powershell
py .\run_all.py --data ..\ADdata.npy --preset paper --out .\outputs\paper_run
```

Useful options:

- `--repeats 5`: repeated subject splits for mean/std
- `--epochs 50`: override RBM epochs
- `--input-prefilter-k 8010`: keep all directed EC features after Relief prefilter
- `--methods all`: run every comparison method

## Main Outputs

The output directory contains:

- `dynamic_ec_cache_w25_s5_l1.npz`: cached dEC features
- `model_performance.csv`: model comparison table
- `ablation_study.csv`: ablation table
- `threshold_curve.csv`: kappa threshold scan
- `selected_features.csv`: selected feature IDs and AAL-90 directed edges
- `roi_frequency.csv`: source, target and total frequency per ROI
- `run_summary.json`: configuration and key metrics

