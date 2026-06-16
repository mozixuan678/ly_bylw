# DCDF-VAE Complete Experiment Workbench

This folder contains a reproducible implementation of the thesis DCDF-VAE idea:

- TCN temporal encoder
- directed GAT-style edge scorer
- Gumbel-Sigmoid dynamic graph sampling
- VAE latent variables
- causal verification bottleneck (graph-gated latent propagation)
- sparse and temporal regularization
- synthetic VAR / Lorenz-96 experiments
- PNC rs-fMRI group experiments and visualizations

## Run

Use the prepared environment:

```powershell
conda activate dcdf
cd D:\bylw_code\dcdf-vae\dcdf_vae_complete
```

Quick run:

```powershell
python scripts\run_all.py --preset quick
```

Synthetic only:

```powershell
python scripts\run_synthetic.py --preset quick --epochs 25 --verbose
```

PNC real data only:

```powershell
python scripts\run_real_pnc.py --preset quick --n-nodes 48 --subject-limit 32 --epochs 12 --verbose
```

Full paper-scale run:

```powershell
python scripts\run_all.py --preset full --verbose
```

The full run uses more subjects, more nodes, longer sequences, and a larger
hyperparameter grid. It is much slower on a 4GB GPU.

## Outputs

Results are written under `outputs/`:

- CSV/XLSX tables
- `summary.json`
- learned EC matrices as `.npy`
- heatmaps, difference maps, training curves, and AUROC/JSD bar charts
