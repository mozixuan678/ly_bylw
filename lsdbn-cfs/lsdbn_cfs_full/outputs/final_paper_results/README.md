# Final Paper-Style LSDBN-CFS Results

This folder contains the final paper-style outputs generated from real experiment runs.

## Final Strict Result

- Model: `LSDBN-CFS + BCFS-124 + focused ExtraTrees`
- Selected features: `124`
- Subject-level split: train / feature-selection / test subjects are separated
- Final threshold selected on feature-selection set: `0.4716939640826283`
- Test accuracy: `84.62%`
- Balanced accuracy: `85.64%`
- Confusion matrix: `[[785, 300], [10, 920]]`

## Tables

Located in `tables/`, each table is available as both `.csv` and `.xlsx`.

- `threshold_curve`: threshold vs feature-selection/test accuracy
- `selected_124_features`: 124 selected directed effective connections
- `model_performance`: final model comparison table
- `ablation_study`: module/optimization ablation table
- `roi_frequency`: source/target/total ROI frequency table
- `final_summary`: one-row final result summary

## Figures

Located in `figures/`.

- `threshold_curve.svg`: final classifier threshold curve
- `connectome_124.svg`: circular directed connectome of 124 selected ECs
- `roi_frequency.svg`: ROI frequency distribution
- `model_performance.svg`: final performance comparison
- `confusion_matrix.svg`: strict test confusion matrix

## Reproduce

From `lsdbn_cfs_full`:

```powershell
..\lsdbn\Scripts\python.exe .\build_final_paper_outputs.py --features ..\feature_vectors.npy --labels ..\expanded_labels.npy --data ..\ADdata.npy --selected .\outputs\precomputed_lsdbn_h128\selected_features.csv --out .\outputs\final_paper_results
D:\anaconda3\python.exe .\tools\export_xlsx_from_csv.py --tables-dir .\outputs\final_paper_results\tables
```

`Window-random split probe` and `Diagnostic test-best BCFS-124` are diagnostic outputs only and should not be reported as strict subject-level evaluation results.

