# ENGR 418 — Facial Age Group Estimation (CNN)

Small CNN (`SmallAgeCNN`, ~94k parameters) that classifies face images into age groups.
Datasets: Face-Age-10K and UTKFace. Each is tested with 4 and 8 age groups.
Seed 418, 200x200 RGB, batch 64, Adam (lr 1e-3), 10 epochs.

## Key results (test set)

| Dataset | Classes | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Face-Age-10K | 4 | 53.16% | 48.07% |
| Face-Age-10K | 8 | 32.90% | 24.70% |
| UTKFace | 4 | 57.18% | 49.56% |
| UTKFace | 8 | 35.82% | 26.30% |

Note: the original 4-class runs already used class-weighted cross-entropy.

Controlled class-weighting experiment (UTKFace, 4 classes, one run per condition):

| Loss | Accuracy | Macro-F1 |
|---|---:|---:|
| Standard CE | 60.75% | 46.00% |
| Weighted CE | 54.71% | 49.32% |

Weighting improved 41-65 (F1 29.62 -> 37.84) and 66+ (27.41 -> 38.20) but lowered 21-40 (74.36 -> 65.45).
Limitation: single run per condition; DataLoader shuffle order was not fixed with a separate generator.

## Files

**Notebooks**
- `Dataset_Inspection_Face_Age_10K.ipynb` - dataset inspection and split creation (Face-Age-10K)
- `CNN_Baseline_Face_Age_10K.ipynb` - Face-Age-10K, 8 classes
- `CNN_Baseline_Face_Age_4Groups.ipynb` - Face-Age-10K, 4 classes
- `UTKFace_8Groups_Experiment.ipynb` - UTKFace, 8 classes
- `UTKFace_4Groups_Experiment_Completed.ipynb` - UTKFace, 4 classes
- `ENGR418_Model_Performance_Analysis.ipynb` - analysis, figures, and the Standard vs Weighted CE experiment

**Data splits** (`*_split_indices.csv`) - fixed train/val/test splits; reused, never regenerated.

**Checkpoints** (`*.pt`)
- `face_age_cnn_best.pt` (Face-Age-10K 8-class), `face_age_cnn_4groups_best.pt` (Face-Age-10K 4-class)
- `utkface_8groups_cnn_best.pt`, `utkface_4groups_cnn_best.pt`
- `UTKFace_4Groups_Standard_CE_best.pt`, `UTKFace_4Groups_Weighted_CE_best.pt`

**Results** - `*_test_results.json`, `*_test_report.csv`, `*_confusion_matrix.csv`, `*_training_history.csv`, `*_history.csv`, `UTKFace_4Groups_ClassWeighting_Comparison.csv`, `UTKFace_4Groups_PerClass_F1_Comparison.csv`

**Figures** - `Class_Distribution_Comparison.png`, `Training_Samples_vs_F1_Score.png`, `UTKFace_Confusion_Matrix_Comparison.png`, `UTKFace_Normalized_Confusion_Matrix.png`, `Actual_vs_Predicted_Distribution.png`, `ClassWeighting_Confusion_Matrix.png`

## Not included (to keep the folder small)

- **UTKFace images (23,708 files).** Download UTKFace and extract it to `UTKFace_source/extracted/UTKFace/`.
- **Face-Age-10K images.** Loaded from Hugging Face: `prithivMLmods/Face-Age-10K`.
- Jupyter checkpoint folders.

The UTKFace split CSVs store absolute paths from the original machine
(`C:\Users\ASUS\Downloads\...`). On another computer, rebuild the `path` column from `file_name`
and your local image folder before running any UTKFace notebook.
