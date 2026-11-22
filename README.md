# ENGR 418: Facial Age Group Estimation

A course project on estimating an **approximate age group from a face image**. The goal is not to predict a person's exact age, which is difficult from a single image, but to classify the image into a broad age interval.

## Project motivation and development

We began with the Face-Age-10K dataset and tested two classification settings: four age groups and eight age groups. The results were limited. We then asked whether the dataset was the main reason for the weak performance, so we repeated both settings with the larger UTKFace dataset. The larger dataset did not solve the problem: performance remained modest, especially when age was divided into eight groups.

This led us to focus on the model as well as the data. Our improvement direction is to increase the CNN's complexity and capacity, then assess whether a stronger model improves age-group recognition. The notebooks and checkpoints currently published here document the compact CNN baselines and an additional class-weighting comparison; they do not include a separate deeper-CNN run. The scores below therefore refer to the baseline experiments.

## Datasets and age groups

- **Face-Age-10K:** 9,165 images used in the reported experiments.
- **UTKFace:** 23,708 images used in the reported experiments.

Four-class labels are the same for both datasets: `01–20`, `21–40`, `41–65`, and `66+`.

The eight-class labels are `01–10`, `11–20`, `21–30`, `31–40`, `41–55`, `56–65`, `66–80`, and `80+` for Face-Age-10K. UTKFace uses `81+` for its final group; preserve this difference when comparing the datasets.

The image datasets are not included in this repository. Face-Age-10K is loaded from Hugging Face (`prithivMLmods/Face-Age-10K`). UTKFace images must be obtained separately.

## Baseline results

The table reports held-out test performance from the four original dataset/grouping experiments.

| Dataset | Age groups | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Face-Age-10K | 4 | 53.16% | 48.07% |
| Face-Age-10K | 8 | 32.90% | 24.70% |
| UTKFace | 4 | 57.18% | 49.56% |
| UTKFace | 8 | 35.82% | 26.30% |

Four groups performed better than eight in these runs. UTKFace had somewhat higher scores than Face-Age-10K, but changing to the larger dataset alone did not produce strong results.

### Additional class-weighting comparison

A separate UTKFace four-class experiment compared standard and weighted cross-entropy. It used one run per condition.

| Loss | Accuracy | Macro-F1 |
|---|---:|---:|
| Standard cross-entropy | 60.75% | 46.00% |
| Weighted cross-entropy | 54.71% | 49.32% |

Weighting raised the F1 score for the `66+` group from 27.41% to 38.20%, while reducing overall accuracy. This is a trade-off, not a uniform improvement. Because there was only one run per condition and the DataLoader shuffle order was not fixed with separate deterministic generators, interpret the comparison cautiously.

## Model and training setup

The archived baseline notebooks use `SmallAgeCNN`: three convolutional blocks with 32, 64, and 128 channels, ReLU activations, max pooling, adaptive average pooling, dropout, and a linear classification head. Images are resized to 200 × 200 RGB. The main training configuration uses seed 418, batch size 64, Adam with learning rate 0.001, and 10 epochs.

## Repository layout

```text
.
├── README.md
├── notebooks/       # Dataset inspection, training, and analysis notebooks
├── data/
│   └── splits/      # Fixed train/validation/test split index files
├── models/          # Saved model checkpoints
└── results/
    ├── tables/      # Metrics, reports, confusion matrices, and histories
    └── figures/     # Plots and confusion-matrix figures
```

## Using the notebooks

1. Clone the repository and install the packages used by the notebooks: `datasets`, `pandas`, `numpy`, `Pillow`, `matplotlib`, `scikit-learn`, `torch`, `torchvision`, and `jupyterlab`.
2. Start Jupyter from the repository root so the relative paths to `data/`, `models/`, and `results/` resolve correctly.
3. Open a notebook in `notebooks/` and run its cells in order. Training time depends on available hardware; the notebooks use CUDA when available.
4. Before running UTKFace notebooks, download the UTKFace images and update the `path` column in the relevant split CSV under `data/splits/` to match their location on your computer. The archived CSVs contain paths from the original machine.

The fixed split files are provided so the experiments can reuse the same train, validation, and test assignments. The notebooks report validation metrics during model selection and reserve the test split for final evaluation.

## Limitations

The datasets and labels are uneven across age groups, and the models often confuse neighboring intervals. Results come from limited runs and should not be treated as evidence of reliable real-world age estimation. The project studies approximate age-group classification for coursework, not identity verification or consequential decisions about people.
