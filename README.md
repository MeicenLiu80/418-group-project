# ENGR 418: Facial Age Group Estimation

A course project on estimating an **approximate age group from a face image**. The goal is not to predict a person's exact age, which is difficult from a single image, but to classify the image into a broad age interval.

## Project motivation and development

We began with the Face-Age-10K dataset and tested two classification settings: four age groups and eight age groups. The results were limited. We then asked whether the dataset was the main reason for the weak performance, so we repeated both settings with the larger UTKFace dataset. The larger dataset did not solve the problem: performance remained modest, especially when age was divided into eight groups.

This led us to focus on the model as well as the data. Our improvement direction is to increase the CNN's complexity and capacity, then assess whether a stronger model improves age-group recognition. The notebooks and checkpoints currently published here document the compact CNN baselines and an additional class-weighting comparison; they do not include a separate deeper-CNN run. The scores below therefore refer to the baseline experiments.

### Questions guiding the experiments

The project can be described through three practical questions:

1. How well can a CNN classify a face into broad age intervals rather than estimate an exact age?
2. How do the results change when the task uses four intervals instead of eight, and when the experiments use the larger UTKFace dataset?
3. Does class weighting change the balance between overall accuracy and performance on less represented age groups?

The experiments suggest that coarser age intervals are easier for this baseline, and that using more images by itself does not remove the difficulty. The class-weighting comparison also illustrates that a metric averaged across classes can improve while accuracy falls. These are observations from the recorded runs, not claims that dataset size or class imbalance alone caused the outcomes.

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

The repository contains fixed train/validation/test split index files. The published experiment records use 6,415 / 1,375 / 1,375 examples for Face-Age-10K and 16,594 / 3,557 / 3,557 for UTKFace, respectively. The notebooks use validation results during model selection and report final performance on the held-out test split. Report the grouping definitions alongside every result: the eight-class final boundary differs slightly between datasets.

The supplemental weighting experiment changes the cross-entropy loss using class weights calculated from the training labels. It compares one standard-loss run with one weighted-loss run on UTKFace's four-class task. This is useful as an exploratory imbalance analysis, but the single run per condition is not enough to establish a robust causal effect.

### Metrics and interpretation

- **Accuracy** is the share of test images assigned to the correct age group. It gives a simple overall measure but can be dominated by common groups.
- **Macro-F1** computes F1 separately for each group and averages the group scores equally. It makes weak performance on a smaller group more visible.
- **Per-class F1 and confusion matrices** show which age intervals are difficult and which neighboring groups are commonly confused. Use the saved reports and matrices under `results/tables/` and `results/figures/` rather than inferring class-level outcomes from accuracy alone.

When discussing the 4- and 8-group comparison, note that the number and width of labels change the task difficulty. When comparing datasets, note that both dataset composition and the eight-group boundary definitions differ; these runs are not a controlled test of dataset size alone.

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

## Notes for writing the project report

The course asks for a 4–6 page, double-column academic paper with at least Introduction, Related Work, Methods, Results, Conclusion, and References. The following outline maps the archived work to those sections; it is a guide for writing, not a substitute for checking the actual notebooks and results.

| Required section | Project-specific content to cover |
|---|---|
| **Introduction** | Explain the practical goal: predict a broad age interval from a face image, not an exact age. Motivate the four- versus eight-group settings and summarize the progression from Face-Age-10K to UTKFace. State the questions listed above. |
| **Related Work** | Summarize prior CNN-based facial age estimation and age-group classification. Introduce the datasets using their original dataset sources, and explain how this project differs in task formulation (discrete age intervals). Cite the sources actually consulted; do not use this README as the only scholarly reference. |
| **Methods** | Describe data sources and exclusions, the exact age-to-group mappings, image resizing and RGB input, fixed data splits, `SmallAgeCNN`, optimizer, learning rate, batch size, epochs, software/hardware if recorded, and evaluation metrics. Describe the class-weighting experiment as a separate comparison. |
| **Results** | Present the four baseline test results in the table above. Add a compact comparison table or figure and use per-class F1/confusion matrices to explain errors. Report the class-weighting numbers separately and call out its accuracy–macro-F1 trade-off. Include the number of runs and avoid implying statistical significance. |
| **Conclusion** | Summarize that the four-group task performed better in these runs, the larger dataset alone did not yield strong results, and performance remains limited. Describe increased CNN complexity as a proposed next improvement unless a separate deeper-model experiment and its results are added to the evidence. |
| **References** | Cite the original Face-Age-10K and UTKFace sources, relevant peer-reviewed work on facial age estimation/classification, and any software or methods sources required by the chosen citation style. Verify bibliographic details from the original publications. |

For reproducibility, the report should make clear which numbers come from the original baseline runs and which come from the additional weighted-loss comparison. It should also acknowledge the limitations of one run per condition for the weighting comparison, class imbalance, possible dataset differences, and the modest performance. Do not describe the deeper-CNN direction as a measured improvement: no such run is included in this repository at present.

## Limitations

The datasets and labels are uneven across age groups, and the models often confuse neighboring intervals. Results come from limited runs and should not be treated as evidence of reliable real-world age estimation. The project studies approximate age-group classification for coursework, not identity verification or consequential decisions about people.

Dataset images are not redistributed here. The archived UTKFace split files include paths from the original machine and need local path updates before those notebooks can be rerun. The recorded experiments also do not include repeated seeds or uncertainty estimates, so small score differences should be interpreted cautiously.
