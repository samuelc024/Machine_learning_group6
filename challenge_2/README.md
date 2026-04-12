# Challenge 2 — Semi-Supervised Learning on ACS PUMS 2022 (Group 6)

## What this is about
This folder contains the code and results for **ML Challenge 2**, focused on **semi-supervised learning (SSL)** over U.S. socioeconomic microdata.

- **Dataset:** American Community Survey (**ACS**) PUMS 2022 (U.S. Census Bureau), person-level data.
- **Goal (multi-class):** predict the income level in **4 categories**.
- **Goal (binary):** predict whether a person’s income is **below vs. at/above the median** (balanced 50/50 split).
- **Models (supervised baselines):** Logistic Regression (LR), Random Forest (RF).
- **SSL methods:** Self-Training (pseudo-labeling), Label Spreading.

The scripts are written as **reproducible experiment runners** (multiple random seeds + stratified cross-validation) and they automatically download the dataset if it is not present locally.

## Contents
- `main_experiment.py` — **4-class** income classification (multiclass)
- `main_experiment_binary.py` — **binary** income classification (below/above median)
- `experiments.json` — experiment definitions (model type, SSL method, thresholds, seeds, CV folds, etc.)
- `pyproject.toml` — Python dependencies and project metadata
- `results_final/` — **saved multiclass results** (already generated)
- `results_binary/` — **saved binary results** (already generated)
- `Challenge 2.pdf` — assignment statement (provided by the course)

## Experiments (what runs)
Both scripts load a list of experiments from `experiments.json`. Each experiment config controls:

- `base_model`: `lr` or `rf` (and in some configs `label_spreading`)
- `ssl_method`: `null` (pure supervised), `self_training`, or `label_spreading`
- `labeled_fraction`: fraction of the training fold treated as labeled (default used: **0.15**)
- `confidence_threshold`, `max_ssl_iter`, `max_pseudo_per_iter`: self-training parameters
- `seeds`: repeated runs with different random seeds (default used: **[42, 123, 7]**)
- `n_folds`: stratified CV folds (default used: **5**)
- `sample_size`: subsampling for speed (default used: **80,000**)

### Multiclass runner (`main_experiment.py`)
- Builds a 4-class target with these bins (PINCP in USD):
  - 0: < $25k
  - 1: $25k–$50k
  - 2: $50k–$100k
  - 3: > $100k
- Trains and evaluates each experiment using **StratifiedKFold CV** for each seed.
- Saves per-experiment artifacts under `--results-dir/<experiment_name>/`.

### Binary runner (`main_experiment_binary.py`)
- Builds a binary target using the **median** PINCP as threshold:
  - 0: income < median
  - 1: income ≥ median
- Uses the same CV + seeds structure.
- Saves per-experiment artifacts under `--results-dir/<experiment_name>/`.

## Outputs and where to find them
For each experiment, the runners typically produce:

- `results.csv` — per-seed, per-fold metrics (one row per evaluation)
- `confusion_matrix.png`
- `roc_curve.png` (OvR for multiclass; standard ROC for binary)
- `metrics_per_seed.png` — mean ± std bars per seed
- `ssl_history.png` — only for self-training (pseudo-label growth over iterations)
- `model_last_seed.pkl`, `scaler_last_seed.pkl` — last trained model/scaler snapshot (may not exist in older runs)

At the top level of the results directory, the multiclass runner can also write:

- `comparison_summary.csv` — mean/std aggregated across all folds and seeds
- `comparison_bar.png`, `comparison_acc_vs_f1.png`
- `threshold_analysis.csv`, `threshold_analysis.png` (unless `--skip-threshold-analysis`)

## Existing results in this repository
### Multiclass results (`results_final/`)
The file `results_final/comparison_summary.csv` summarizes the experiments that were executed and saved.

Computed from the existing `results.csv` files (mean ± std across 15 evals = 3 seeds × 5 folds):

- `baseline_lr`: accuracy **0.5348 ± 0.0030**, f1_macro **0.5044 ± 0.0030**, auc **0.7894 ± 0.0015**
- `baseline_rf`: accuracy **0.5527 ± 0.0027**, f1_macro **0.5295 ± 0.0028**, auc **0.7962 ± 0.0012**
- `ssl_self_training_lr`: accuracy **0.5049 ± 0.0038**, f1_macro **0.5033 ± 0.0034**, auc **0.7847 ± 0.0016**

Note: `results_final/ssl_self_training_rf/` currently contains plots but **does not include `results.csv`** in this workspace snapshot, so it is not part of the computed summary above.

### Binary results (`results_binary/`)
Computed from the existing `results.csv` files (mean ± std across 15 evals = 3 seeds × 5 folds):

- `baseline_lr`: accuracy **0.7727 ± 0.0020**, f1_binary **0.7761 ± 0.0021**, auc **0.8545 ± 0.0015**
- `baseline_rf`: accuracy **0.7783 ± 0.0015**, f1_binary **0.7801 ± 0.0020**, auc **0.8614 ± 0.0011**
- `ssl_self_training_lr`: accuracy **0.7713 ± 0.0024**, f1_binary **0.7799 ± 0.0021**, auc **0.8544 ± 0.0015**
- `ssl_self_training_rf`: accuracy **0.7778 ± 0.0016**, f1_binary **0.7811 ± 0.0020**, auc **0.8611 ± 0.0011**

## How to run
### 1) Create a virtual environment (recommended)
From the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
Install this challenge as an editable project:

```bash
pip install -e challenge_2
```

Or, if you are inside `challenge_2/`:

```bash
pip install -e .
```

### 3) Run multiclass experiments
From `challenge_2/`:

```bash
python main_experiment.py
```

Useful options:

```bash
python main_experiment.py --list
python main_experiment.py --experiment baseline_lr
python main_experiment.py --experiment baseline_lr baseline_rf
python main_experiment.py --results-dir results_final
python main_experiment.py --skip-eda
python main_experiment.py --skip-threshold-analysis
python main_experiment.py --sample-size 50000
python main_experiment.py --replot --results-dir results_final
```

### 4) Run binary experiments
From `challenge_2/`:

```bash
python main_experiment_binary.py
```

Useful options:

```bash
python main_experiment_binary.py --list
python main_experiment_binary.py --experiment baseline_rf
python main_experiment_binary.py --results-dir results_binary
python main_experiment_binary.py --skip-eda
python main_experiment_binary.py --skip-threshold-analysis
python main_experiment_binary.py --sample-size 50000
python main_experiment_binary.py --replot --results-dir results_binary
```

## Dependencies
Defined in `pyproject.toml`:

- Python **>= 3.10**
- `numpy`, `pandas`
- `scikit-learn`, `joblib`
- `matplotlib`, `seaborn`
- `requests`

## Notes / reproducibility
- First run may take time because it downloads `csv_pus.zip` (~150MB) from the Census website and extracts the needed CSV into `challenge_2/data/`.
- Results depend on `sample_size`, `seeds`, and CV settings in `experiments.json` (or overridden via CLI).
