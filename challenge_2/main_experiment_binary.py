"""
ML Challenge 2 â€” Semi-Supervised Learning
Domain: Economics & Socioeconomic Data
Task: Predict income level (BINARY: above / below the mean) â€” Binary Classification
Dataset: American Community Survey (ACS) PUMS 2022 â€” U.S. Census Bureau

The threshold is computed as the mean of PINCP across all adults with income > 0
in the loaded sample, so it adapts to whatever sample size you use.

Baselines: Logistic Regression, Random Forest (supervised)
SSL Methods: Self-Training (pseudo-labeling), Label Spreading

Usage:
    python main_experiment_binary.py                                      # Run all
    python main_experiment_binary.py --experiment baseline_lr             # One experiment
    python main_experiment_binary.py --list                               # List experiments
    python main_experiment_binary.py --replot --results-dir results_bin   # Regenerate plots
"""

import argparse
import json
import os
import sys
import warnings
import zipfile
import io
from copy import deepcopy

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading

warnings.filterwarnings("ignore")

# ============================================================
# CONSTANTS
# ============================================================
DATA_DIR = "data"
DATA_URL = (
    "https://www2.census.gov/programs-surveys/acs/data/pums/2022/1-Year/csv_pus.zip"
)
LOCAL_FILE = os.path.join(DATA_DIR, "psam_pusa.csv")

FEATURE_COLS = [
    "AGEP",   # Age
    "SCHL",   # Educational attainment
    "SEX",    # Sex
    "WKHP",   # Hours worked per week
    "ESR",    # Employment status
    "COW",    # Class of worker
    "OCCP",   # Occupation code
    "INDP",   # Industry code
    "POBP",   # Place of birth
    "DIS",    # Disability
    "RAC1P",  # Race
    "HISP",   # Hispanic origin
    "ST",     # State (FIPS)
]
TARGET_COL = "PINCP"   # Total person income
TEST_FRACTION = 0.20
N_CLASSES = 2

# Binary labels - threshold computed at runtime from data mean
INCOME_LABELS = ["Bajo (< mediana)", "Alto (>= mediana)"]


# ============================================================
# 1. DATA DOWNLOAD & LOAD
# ============================================================
def download_acs_data() -> pd.DataFrame:
    """Download ACS PUMS 2022 person-level data from U.S. Census Bureau."""
    os.makedirs(DATA_DIR, exist_ok=True)
    all_cols = FEATURE_COLS + [TARGET_COL]

    if os.path.exists(LOCAL_FILE):
        print(f"[INFO] Local file found: {LOCAL_FILE}")
        return pd.read_csv(LOCAL_FILE, usecols=all_cols, low_memory=False)

    print("[INFO] Downloading ACS PUMS 2022 (~150 MB)... this may take a few minutes.")
    resp = requests.get(DATA_URL, stream=True, timeout=600)
    resp.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    z.extractall(DATA_DIR)
    print("[INFO] Download complete.")
    return pd.read_csv(LOCAL_FILE, usecols=all_cols, low_memory=False)


# ============================================================
# 2. DATA EXPLORATION
# ============================================================
def explore_data(df: pd.DataFrame, results_dir: str) -> None:
    """Generate and save EDA plots."""
    print("\n" + "=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nIncome statistics:\n{df[TARGET_COL].describe()}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    valid_income = df[TARGET_COL].dropna()
    valid_income = valid_income[valid_income > 0]

    # Income distribution with median line
    axes[0].hist(np.log1p(valid_income), bins=50, color="steelblue", edgecolor="white")
    median_val = valid_income.median()
    mean_val   = valid_income.mean()
    axes[0].axvline(np.log1p(median_val), color="red", linestyle="--", lw=2,
                    label=f"Mediana = ${median_val:,.0f} (threshold)")
    axes[0].axvline(np.log1p(mean_val), color="orange", linestyle=":", lw=1.5,
                    label=f"Media   = ${mean_val:,.0f}")
    axes[0].set_title("Log(Income) -- threshold = mediana (50/50 split)")
    axes[0].set_xlabel("log(PINCP + 1)")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=8)

    # Age distribution
    axes[1].hist(df["AGEP"].dropna(), bins=50, color="#e67e22", edgecolor="white")
    axes[1].set_title("Age Distribution")
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Count")

    # Median income by education level
    edu = df.groupby("SCHL")[TARGET_COL].median().dropna().reset_index()
    axes[2].bar(edu["SCHL"].astype(str), edu[TARGET_COL], color="steelblue")
    axes[2].set_title("Median Income by Education (SCHL)")
    axes[2].set_xlabel("Education Level")
    axes[2].set_ylabel("Median Income ($)")
    axes[2].tick_params(axis="x", labelsize=6, rotation=90)

    plt.tight_layout()
    path = os.path.join(results_dir, "01_eda_exploration.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] EDA plot saved: {path}")


# ============================================================
# 3. PREPROCESSING & FEATURE ENGINEERING
# ============================================================

_OCCP_GROUPS = {
    range(0, 500): 0,
    range(500, 1000): 1,
    range(1000, 1600): 2,
    range(1600, 2000): 3,
    range(2000, 2100): 4,
    range(2100, 2600): 5,
    range(2600, 3000): 6,
    range(3000, 3600): 7,
    range(3600, 4000): 8,
    range(4000, 4200): 9,
    range(4200, 4300): 10,
    range(4300, 4700): 11,
    range(4700, 5000): 12,
    range(5000, 6000): 13,
    range(6000, 7000): 14,
    range(7000, 8000): 15,
    range(8000, 9000): 16,
    range(9000, 10000): 17,
    range(9800, 9850): 18,
}


def _map_occp_group(code):
    if pd.isna(code):
        return 19
    code = int(code)
    for rng, group in _OCCP_GROUPS.items():
        if code in rng:
            return group
    return 19


def _map_indp_group(code):
    if pd.isna(code):
        return 13
    code = int(code)
    if code < 300: return 0
    if code < 600: return 1
    if code < 4000: return 2
    if code < 4600: return 3
    if code < 5800: return 4
    if code < 6400: return 5
    if code < 6800: return 6
    if code < 7200: return 7
    if code < 7800: return 8
    if code < 8200: return 9
    if code < 8500: return 10
    if code < 8700: return 11
    if code < 9300: return 12
    return 13


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features from raw ACS PUMS columns."""
    out = pd.DataFrame()

    out["AGEP"] = df["AGEP"].fillna(df["AGEP"].median())
    out["WKHP"] = df["WKHP"].fillna(0)
    out["SCHL"] = df["SCHL"].fillna(df["SCHL"].median())

    out["AGE_SQ"] = out["AGEP"] ** 2
    out["LOG_WKHP"] = np.log1p(out["WKHP"])
    out["AGE_x_WKHP"] = out["AGEP"] * out["WKHP"]
    out["AGE_x_SCHL"] = out["AGEP"] * out["SCHL"]
    out["WKHP_x_SCHL"] = out["WKHP"] * out["SCHL"]

    out["SEX"] = df["SEX"].fillna(1).astype(int)
    out["DIS"] = df["DIS"].fillna(2).astype(int)
    out["ESR"] = df["ESR"].fillna(6).astype(int)
    out["COW"] = df["COW"].fillna(0).astype(int)
    out["RAC1P"] = df["RAC1P"].fillna(1).astype(int)

    out["IS_HISPANIC"] = (df["HISP"].fillna(1) > 1).astype(int)
    out["BORN_US"] = (df["POBP"].fillna(999) <= 56).astype(int)

    out["OCCP_GROUP"] = df["OCCP"].apply(_map_occp_group).astype(int)
    out["INDP_GROUP"] = df["INDP"].apply(_map_indp_group).astype(int)

    schl = out["SCHL"]
    out["EDU_NO_HS"] = (schl < 16).astype(int)
    out["EDU_HS"] = ((schl >= 16) & (schl <= 19)).astype(int)
    out["EDU_SOME_COLLEGE"] = (schl == 20).astype(int)
    out["EDU_BACHELORS"] = (schl == 21).astype(int)
    out["EDU_GRADUATE"] = (schl >= 22).astype(int)

    out["AGE_18_25"] = ((out["AGEP"] >= 18) & (out["AGEP"] <= 25)).astype(int)
    out["AGE_26_35"] = ((out["AGEP"] >= 26) & (out["AGEP"] <= 35)).astype(int)
    out["AGE_36_50"] = ((out["AGEP"] >= 36) & (out["AGEP"] <= 50)).astype(int)
    out["AGE_51_65"] = ((out["AGEP"] >= 51) & (out["AGEP"] <= 65)).astype(int)
    out["AGE_OVER_65"] = (out["AGEP"] > 65).astype(int)

    out["FULL_TIME"] = (out["WKHP"] >= 35).astype(int)
    out["ST"] = df["ST"].fillna(0).astype(int)

    return out


def preprocess(df: pd.DataFrame, sample_size: int = 80_000) -> tuple:
    """Clean, filter, create BINARY income target (above/below median), engineer features.

    Uses the MEDIAN (not mean) as the split threshold so that classes are
    perfectly balanced 50/50.  The mean income is skewed by high earners,
    which pushes ~68% of workers below the mean and causes models to collapse
    to predicting the majority class.  The median is the true middle of the
    income distribution and yields a much better-behaved binary task.
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING & FEATURE ENGINEERING  [BINARY MODE]")
    print("=" * 60)

    df = df.copy()
    df = df[(df["AGEP"] >= 18) & (df[TARGET_COL].notna()) & (df[TARGET_COL] > 0)]
    print(f"[INFO] After filtering adults with income > 0: {len(df):,} rows")

    # Binary target: 0 = below median (Bajo), 1 = at/above median (Alto)
    # Median gives a balanced 50/50 split on the income distribution.
    income_median = df[TARGET_COL].median()
    income_mean   = df[TARGET_COL].mean()
    df["TARGET"] = (df[TARGET_COL] >= income_median).astype(int)

    print(f"[INFO] Income mean   : ${income_mean:,.0f}")
    print(f"[INFO] Income median : ${income_median:,.0f}  (used as binary threshold)")
    print(f"[INFO]   0 - Bajo (< mediana) : {(df['TARGET'] == 0).sum():,} rows "
          f"({(df['TARGET'] == 0).mean():.1%})")
    print(f"[INFO]   1 - Alto (>= mediana): {(df['TARGET'] == 1).sum():,} rows "
          f"({(df['TARGET'] == 1).mean():.1%})")

    # Subsample
    n = min(sample_size, len(df))
    df = df.sample(n=n, random_state=42).reset_index(drop=True)

    X_df = engineer_features(df)
    feature_names = list(X_df.columns)
    print(f"[INFO] Engineered features ({len(feature_names)}): {feature_names}")
    print(f"[INFO] Final dataset: ({len(X_df)}, {len(feature_names)})  |  Nulls: {X_df.isnull().sum().sum()}")

    X = X_df.values.astype(float)
    y = df["TARGET"].values
    return X, y, feature_names


# ============================================================
# 4. DATA PARTITIONING
# ============================================================
def make_split(X, y, labeled_fraction: float, seed: int):
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=seed, stratify=y,
    )
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
        X_pool, y_pool,
        train_size=labeled_fraction,
        random_state=seed,
        stratify=y_pool,
    )
    return X_labeled, X_unlabeled, X_test, y_labeled, y_unlabeled, y_test


# ============================================================
# 5. MODEL BUILDERS
# ============================================================
def build_model(config: dict, seed: int):
    model_type = config["base_model"]
    if model_type == "lr":
        return LogisticRegression(
            max_iter=config.get("lr_max_iter") or 2000,
            C=config.get("lr_C") or 1.0,
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
        )
    elif model_type == "rf":
        return RandomForestClassifier(
            n_estimators=config.get("rf_n_estimators") or 400,
            max_depth=config.get("rf_max_depth") or None,
            min_samples_leaf=config.get("rf_min_samples_leaf") or 2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown base_model: {model_type}")


# ============================================================
# 6. SELF-TRAINING
# ============================================================
def self_training(
    X_labeled, y_labeled, X_unlabeled,
    base_model, scaler,
    threshold: float = 0.85,
    max_iter: int = 10,
    max_per_iter: int = 500,
    verbose: bool = True,
):
    """Iterative self-training with confidence threshold."""
    X_l = X_labeled.copy()
    y_l = y_labeled.copy()
    X_u = X_unlabeled.copy()
    history = []

    for iteration in range(max_iter):
        if len(X_u) == 0:
            break

        X_l_sc = scaler.fit_transform(X_l)
        X_u_sc = scaler.transform(X_u)
        base_model.fit(X_l_sc, y_l)

        probs = base_model.predict_proba(X_u_sc)
        confidence = probs.max(axis=1)
        pseudo_labels = probs.argmax(axis=1)

        high_conf_idx = np.where(confidence >= threshold)[0]
        if len(high_conf_idx) > max_per_iter:
            top_idx = np.argsort(confidence[high_conf_idx])[-max_per_iter:]
            high_conf_idx = high_conf_idx[top_idx]

        if len(high_conf_idx) == 0:
            if verbose:
                print(f"    Iter {iteration + 1}: no high-confidence samples. Stopping.")
            break

        X_l = np.vstack([X_l, X_u[high_conf_idx]])
        y_l = np.concatenate([y_l, pseudo_labels[high_conf_idx]])
        X_u = np.delete(X_u, high_conf_idx, axis=0)

        history.append({
            "iteration": iteration + 1,
            "added": len(high_conf_idx),
            "total_labeled": len(y_l),
            "unlabeled_remaining": len(X_u),
            "mean_confidence": float(confidence[high_conf_idx].mean()),
        })

        if verbose:
            h = history[-1]
            print(
                f"    Iter {h['iteration']}: +{h['added']} pseudo-labels "
                f"(mean conf: {h['mean_confidence']:.3f}) | "
                f"Total labeled: {h['total_labeled']:,}"
            )

    X_l_sc = scaler.fit_transform(X_l)
    base_model.fit(X_l_sc, y_l)
    return base_model, scaler, pd.DataFrame(history)


# ============================================================
# 7. EVALUATION METRICS  (binary)
# ============================================================
def eval_metrics(y_true, y_pred, y_prob=None):
    """Compute binary classification metrics."""
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_binary": f1_score(y_true, y_pred, average="binary", zero_division=0),
    }
    if y_prob is not None:
        try:
            # Binary: use probability of positive class
            pos_prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            m["auc"] = roc_auc_score(y_true, pos_prob)
        except ValueError:
            m["auc"] = float("nan")
    return m


# ============================================================
# 8. SINGLE EVALUATION HELPER
# ============================================================
def _run_single_eval(config: dict, seed: int, X_l, y_l, X_u, X_t, y_t, verbose: bool = False):
    ssl_method = config.get("ssl_method")
    scaler = StandardScaler()

    if ssl_method is None:
        model = build_model(config, seed)
        needs_scaling = config["base_model"] == "lr"
        if needs_scaling:
            model.fit(scaler.fit_transform(X_l), y_l)
            X_t_sc = scaler.transform(X_t)
            y_pred = model.predict(X_t_sc)
            y_prob = model.predict_proba(X_t_sc) if hasattr(model, "predict_proba") else None
        else:
            model.fit(X_l, y_l)
            y_pred = model.predict(X_t)
            y_prob = model.predict_proba(X_t) if hasattr(model, "predict_proba") else None
        metrics = eval_metrics(y_t, y_pred, y_prob)
        history = pd.DataFrame()

    elif ssl_method == "self_training":
        model = build_model(config, seed)
        model, scaler, history = self_training(
            X_l, y_l, X_u,
            base_model=model,
            scaler=scaler,
            threshold=config.get("confidence_threshold", 0.85),
            max_iter=config.get("max_ssl_iter", 10),
            max_per_iter=config.get("max_pseudo_per_iter", 500),
            verbose=verbose,
        )
        X_t_sc = scaler.transform(X_t)
        y_pred = model.predict(X_t_sc)
        y_prob = model.predict_proba(X_t_sc) if hasattr(model, "predict_proba") else None
        metrics = eval_metrics(y_t, y_pred, y_prob)

    elif ssl_method == "label_spreading":
        X_combined = np.vstack([X_l, X_u])
        y_combined = np.concatenate([y_l, np.full(X_u.shape[0], -1)])
        scaler.fit(X_combined)
        ls = LabelSpreading(
            kernel="knn",
            n_neighbors=7,
            max_iter=config.get("max_ssl_iter", 30),
            alpha=0.2,
            n_jobs=-1,
        )
        ls.fit(scaler.transform(X_combined), y_combined)
        X_t_sc = scaler.transform(X_t)
        y_pred = ls.predict(X_t_sc)
        y_prob = ls.predict_proba(X_t_sc) if hasattr(ls, "predict_proba") else None
        metrics = eval_metrics(y_t, y_pred, y_prob)
        model = ls
        history = pd.DataFrame()

    else:
        raise ValueError(f"Unknown ssl_method: {ssl_method}")

    return metrics, history, y_pred, y_prob, model, scaler


# ============================================================
# 9. RUN ONE EXPERIMENT (with StratifiedKFold CV)
# ============================================================
def run_experiment(config: dict, X, y, results_dir: str):
    name = config["name"]
    seeds = config.get("seeds", [42, 123, 7])
    labeled_fraction = config.get("labeled_fraction", 0.15)
    n_folds = config.get("n_folds", 5)

    print("\n" + "#" * 70)
    print(f"  EXPERIMENT : {name}")
    print(f"  {config.get('description', '')}")
    print(f"  CV folds   : {n_folds}  |  Seeds: {seeds}")
    print("#" * 70)

    exp_dir = os.path.join(results_dir, name)
    os.makedirs(exp_dir, exist_ok=True)

    all_run_results = []
    all_histories = []
    last_fold_data = {}

    for seed in seeds:
        print(f"\n  === Seed {seed} ===")
        np.random.seed(seed)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold, X_val = X[train_idx], X[val_idx]
            y_train_fold, y_val = y[train_idx], y[val_idx]

            X_l, X_u, y_l, y_u = train_test_split(
                X_train_fold, y_train_fold,
                train_size=labeled_fraction,
                random_state=seed + fold_idx,
                stratify=y_train_fold,
            )

            verbose_fold = (fold_idx == 0 and seed == seeds[0])

            metrics, history, y_pred, y_prob, model, scaler = _run_single_eval(
                config, seed, X_l, y_l, X_u, X_val, y_val, verbose=verbose_fold,
            )

            metrics["seed"] = seed
            metrics["fold"] = fold_idx + 1
            metrics["experiment"] = name
            all_run_results.append(metrics)
            all_histories.append(history)

            last_fold_data = {
                "y_test": y_val, "y_pred": y_pred, "y_prob": y_prob,
                "model": model, "scaler": scaler,
            }

            auc_str = f"{metrics['auc']:.4f}" if not np.isnan(metrics.get("auc", float("nan"))) else "N/A"
            print(
                f"    Fold {fold_idx + 1}/{n_folds}: "
                f"Labeled={X_l.shape[0]:,} | Unlabeled={X_u.shape[0]:,} | Val={X_val.shape[0]:,} "
                f"=> Acc={metrics['accuracy']:.4f} | F1m={metrics['f1_macro']:.4f} | AUC={auc_str}"
            )

        seed_rows = [r for r in all_run_results if r["seed"] == seed]
        print(
            f"  Seed {seed} ({n_folds} folds): "
            f"Acc={np.mean([r['accuracy'] for r in seed_rows]):.4f} "
            f"+/- {np.std([r['accuracy'] for r in seed_rows]):.4f}  "
            f"F1m={np.mean([r['f1_macro'] for r in seed_rows]):.4f} "
            f"+/- {np.std([r['f1_macro'] for r in seed_rows]):.4f}"
        )

    _plot_experiment_results(all_run_results, all_histories, last_fold_data, exp_dir, config)

    df_results = pd.DataFrame(all_run_results)
    df_results.to_csv(os.path.join(exp_dir, "results.csv"), index=False)

    metric_cols = [m for m in ["accuracy", "f1_macro", "f1_weighted", "f1_binary", "auc"]
                   if m in df_results.columns]
    print(f"\n[INFO] {name} â€” Overall ({len(seeds) * n_folds} evals):")
    for mc in metric_cols:
        vals = df_results[mc].dropna()
        print(f"  {mc:<15s}: {vals.mean():.4f} +/- {vals.std():.4f}")
    print(f"[INFO] Results saved: {exp_dir}/results.csv")

    joblib.dump(last_fold_data["model"], os.path.join(exp_dir, "model_last_seed.pkl"))
    if last_fold_data.get("scaler") is not None:
        joblib.dump(last_fold_data["scaler"], os.path.join(exp_dir, "scaler_last_seed.pkl"))

    return all_run_results, all_histories


# ============================================================
# 10. PLOTTING
# ============================================================
def _plot_experiment_results(all_results, all_histories, last_data, exp_dir, config):
    """Generate all plots for a single binary experiment."""
    name = config["name"]
    seeds = config.get("seeds", [42, 123, 7])
    n_folds = config.get("n_folds", 5)
    y_test = last_data["y_test"]
    y_pred = last_data["y_pred"]
    y_prob = last_data.get("y_prob")

    # ---- Confusion Matrix ----
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=INCOME_LABELS)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix â€” {name}", fontweight="bold")
    plt.xticks(rotation=15, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---- ROC Curve (binary) ----
    if y_prob is not None:
        pos_prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
        fpr, tpr, _ = roc_curve(y_test, pos_prob)
        auc_val = roc_auc_score(y_test, pos_prob)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"AUC = {auc_val:.4f}")
        ax.fill_between(fpr, tpr, alpha=0.1, color="#3498db")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.50)")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(f"ROC Curve â€” {name}", fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, "roc_curve.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # ---- Metrics per Seed â€” Bar chart (mean Â+/- std) ----
    df = pd.DataFrame(all_results)
    actual_seeds = sorted(df["seed"].unique())
    plot_seeds = [s for s in seeds if s in actual_seeds] or actual_seeds

    metric_cols = ["accuracy", "f1_macro", "f1_binary"]
    if "auc" in df.columns and not df["auc"].isna().all():
        metric_cols.append("auc")

    bar_colors = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#e74c3c"]
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(5 * len(metric_cols), 5))
    if len(metric_cols) == 1:
        axes = [axes]

    for ax, mc in zip(axes, metric_cols):
        means = [df[df["seed"] == s][mc].mean() for s in plot_seeds]
        stds  = [df[df["seed"] == s][mc].std() for s in plot_seeds]
        x = np.arange(len(plot_seeds))
        ax.bar(x, means, yerr=stds, capsize=5, width=0.5,
               color=bar_colors[:len(plot_seeds)], edgecolor="white", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Seed {s}" for s in plot_seeds])
        ax.set_ylabel("Score")
        ax.set_title(mc, fontweight="bold")
        y_hi = min(1.0, max(m + s for m, s in zip(means, stds)) + 0.06)
        ax.set_ylim(0.0, y_hi)
        ax.grid(True, alpha=0.3, axis="y")
        for xi, (m, s) in enumerate(zip(means, stds)):
            ax.text(xi, m + s + 0.01, f"{m:.3f}", ha="center", va="bottom",
                    fontsize=9, color="navy", fontweight="bold")

    fig.suptitle(
        f"Metric Distribution across {n_folds} CV Folds â€” {name}",
        fontweight="bold", fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "metrics_per_seed.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---- SSL history ----
    has_history = any(len(h) > 0 for h in all_histories)
    if has_history:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for hist, seed in zip(all_histories, seeds):
            if len(hist) > 0:
                axes[0].plot(hist["iteration"], hist["added"], marker="o", label=f"Seed {seed}")
                axes[1].plot(hist["iteration"], hist["total_labeled"], marker="s", label=f"Seed {seed}")
                axes[2].plot(hist["iteration"], hist["mean_confidence"], marker="^", label=f"Seed {seed}")

        axes[0].set_title("Pseudo-labels Added per Iteration")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Samples Added")
        axes[0].legend()
        axes[1].set_title("Cumulative Labeled Samples")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Total Labeled")
        axes[1].legend()
        axes[2].set_title("Mean Confidence per Iteration")
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Mean Confidence")
        axes[2].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, "ssl_history.png"), dpi=150, bbox_inches="tight")
        plt.close()


# ============================================================
# 11. CROSS-EXPERIMENT COMPARISON
# ============================================================
def compare_experiments(all_experiment_results: list, results_dir: str):
    if len(all_experiment_results) < 2:
        return

    print("\n" + "=" * 60)
    print("CROSS-EXPERIMENT COMPARISON")
    print("=" * 60)

    flat = []
    for exp_results in all_experiment_results:
        flat.extend(exp_results)
    df = pd.DataFrame(flat)

    metrics = ["accuracy", "f1_macro", "f1_binary"]
    if "auc" in df.columns:
        metrics.append("auc")

    summary = df.groupby("experiment")[metrics].agg(["mean", "std"]).round(4)
    print(summary.to_string())
    summary.to_csv(os.path.join(results_dir, "comparison_summary.csv"))

    experiments = summary.index.tolist()
    fig, ax = plt.subplots(figsize=(max(12, len(experiments) * 2), 6))
    x = np.arange(len(experiments))
    width = 0.8 / len(metrics)
    colors = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"]

    for i, mc in enumerate(metrics):
        means = summary[(mc, "mean")].values
        stds = summary[(mc, "std")].values
        ax.bar(
            x + i * width, means, width, yerr=stds, capsize=3,
            color=colors[i % len(colors)], label=mc, edgecolor="white",
        )
        for j, (m, s) in enumerate(zip(means, stds)):
            ax.text(x[j] + i * width, m + s + 0.01, f"{m:.3f}", ha="center", fontsize=7)

    ax.set_xticks(x + width * len(metrics) / 2)
    ax.set_xticklabels(experiments, rotation=45, ha="right", fontsize=8)
    all_tops = [summary[(mc, "mean")].values + summary[(mc, "std")].values for mc in metrics]
    y_ceil = min(1.0, max(v.max() for v in all_tops) + 0.08)
    ax.set_ylim(0, y_ceil)
    ax.set_ylabel("Score")
    ax.set_title("Experiment Comparison â€” Mean +/- Std (Binary)", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "comparison_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Accuracy vs F1 binary ----
    exp_names, acc_vals, f1_vals = [], [], []
    for exp_results in all_experiment_results:
        exp_names.append(exp_results[0]["experiment"])
        acc_vals.append(np.mean([r["accuracy"] for r in exp_results]))
        f1_vals.append(np.mean([r["f1_binary"] for r in exp_results]))

    fig, ax = plt.subplots(figsize=(max(10, len(exp_names) * 1.5), 5))
    x = np.arange(len(exp_names))
    ax.bar(x - 0.15, acc_vals, 0.3, label="Accuracy", color="#3498db")
    ax.bar(x + 0.15, f1_vals, 0.3, label="F1 Binary", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha="right", fontsize=8)
    y_ceil = min(1.0, max(max(acc_vals), max(f1_vals)) + 0.08)
    ax.set_ylim(0, y_ceil)
    ax.set_ylabel("Score")
    ax.set_title("Accuracy vs F1 Binary â€” All Experiments", fontweight="bold")
    ax.legend()
    for i, (a, f) in enumerate(zip(acc_vals, f1_vals)):
        ax.text(i - 0.15, a + 0.01, f"{a:.3f}", ha="center", fontsize=7)
        ax.text(i + 0.15, f + 0.01, f"{f:.3f}", ha="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "comparison_acc_vs_f1.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Comparison plots saved to {results_dir}/")
    return summary


# ============================================================
# 12. THRESHOLD ANALYSIS
# ============================================================
def run_threshold_analysis(X, y, results_dir: str, seeds=(42, 123, 7)):
    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS  [BINARY]")
    print("=" * 60)

    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    threshold_results = []

    baseline_f1s = []
    for seed in seeds:
        X_l, X_u, X_t, y_l, y_u, y_t = make_split(X, y, 0.15, seed)
        scaler = StandardScaler()
        X_l_sc = scaler.fit_transform(X_l)
        X_t_sc = scaler.transform(X_t)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        lr.fit(X_l_sc, y_l)
        baseline_f1s.append(f1_score(y_t, lr.predict(X_t_sc), average="binary"))
    baseline_f1_mean = np.mean(baseline_f1s)

    for t in thresholds:
        run_metrics = []
        for seed in seeds:
            X_l, X_u, X_t, y_l, y_u, y_t = make_split(X, y, 0.15, seed)
            scaler = StandardScaler()
            model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
            final_model, final_scaler, _ = self_training(
                X_l, y_l, X_u, model, scaler,
                threshold=t, max_iter=10, max_per_iter=500, verbose=False,
            )
            X_t_sc = final_scaler.transform(X_t)
            y_pred = final_model.predict(X_t_sc)
            y_prob = final_model.predict_proba(X_t_sc)
            run_metrics.append(eval_metrics(y_t, y_pred, y_prob))

        threshold_results.append({
            "threshold": t,
            "f1_mean": np.mean([r["f1_binary"] for r in run_metrics]),
            "f1_std": np.std([r["f1_binary"] for r in run_metrics]),
            "auc_mean": np.mean([r["auc"] for r in run_metrics]),
            "auc_std": np.std([r["auc"] for r in run_metrics]),
            "acc_mean": np.mean([r["accuracy"] for r in run_metrics]),
        })
        print(
            f"  Threshold {t:.2f}: F1={threshold_results[-1]['f1_mean']:.4f} "
            f"+/- {threshold_results[-1]['f1_std']:.4f}"
        )

    thr_df = pd.DataFrame(threshold_results)
    thr_df.to_csv(os.path.join(results_dir, "threshold_analysis.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        thr_df["threshold"], thr_df["f1_mean"], yerr=thr_df["f1_std"],
        marker="o", capsize=4, color="steelblue", lw=2, label="F1 Binary (SSL)",
    )
    ax.plot(thr_df["threshold"], thr_df["auc_mean"], marker="s",
            linestyle="--", color="darkorange", lw=2, label="AUC (SSL)")
    ax.plot(thr_df["threshold"], thr_df["acc_mean"], marker="^",
            linestyle=":", color="#2ecc71", lw=2, label="Accuracy (SSL)")
    ax.axhline(y=baseline_f1_mean, color="red", linestyle=":", alpha=0.7, lw=2,
               label=f"LR Baseline F1 = {baseline_f1_mean:.4f}")
    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_ylabel("Metric Score", fontsize=12)
    ax.set_title("Impact of Confidence Threshold on Self-Training [Binary]",
                 fontweight="bold", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "threshold_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Threshold analysis saved to {results_dir}/")


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="ML Challenge 2 â€” SSL Experiments â€” BINARY income classification",
    )
    parser.add_argument("--config", type=str, default="experiments.json")
    parser.add_argument("--experiment", nargs="*", default=None)
    parser.add_argument("--list", action="store_true", dest="list_experiments")
    parser.add_argument("--results-dir", type=str, default="results_binary")
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--skip-threshold-analysis", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--replot", action="store_true",
        help="Regenerate plots from existing results CSVs without retraining.",
    )
    return parser.parse_args()


def load_config(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    if not isinstance(configs, list):
        raise ValueError("Config JSON must be a list of experiment objects.")
    for c in configs:
        if "name" not in c:
            raise ValueError("Each experiment must have a 'name' field.")
        if "base_model" not in c:
            raise ValueError(f"Experiment '{c['name']}' must have a 'base_model' field.")
    return configs


def main():
    args = parse_args()
    configs = load_config(args.config)

    if args.list_experiments:
        print("\nAvailable experiments (binary mode):")
        print("-" * 60)
        for c in configs:
            ssl = c.get("ssl_method") or "none (supervised)"
            print(f"  {c['name']:<40s}  SSL: {ssl}")
            if c.get("description"):
                print(f"    {c['description']}")
        return

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    # ---- REPLOT MODE ----
    if args.replot:
        exp_names = args.experiment or [c["name"] for c in configs]
        all_experiment_results = []
        for exp_name in exp_names:
            csv_path = os.path.join(results_dir, exp_name, "results.csv")
            if not os.path.exists(csv_path):
                print(f"[WARN] No results.csv found for {exp_name}, skipping.")
                continue
            df_r = pd.read_csv(csv_path)
            records = df_r.to_dict("records")
            all_experiment_results.append(records)
            cfg = next((c for c in configs if c["name"] == exp_name),
                       {"name": exp_name, "seeds": [42, 123, 7], "n_folds": 5})
            exp_dir = os.path.join(results_dir, exp_name)
            seeds = cfg.get("seeds", [42, 123, 7])
            n_folds = cfg.get("n_folds", 5)
            metric_cols = ["accuracy", "f1_macro", "f1_binary"]
            if "auc" in df_r.columns and not df_r["auc"].isna().all():
                metric_cols.append("auc")
            bar_colors = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#e74c3c"]
            fig, axes = plt.subplots(1, len(metric_cols), figsize=(5 * len(metric_cols), 5))
            if len(metric_cols) == 1:
                axes = [axes]
            for ax, mc in zip(axes, metric_cols):
                valid_cols = [s for s in seeds if s in df_r["seed"].values]
                means = [df_r[df_r["seed"] == s][mc].mean() for s in valid_cols]
                stds  = [df_r[df_r["seed"] == s][mc].std()  for s in valid_cols]
                x = np.arange(len(valid_cols))
                ax.bar(x, means, yerr=stds, capsize=5, width=0.5,
                       color=bar_colors[:len(valid_cols)], edgecolor="white", alpha=0.85)
                ax.set_xticks(x)
                ax.set_xticklabels([f"Seed {s}" for s in valid_cols])
                ax.set_ylabel("Score")
                ax.set_title(mc, fontweight="bold")
                y_hi = min(1.0, max(m + s for m, s in zip(means, stds)) + 0.06)
                ax.set_ylim(0.0, y_hi)
                ax.grid(True, alpha=0.3, axis="y")
                for xi, (m, s) in enumerate(zip(means, stds)):
                    ax.text(xi, m + s + 0.01, f"{m:.3f}", ha="center", va="bottom",
                            fontsize=9, color="navy", fontweight="bold")
            fig.suptitle(f"Metric Distribution across {n_folds} CV Folds â€” {exp_name}",
                         fontweight="bold", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(exp_dir, "metrics_per_seed.png"), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[INFO] Replotted: {exp_name}")
        compare_experiments(all_experiment_results, results_dir)
        print("[INFO] Replot complete.")
        return

    if args.experiment:
        selected = [c for c in configs if c["name"] in args.experiment]
        not_found = set(args.experiment) - {c["name"] for c in selected}
        if not_found:
            print(f"[ERROR] Experiments not found: {not_found}")
            print("Use --list to see available experiments.")
            sys.exit(1)
        configs = selected

    print("=" * 70)
    print("  ML CHALLENGE 2 â€” SEMI-SUPERVISED LEARNING  [BINARY]")
    print("  Domain: Economics & Socioeconomic Data (ACS PUMS 2022)")
    print("  Task: Predict Income â€” Above / Below Mean (Binary)")
    print(f"  Experiments to run: {[c['name'] for c in configs]}")
    print("=" * 70)

    raw_df = download_acs_data()

    if not args.skip_eda:
        explore_data(raw_df, results_dir)

    if args.sample_size:
        sample_size = args.sample_size
    else:
        sample_size = max(c.get("sample_size", 50_000) for c in configs)

    X, y, feature_names = preprocess(raw_df, sample_size=sample_size)

    all_experiment_results = []
    for config in configs:
        exp_results, exp_histories = run_experiment(config, X, y, results_dir)
        all_experiment_results.append(exp_results)

    compare_experiments(all_experiment_results, results_dir)

    seeds_for_analysis = configs[0].get("seeds", [42, 123, 7])
    if not args.skip_threshold_analysis:
        run_threshold_analysis(X, y, results_dir, seeds=tuple(seeds_for_analysis))

    print("\n" + "=" * 70)
    print("  ALL EXPERIMENTS COMPLETE  [BINARY]")
    print(f"  Results saved to: {os.path.abspath(results_dir)}/")
    print("=" * 70)


if __name__ == "__main__":
    main()

