"""
Challenge 5 — Grupo 6: Housing & Urban Development
====================================================
Uso:
    python challenge5_grupo6.py                     # config.json
    python challenge5_grupo6.py --config mi.json    # config given in mi.json

Salida:
    runs/<timestamp>/
        config_used.json      copy config used for this run
        figures/              figures .png
        results/              .csv with metrics and profiles
"""

import argparse
import json
import os
import sys
import warnings
import zipfile
import urllib.request
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.preprocessing    import StandardScaler
from sklearn.decomposition    import PCA
from sklearn.impute           import SimpleImputer
from sklearn.cluster          import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors        import NearestNeighbors
from sklearn.metrics          import (silhouette_score,
                                      davies_bouldin_score,
                                      calinski_harabasz_score)
from scipy.cluster.hierarchy  import dendrogram, linkage as scipy_linkage

import umap as umap_lib
HAS_UMAP = True

# =============================================================================
# CLI
# =============================================================================
parser = argparse.ArgumentParser(description="Challenge 5 — Grupo 6")
parser.add_argument("--config", default="config.json",
                    help="Ruta al archivo JSON de configuración (default: config.json)")
args = parser.parse_args()

CONFIG_PATH = args.config
if not os.path.exists(CONFIG_PATH):
    print(f"✗ No se encontró el archivo de configuración: {CONFIG_PATH}")
    sys.exit(1)

with open(CONFIG_PATH) as f:
    C = json.load(f)

# =============================================================================
# SETUP PATHS AND RUN DIRECTORIES
# =============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR   = Path(C["output"]["base_dir"]) / timestamp
FIG_DIR   = RUN_DIR / C["output"]["figures_subdir"]
RES_DIR   = RUN_DIR / C["output"]["results_subdir"]

for d in (FIG_DIR, RES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Guardar copia del config en el run
import shutil
shutil.copy(CONFIG_PATH, RUN_DIR / "config_used.json")

DPI = C["output"]["dpi"]

print(f"\n{'='*60}")
print(f"  Challenge 5 — Grupo 6: Housing & Urban Development")
print(f"  Run directory: {RUN_DIR}")
print(f"{'='*60}\n")

# =============================================================================
# METRICS HELPER
# =============================================================================
def compute_metrics(X, labels, sample_size=5000, random_state=42):
    labels    = np.asarray(labels)
    mask      = labels != -1
    n_samples = int(mask.sum())
    n_clusters = int(len(set(labels[mask])))
    noise_frac = float((~mask).mean()) if np.any(labels == -1) else 0.0
    if n_clusters < 2 or n_samples < 2:
        return dict(clusters=n_clusters, noise_frac=noise_frac,
                    silhouette=np.nan, davies_bouldin=np.nan,
                    calinski_harabasz=np.nan, n_samples=n_samples)
    use_sample = int(min(sample_size, n_samples))
    ss  = silhouette_score(X[mask], labels[mask],
                           sample_size=use_sample, random_state=random_state)
    dbi = davies_bouldin_score(X[mask], labels[mask])
    chi = calinski_harabasz_score(X[mask], labels[mask])
    return dict(clusters=n_clusters, noise_frac=noise_frac,
                silhouette=float(ss), davies_bouldin=float(dbi),
                calinski_harabasz=float(chi), n_samples=n_samples)


def knn_distance_curve(X, min_samples):
    nn = NearestNeighbors(n_neighbors=min_samples, algorithm="ball_tree")
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    return np.sort(distances[:, -1])


def estimate_eps(knn_dist):
    if len(knn_dist) == 0:
        return 0.5
    grads = np.gradient(knn_dist)
    idx   = int(np.argmax(grads))
    eps   = float(knn_dist[idx])
    if not np.isfinite(eps) or eps <= 0:
        eps = float(np.quantile(knn_dist, 0.9))
    return eps


def make_hc(n_clusters, lnk):
    try:
        return AgglomerativeClustering(n_clusters=n_clusters,
                                       linkage=lnk, metric="euclidean")
    except TypeError:
        return AgglomerativeClustering(n_clusters=n_clusters,
                                       linkage=lnk, affinity="euclidean")


def savefig(name):
    path = FIG_DIR / name
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"    ✓ {path}")
    return path


PALETTE = cm.get_cmap("tab10")

def scatter(ax, X_emb, labels, title):
    unique = sorted(set(labels))
    ci = 0
    for u in unique:
        mask  = labels == u
        color = "lightgray" if u == -1 else PALETTE(ci % 10)
        ax.scatter(X_emb[mask, 0], X_emb[mask, 1],
                   c=[color], s=3, alpha=0.5, linewidths=0)
        if u != -1:
            ci += 1
    n_cl = len([u for u in unique if u != -1])
    ax.set_title(f"{title} ({n_cl} cl.)", fontsize=9)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")

# =============================================================================
# 1. DOWNLOAD
# =============================================================================
print("── 1. Download ──")
ZIP_FILE = C["data"]["zip_file"]
URL      = C["data"]["url"]

if not os.path.exists(ZIP_FILE):
    print(f"  Downloading {URL} …")
    urllib.request.urlretrieve(URL, ZIP_FILE)
print("  ✓ File available")

# =============================================================================
# 2. COLUMN INSPECTION
# =============================================================================
print("\n── 2. Column Inspection ──")

HOUSING_WANTED = C["features"]["all"]

with zipfile.ZipFile(ZIP_FILE) as z:
    csv_names = [n for n in z.namelist() if n.endswith(".csv")]
    if not csv_names:
        raise FileNotFoundError("No csv files found in the zip file.")

    def pick_housing_csv(names):
        for key in ("hus", "housing"):
            for name in names:
                if key in name.lower():
                    return name
        return names[0]

    csv_name = pick_housing_csv(csv_names)
    print(f"  CSV: {csv_name}")
    with z.open(csv_name) as f:
        all_cols = list(pd.read_csv(f, nrows=0).columns)

print(f"  All available columns: {len(all_cols)}")
cols_upper_map = {c.upper(): c for c in all_cols}

found_cols = []
for want in HOUSING_WANTED:
    real = cols_upper_map.get(want.upper())
    if real and real not in found_cols:
        found_cols.append(real)

print(f"  Found columns: {len(found_cols)} → {found_cols}")

if len(found_cols) < 5:
    EXCLUDE = {"WGTP","PWGTP","SERIALNO","RT","SPORDER","ADJINC","ADJHSG",
               "AGEP","CIT","COW","DDRS","DEAR","DEYE","DOUT","DPHY","DREM",
               "ENG","FER","GCL","GCM","GCR","LANX","MAR","MIG","MIL",
               "POBP","POVPIP","POWPUMA","POWSP","QTRBIR","RAC1P","RAC2P",
               "RAC3P","RACAIAN","RACASN","RACBLK","RACNH","RACNUM","RACPI",
               "RACSOR","RACWHT","RC","SCIENGP","SCIENGRLP","SCH","SCHG",
               "SCHL","SEX","SFN","SFR","SOCP","WAGP","WKHP","WKL","WKW",
               "WRK","YOEP","ANC","ANC1P","ANC2P","DECADE","DIS","DRIVESP",
               "ESP","ESR","FHISP","FOD1P","FOD2P","HICOV","INDP","JWMNP",
               "JWRIP","JWTR","LANP","MIGPUMA","MIGSP","OCCP","PAOC","PERNP",
               "PINCP","RELSHIPP","SWP","VPS"}
    with zipfile.ZipFile(ZIP_FILE) as z:
        with z.open(csv_name) as f:
            sample_df = pd.read_csv(f, nrows=1000, low_memory=False)
    num_cols   = sample_df.select_dtypes(include=[np.number]).columns.tolist()
    found_cols = [c for c in num_cols
                  if c.upper() not in EXCLUDE and c not in found_cols][:15]
    print(f"  (fallback) Found columns: {found_cols}")

# =============================================================================
# 3. LOADING THE DATASET
# =============================================================================
print("\n── 3. Loading the dataset ──")
with zipfile.ZipFile(ZIP_FILE) as z:
    with z.open(csv_name) as f:
        df_raw = pd.read_csv(f, usecols=found_cols,
                             dtype={c: "float32" for c in found_cols},
                             low_memory=True)

df_raw.columns = [c.upper() for c in df_raw.columns]
print(f"  Shape: {df_raw.shape}  |  RAM: {df_raw.memory_usage(deep=True).sum()/1e6:.1f} MB")

# =============================================================================
# 4. DATA CLEANING
# =============================================================================
print("\n── 4. Data Cleaning ──")
N_SAMPLE = C["data"]["n_sample"]
SEED0    = C["global_seeds"][0]

df = df_raw.copy(); del df_raw
df.replace([9999999, 9999998, -1], np.nan, inplace=True)
df.replace([np.inf, -np.inf],     np.nan, inplace=True)

for col in ["GRNTP","SMOCP","RNTP","ELEP","GASP","WATP"]:
    if col in df.columns:
        df[col] = df[col].replace(0, np.nan)

null_pct  = df.isnull().mean()
miss_thr  = C["data"]["missing_threshold"]
drop_cols = null_pct[null_pct > miss_thr].index.tolist()
df.drop(columns=drop_cols, inplace=True)
print(f"  Columns dropped (>{miss_thr*100:.0f}% nulls): {drop_cols}")

low_var_thr = C["data"]["low_variance_unique_threshold"]
low_var = [c for c in df.columns if df[c].dropna().nunique() <= low_var_thr]
df.drop(columns=low_var, inplace=True)
print(f"  Low variance columns dropped: {low_var}")

if len(df) > N_SAMPLE:
    df = df.sample(N_SAMPLE, random_state=SEED0).reset_index(drop=True)

print(f"  Final sample: {df.shape}  |  columns: {list(df.columns)}")
all_feature_names = list(df.columns)

# =============================================================================
# 5. BASE PROCESSING (imputation + scaling)
# =============================================================================
print("\n── 5. Base Processing ──")
imputer  = SimpleImputer(strategy="median")
X_imp    = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, dtype=np.float32)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_imp).astype(np.float32)
del df

print(f"  Cluster matrix: {X_scaled.shape}")

# =============================================================================
# 6. EDA
# =============================================================================
print("\n── 6. EDA ──")
n_cols_plot = min(12, len(all_feature_names))
fig, axes   = plt.subplots(3, 4, figsize=(16, 10))
axes_flat   = axes.flatten()

for i, col in enumerate(all_feature_names[:n_cols_plot]):
    vals    = X_imp[col].dropna()
    n_unique = vals.nunique()
    bins    = min(30, n_unique) if n_unique > 1 else 1
    axes_flat[i].hist(vals, bins=bins, color="#4C72B0", edgecolor="white", alpha=0.8)
    axes_flat[i].set_title(col, fontsize=8)

for j in range(n_cols_plot, 12):
    axes_flat[j].axis("off")

plt.suptitle("ACS PUMS (Housing features)", fontsize=11)
plt.tight_layout()
savefig("eda_histograms.png")

plt.figure(figsize=(10, 8))
sns.heatmap(X_imp.corr(), cmap="coolwarm", center=0,
            annot=len(all_feature_names) <= 12, fmt=".2f",
            linewidths=0.3, cbar_kws={"shrink": 0.8})
plt.title("Characteristics Correlation"); plt.tight_layout()
savefig("eda_correlation.png")

# =============================================================================
# 7. PCA (FULL FEATURE SET)
# =============================================================================
print("\n── 7. PCA ──")
var_thr   = C["pca"]["variance_threshold"]
pca_full  = PCA(random_state=SEED0)
pca_full.fit(X_scaled)
cumvar    = np.cumsum(pca_full.explained_variance_ratio_)
n_comp    = max(2, int(np.searchsorted(cumvar, var_thr)) + 1)
print(f"  Components for ≥{var_thr*100:.0f}% variance: {n_comp}")

pca      = PCA(n_components=n_comp, random_state=SEED0)
X_pca    = pca.fit_transform(X_scaled).astype(np.float32)

pca2d    = PCA(n_components=2, random_state=SEED0)
X_pca2d  = pca2d.fit_transform(X_scaled).astype(np.float32)

plt.figure(figsize=(7, 3))
plt.plot(range(1, len(cumvar)+1), cumvar*100, marker="o", ms=3, color="#2ca02c")
plt.axhline(var_thr*100, ls="--", color="red",  lw=1, label=f"{var_thr*100:.0f}%")
plt.axvline(n_comp,      ls="--", color="gray", lw=1)
plt.xlabel("Componentes"); plt.ylabel("Acumulate variance (%)"); plt.grid(alpha=0.3)
plt.legend(); plt.title("PCA — Variance"); plt.tight_layout()
savefig("pca_variance.png")

# UMAP (opcional)
if HAS_UMAP:
    print("\n── UMAP ──")
    reducer = umap_lib.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                             low_memory=True, random_state=SEED0)
    X_umap  = reducer.fit_transform(X_pca).astype(np.float32)
    print("  ✓ UMAP completed")
else:
    X_umap = None

# =============================================================================
# 8. FEATURE SUBSETS
# =============================================================================
print("\n── 8. Feature subsets ──")

def build_subset_matrix(subset_keys):
    """Builds X_scaled for a subset of features from the config."""
    cols_wanted = C["features"].get(subset_keys, [])
    cols_available = [c for c in cols_wanted if c in X_imp.columns]
    if len(cols_available) < 2:
        return None, cols_available
    sub = X_imp[cols_available].copy()
    sub_imp    = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(sub),
                               columns=sub.columns, dtype=np.float32)
    sub_scaled = StandardScaler().fit_transform(sub_imp).astype(np.float32)
    return sub_scaled, cols_available

spatial_scaled,  spatial_cols  = build_subset_matrix("spatial")
economic_scaled, economic_cols = build_subset_matrix("economic")
household_scaled,household_cols= build_subset_matrix("household")

subsets = {"full": (X_scaled, all_feature_names)}
if spatial_scaled   is not None: subsets["spatial"]   = (spatial_scaled,   spatial_cols)
if economic_scaled  is not None: subsets["economic"]  = (economic_scaled,  economic_cols)
if household_scaled is not None: subsets["household"] = (household_scaled, household_cols)

print(f"  Available subsets: {list(subsets.keys())}")

# =============================================================================
# 9. K-MEANS — GRID DONE (elbow + silhouette, multiples seeds)
# =============================================================================
print("\n── 9. K-Means grid ──")

K_RANGE      = C["kmeans"]["k_range"]
KM_SEEDS     = C["kmeans"]["seeds"]
KM_N_INIT    = C["kmeans"]["n_init"]
KM_BATCH     = C["kmeans"]["batch_size"]
KM_METR_SAMP = C["kmeans"]["metric_sample"]

all_km_rows = []

for subset_name, (X_sub, _) in subsets.items():
    # PCA for the subsets
    pca_sub = PCA(n_components=min(n_comp, X_sub.shape[1]), random_state=SEED0)
    X_sub_pca = pca_sub.fit_transform(X_sub).astype(np.float32)

    for k in K_RANGE:
        for seed in KM_SEEDS:
            km = MiniBatchKMeans(n_clusters=k, n_init=KM_N_INIT,
                                 random_state=seed, batch_size=KM_BATCH)
            labels = km.fit_predict(X_sub_pca)
            m = compute_metrics(X_sub_pca, labels,
                                sample_size=KM_METR_SAMP, random_state=seed)
            all_km_rows.append({
                "subset": subset_name, "k": k, "seed": seed,
                "inertia": float(km.inertia_),
                **m
            })
            print(f"  [KMeans|{subset_name}] k={k:2d} seed={seed}  "
                  f"inertia={km.inertia_:,.0f}  sil={m['silhouette']:.4f}")

df_km = pd.DataFrame(all_km_rows)
df_km.to_csv(RES_DIR / "kmeans_grid.csv", index=False)
print(f"  ✓ {RES_DIR / 'kmeans_grid.csv'}")

# Elbow + Silhouette (full, media ± std sobre seeds)
for subset_name in subsets:
    sub_df  = df_km[df_km["subset"] == subset_name]
    summary = (sub_df.groupby("k")
               .agg(inertia_mean=("inertia","mean"), inertia_std=("inertia","std"),
                    silhouette_mean=("silhouette","mean"), silhouette_std=("silhouette","std"))
               .reset_index())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.errorbar(summary["k"], summary["inertia_mean"], yerr=summary["inertia_std"],
                 marker="o", color="#1f77b4", capsize=4)
    ax1.set_title(f"Elbow — {subset_name}"); ax1.set_xlabel("k"); ax1.grid(alpha=0.3)

    ax2.errorbar(summary["k"], summary["silhouette_mean"], yerr=summary["silhouette_std"],
                 marker="s", color="#d62728", capsize=4)
    ax2.set_title(f"Silhouette Score — {subset_name}"); ax2.set_xlabel("k"); ax2.grid(alpha=0.3)
    plt.suptitle(f"K-Means | subset={subset_name} | seeds={KM_SEEDS}", fontsize=10)
    plt.tight_layout()
    savefig(f"kmeans_elbow_sil_{subset_name}.png")

# Mejor configuración global (full, por silhouette_mean)
best_km_summary = (df_km[df_km["subset"] == "full"]
                   .groupby("k")
                   .agg(silhouette_mean=("silhouette","mean"),
                        silhouette_std=("silhouette","std"))
                   .reset_index())
best_k = int(best_km_summary.sort_values("silhouette_mean", ascending=False).iloc[0]["k"])
best_km_seed = int(
    df_km[(df_km["subset"] == "full") & (df_km["k"] == best_k)]
    .sort_values("silhouette", ascending=False).iloc[0]["seed"]
)
print(f"\n  → Mejor k = {best_k}  (seed={best_km_seed})")

# Label final de K-Means oveer X_pca
km_final  = MiniBatchKMeans(n_clusters=best_k, n_init=KM_N_INIT,
                             random_state=best_km_seed, batch_size=KM_BATCH)
labels_km = km_final.fit_predict(X_pca)

# =============================================================================
# 10. DBSCAN — grid over min_samples + eps sweep
# =============================================================================
print("\n── 10. DBSCAN grid ──")

DB_MS_LIST  = list(C["dbscan"]["min_samples_list"])
if C["dbscan"]["use_2x_d"]:
    two_d = 2 * X_pca.shape[1]
    if two_d not in DB_MS_LIST:
        DB_MS_LIST.append(two_d)
DB_STEPS    = C["dbscan"]["eps_sweep_steps"]

all_db_rows = []

fig_knn, axes_knn = plt.subplots(1, len(DB_MS_LIST),
                                  figsize=(5 * len(DB_MS_LIST), 3))
if len(DB_MS_LIST) == 1:
    axes_knn = [axes_knn]

for i, ms in enumerate(DB_MS_LIST):
    print(f"  [DBSCAN] min_samples={ms}")
    knn_dist = knn_distance_curve(X_pca, ms)
    eps_auto = estimate_eps(knn_dist)
    eps_values = np.round(np.linspace(eps_auto * 0.5, eps_auto * 2.0, DB_STEPS), 4)

    axes_knn[i].plot(knn_dist, color="#9467bd", lw=1.2)
    axes_knn[i].axhline(eps_auto, ls="--", color="red", lw=1, label=f"ε≈{eps_auto:.3f}")
    axes_knn[i].set_title(f"k-NN dist | ms={ms}", fontsize=9)
    axes_knn[i].set_xlabel("Puntos"); axes_knn[i].set_ylabel(f"dist al vecino {ms}")
    axes_knn[i].legend(fontsize=7); axes_knn[i].grid(alpha=0.3)

    for eps in eps_values:
        db     = DBSCAN(eps=float(eps), min_samples=ms)
        labels = db.fit_predict(X_pca)
        m      = compute_metrics(X_pca, labels)
        all_db_rows.append({
            "min_samples": ms, "eps": float(eps), **m
        })
        print(f"    eps={eps:.4f}  cl={m['clusters']}  "
              f"noise={m['noise_frac']:.3f}  sil={m['silhouette']:.4f}")

plt.suptitle("DBSCAN — k-NN Distance Plots", fontsize=11)
plt.tight_layout()
savefig("dbscan_knn_grid.png")

df_db = pd.DataFrame(all_db_rows)
df_db.to_csv(RES_DIR / "dbscan_grid.csv", index=False)
print(f"  ✓ {RES_DIR / 'dbscan_grid.csv'}")

valid_db  = df_db[df_db["silhouette"].notna() & (df_db["clusters"] >= 2)]
best_db_row = (valid_db.sort_values("silhouette", ascending=False).iloc[0]
               if len(valid_db) > 0 else df_db.sort_values("clusters", ascending=False).iloc[0])

best_eps = float(best_db_row["eps"])
best_ms  = int(best_db_row["min_samples"])
print(f"\n  → Mejor ε = {best_eps}  min_samples = {best_ms}")

db_final   = DBSCAN(eps=best_eps, min_samples=best_ms)
labels_db  = db_final.fit_predict(X_pca)
noise_frac = float((labels_db == -1).mean())
n_cl_db    = len(set(labels_db)) - (1 if -1 in labels_db else 0)
print(f"  Clusters: {n_cl_db}  |  Ruido: {noise_frac:.3f}")

# =============================================================================
# 11. HIERARCHICAL — grid for linkage + dendrogram
# =============================================================================
print("\n── 11. Hierarchical grid ──")

HC_LINKAGES   = C["hierarchical"]["linkages"]
HC_K_RANGE    = C["hierarchical"]["k_range"]
HC_SAMPLE     = C["hierarchical"]["sample_limit"]
HC_TRUNC_LVL  = C["hierarchical"]["dendrogram_truncate_level"]
HC_METR_SAMP  = C["hierarchical"]["metric_sample"]

np.random.seed(SEED0)
idx_hc  = (np.random.choice(len(X_pca), min(HC_SAMPLE, len(X_pca)), replace=False)
           if len(X_pca) > HC_SAMPLE else np.arange(len(X_pca)))
X_hc    = X_pca[idx_hc]
print(f"  Muestra para HC: {len(X_hc)} puntos")

# Dendrogram
fig_dend, axes_dend = plt.subplots(1, len(HC_LINKAGES), figsize=(6 * len(HC_LINKAGES), 5))
if len(HC_LINKAGES) == 1:
    axes_dend = [axes_dend]

all_hc_rows = []

for i, lnk in enumerate(HC_LINKAGES):
    print(f"  [HC] linkage={lnk}")
    Z = scipy_linkage(X_hc, method=lnk)

    dendrogram(Z, ax=axes_dend[i], truncate_mode="level", p=HC_TRUNC_LVL,
               no_labels=True, above_threshold_color="gray",
               color_threshold=0.7 * max(Z[:, 2]))
    axes_dend[i].set_title(f"Dendrogram — {lnk}", fontsize=9)

    for k in HC_K_RANGE:
        hc = make_hc(n_clusters=k, lnk=lnk)
        labels = hc.fit_predict(X_hc)
        m = compute_metrics(X_hc, labels, sample_size=HC_METR_SAMP)
        all_hc_rows.append({
            "linkage": lnk, "k": k, **m
        })
        print(f"    k={k:2d}  sil={m['silhouette']:.4f}")

plt.suptitle(f"Dendrogramas — Hierarchical (muestra {len(X_hc)})", fontsize=11)
plt.tight_layout()
savefig("dendrogram_comparison.png")

df_hc = pd.DataFrame(all_hc_rows)
df_hc.to_csv(RES_DIR / "hierarchical_grid.csv", index=False)
print(f"  ✓ {RES_DIR / 'hierarchical_grid.csv'}")

valid_hc = df_hc[df_hc["silhouette"].notna() & (df_hc["clusters"] >= 2)]
best_hc_row = (valid_hc.sort_values("silhouette", ascending=False).iloc[0]
               if len(valid_hc) > 0 else df_hc.iloc[0])
best_hc_k   = int(best_hc_row["k"])
best_lnk    = str(best_hc_row["linkage"])
print(f"\n  → Mejor linkage={best_lnk}  k={best_hc_k}")

hc_final  = make_hc(n_clusters=best_hc_k, lnk=best_lnk)
labels_hc = hc_final.fit_predict(X_hc)
# Extend labels to full X_pca (assign -2 to non-sampled points)
labels_hc_full = np.full(len(X_pca), -2, dtype=int)
labels_hc_full[idx_hc] = labels_hc

labels_hc_use = labels_hc

# =============================================================================
# 12. DIMENSIONALITY ABLATION COMPARISON (FULL vs PCA)
# =============================================================================
print("\n── 12. Dimensionality Ablation ──")

ablation_rows = []
for rep_name, X_rep in [("full_scaled", X_scaled), ("pca_reduced", X_pca)]:
    for seed in KM_SEEDS:
        km = MiniBatchKMeans(n_clusters=best_k, n_init=KM_N_INIT,
                             random_state=seed, batch_size=KM_BATCH)
        lbl = km.fit_predict(X_rep)
        m   = compute_metrics(X_rep, lbl, sample_size=KM_METR_SAMP, random_state=seed)
        ablation_rows.append({
            "representation": rep_name, "seed": seed, "k": best_k,
            "dims": X_rep.shape[1], **m
        })
        print(f"  [{rep_name}] seed={seed}  sil={m['silhouette']:.4f}  "
              f"dbi={m['davies_bouldin']:.4f}  chi={m['calinski_harabasz']:.2f}")

df_ablation = pd.DataFrame(ablation_rows)
df_ablation.to_csv(RES_DIR / "ablation_full_vs_pca.csv", index=False)
print(f"  ✓ {RES_DIR / 'ablation_full_vs_pca.csv'}")

# =============================================================================
# 13. FINAL
# =============================================================================
print("\n── 13. Table ──")

def metrics_row(X, labels, algo_label, sample_size=5000):
    m = compute_metrics(X, labels, sample_size=sample_size, random_state=SEED0)
    return {
        "Algoritmo":           algo_label,
        "Clusters":            m["clusters"],
        "Ruido (%)":           round(m["noise_frac"] * 100, 1),
        "Silhouette ↑":        round(m["silhouette"],         4) if not np.isnan(m["silhouette"])         else "-",
        "Davies-Bouldin ↓":    round(m["davies_bouldin"],     4) if not np.isnan(m["davies_bouldin"])     else "-",
        "Calinski-Harabasz ↑": round(m["calinski_harabasz"],  2) if not np.isnan(m["calinski_harabasz"])  else "-",
    }

comp_rows = [
    metrics_row(X_pca, labels_km, f"K-Means (k={best_k}, seed={best_km_seed})"),
    metrics_row(X_pca, labels_db, f"DBSCAN (ε={best_eps:.4f}, ms={best_ms})"),
    metrics_row(X_hc,  labels_hc_use, f"Hierarchical (k={best_hc_k}, {best_lnk})"),
]

df_comp = pd.DataFrame(comp_rows)
print("\n" + df_comp.to_string(index=False))
df_comp.to_csv(RES_DIR / "metrics_comparison.csv", index=False)
print(f"\n  ✓ {RES_DIR / 'metrics_comparison.csv'}")

# =============================================================================
# 14. VIEWS
# =============================================================================
print("\n── 14. Views 2D ──")

ALGO_LABELS = {
    f"K-Means k={best_k}":   labels_km,
    f"DBSCAN ε={best_eps:.3f}": labels_db,
    f"Hierarchical {best_lnk}": labels_hc_full,
}

# PCA 2D
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Clusters — PCA 2D  |  Grupo 6 Housing", fontsize=12)
for ax, (algo, lbl) in zip(axes, ALGO_LABELS.items()):
    scatter(ax, X_pca2d, lbl, algo)
plt.tight_layout()
savefig("clusters_pca2d.png")

# UMAP (si disponible)
if X_umap is not None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Clusters — UMAP 2D  |  Grupo 6 Housing", fontsize=12)
    for ax, (algo, lbl) in zip(axes, ALGO_LABELS.items()):
        scatter(ax, X_umap, lbl, algo)
    plt.tight_layout()
    savefig("clusters_umap.png")

# =============================================================================
# 15. CLUSTERS PROFILES
# =============================================================================
print("\n── 15. Clusters Profiles ──")

df_prof = X_imp.copy()
df_prof["cluster"] = labels_km
profile = (df_prof[df_prof["cluster"] != -1]
           .groupby("cluster").mean().round(2))
profile.to_csv(RES_DIR / "cluster_profiles_kmeans.csv")
print(f"  ✓ {RES_DIR / 'cluster_profiles_kmeans.csv'}")

prof_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
plt.figure(figsize=(11, max(3, best_k * 0.7)))
sns.heatmap(prof_norm, annot=profile.values, fmt=".1f",
            cmap="YlOrRd", linewidths=0.4)
plt.title(f"Perfil K-Means (k={best_k}) — original feature means", fontsize=11)
plt.tight_layout()
savefig("profile_heatmap.png")

# =============================================================================
# FINAL OUTPUT
# =============================================================================
print(f"\n{'='*60}")
print(f"  DONE — Challenge 5 Grupo 6")
print(f"  Resultados en: {RUN_DIR}")
print(f"{'='*60}")
print("\n" + df_comp.to_string(index=False))
print(f"\n  Figuras:   {list(FIG_DIR.iterdir())}")
print(f"  Tablas:    {list(RES_DIR.iterdir())}")
