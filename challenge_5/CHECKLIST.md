# Grading Checklist — Challenge 5, Grupo 6

## Dataset

| Field | Value |
|---|---|
| **Name** | ACS PUMS 2022 1-Year Housing File |
| **Source URL** | https://www2.census.gov/programs-surveys/acs/data/pums/2022/1-Year/csv_hus.zip |
| **Records after sampling** | 20,000 |
| **Features after preprocessing** | "BDSP", "RMSP", "BLD", "YBL", "TEN","GRNTP", "SMOCP", "RNTP","HINCP", "NP", "VEH", "FS","ELEP", "GASP", "WATP", "HHT", "HHL", "HUGCL", "HUPAC", "GRPIP", "OCPIP" |

---

## Final Hyperparameter Configurations

| Algorithm | Hyperparameter | Value |
|---|---|---|
| **K-Means** | k | 2 |
| | seeds | 42, 52, 62 |
| | n_init | 10 |
| | batch_size | 2048 |
| | feature subset | economic |
| **DBSCAN** | ε (eps) | 6.105 |
| | min_samples | 10 |
| | feature subset | full (PCA-reduced) |
| **Hierarchical** | k | 2 |
| | linkage | average |
| | sample limit | 5,000 |
| | feature subset | full (PCA-reduced) |

---

## Best-Configuration Metrics

> Values come from `results/metrics_comparison.csv` after running the pipeline.

| Algorithm | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ |
|---|---|---|---|
| K-Means | 0.2486 | 1.9416 | 3995.44 |
| DBSCAN | 0.7804 | 0.3423 | 303.90 |
| Hierarchical | 0.7720 | 0.4883 | 167.52 |

---

## Algorithm Comparison

On the ACS PUMS housing dataset, DBSCAN (eps=6.10, min_samples=10) is mathematically the most appropriate algorithm for geometric isolation, achieving the highest Silhouette Score (0.780) and the lowest Davies-Bouldin Index (0.342). The PCA ablation down to 11 components reveals that housing data forms an elongated, continuous density structure rather than tight, uniform hyperspheres. Consequently, DBSCAN excels by capturing this continuous macro-population into a single baseline cluster while cleanly isolating extreme financial outliers (0.1% noise fraction) without generating artificial boundaries.Conversely, K-Means struggles on the full feature space (Silhouette of 0.249) because its centroid-based logic is forced to split this continuous density mass with arbitrary linear cuts. However, a feature ablation study reveals that K-Means restricted exclusively to the economic subset ($k=2$) optimizes its performance to a Silhouette of 0.517. This economic partition provides the most practical and interpretable semantic profiling, mapping clearly to recognizable socio-economic archetypes: wealthy owner-occupiers and lower-income renters.Hierarchical Clustering (Average linkage) confirms this structure, reaching a solid Silhouette of 0.772 at $k=2$, but its cubic time complexity requires a 5,000-record subsample, restricting its scalability. Therefore, for pure density validation, DBSCAN is the superior model, whereas K-Means restricted to economic features remains the most practical choice for actionable macroeconomic segmentation.

---

## How to Reproduce

```bash
# from the challenge_5/ directory
python challenge5_grupo6.py --config config.json
```
