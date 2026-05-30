from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig
from .data import build_dataset_loader

def run_post_analysis(json_path: Path, config: ExperimentConfig) -> None:
    print(f"\nCargando métricas no supervisadas desde: {json_path.name}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["full_result"]

    print(" Cargando los 2 datasets (psam_husa.csv y psam_husb.csv) para alinear características")
    dataset = build_dataset_loader(config.dataset).load()
    features = dataset.features

    iforest_scores = np.array(data["iforest_scores"])
    ae_z = np.array(data["ae_latent_z"])
    vae_mu = np.array(data["vae_latent_mu"])

    # ---------------------------------------------------------
    # PARTE A: TOP 10 DE ANOMALÍAS
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print(" TOP 10 VIVIENDAS MÁS ANÓMALAS (ESTUDIO DE CASO)")
    print("="*70)
    
    features_with_scores = features.copy()
    features_with_scores["ANOMALY_SCORE"] = iforest_scores
    top_10 = features_with_scores.sort_values(by="ANOMALY_SCORE", ascending=False).head(10)
    
    # Seleccionamos las columnas más relevantes basándonos en tu cluster_profiles_kmeans.csv
    cols_to_show = ["VALP", "HINCP", "BDSP", "RMSP", "BLD", "YBL", "TEN", "TAXA", "ANOMALY_SCORE"]
    cols_exist = [col for col in cols_to_show if col in top_10.columns]
    
    print(top_10[cols_exist].to_string(index=False))
    print("-" * 70)

    # ---------------------------------------------------------
    # PARTE B: CÁLCULO DE SILHOUETTE SCORE (Para la Tabla II)
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print(" CÁLCULO DE SILHOUETTE SCORES (Basado en K-Means Reto 5)")
    print("="*70)
    print("[*] Submuestreando a 20,000 registros para evitar colapso")
    
    sample_size = min(20000, len(features))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(features), size=sample_size, replace=False)

    ae_z_sample = ae_z[sample_idx]
    vae_mu_sample = vae_mu[sample_idx]
    
    # 1. Recrear el espacio crudo estandarizado del Reto 5
    raw_sample = features.iloc[sample_idx]
    preprocessor = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    scaled_sample = preprocessor.fit_transform(raw_sample)
    
    # 2. Recrear las etiquetas (K=2) que obtuvieron en el Reto 5
    print("[*] Ejecutando K-Means (k=2) sobre el espacio original...")
    kmeans = MiniBatchKMeans(n_clusters=2, random_state=62, batch_size=1024)
    labels_kmeans = kmeans.fit_predict(scaled_sample)

    # 3. Evaluar cómo se agrupan esas mismas etiquetas en los espacios profundos
    print("[*] Calculando métricas de cohesión topológica...\n")
    
    sil_raw = silhouette_score(scaled_sample, labels_kmeans)
    sil_ae = silhouette_score(ae_z_sample, labels_kmeans)
    sil_vae = silhouette_score(vae_mu_sample, labels_kmeans)
    
    print(f"-> Silhouette Score en Raw Features (Baseline Reto 5): {sil_raw:.4f}")
    print(f"-> Silhouette Score en Espacio Latente AE (z):         {sil_ae:.4f}")
    print(f"-> Silhouette Score en Espacio Latente VAE (mu):       {sil_vae:.4f}")
    print("="*70)