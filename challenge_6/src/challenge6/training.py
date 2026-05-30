"""
Módulo de Ejecución y Orquestación de Experimentos.
Administra el flujo de trabajo completo: carga de datos, preprocesamiento, 
entrenamiento de modelos (Isolation Forest, AE, VAE), y extracción de métricas 
estadísticas y topológicas.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import os
from datetime import datetime

from .config import ExperimentConfig
from .data import build_dataset_loader
from .evaluation import anomaly_score_summary, spearman_rank_correlation
from .models import score_autoencoder, score_variational_autoencoder
from .preprocessing import build_numeric_preprocessor, isolation_forest_scores


@dataclass(frozen=True)
class ExperimentResult:
    """
    Estructura inmutable que encapsula los resultados integrales de una ejecución.
    Almacena configuraciones, métricas de correlación, representaciones del espacio 
    latente y vectores de puntaje de anomalía para análisis downstream.
    """
    config: ExperimentConfig
    correlations: dict[str, float]
    score_summaries: dict[str, dict[str, float]]
    iforest_scores: np.ndarray
    ae_latent_z: np.ndarray
    vae_latent_mu: np.ndarray
    vae_latent_z: np.ndarray
    price_values: np.ndarray | None
    rows: int
    feature_count: int


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """
    Ejecuta un ciclo completo de entrenamiento y evaluación.
    Secuencia:
    1. Carga y preprocesamiento de la matriz de características.
    2. Aislamiento estocástico (Isolation Forest).
    3. Compresión determinista (AE).
    4. Modelado probabilístico y regularización KL (VAE).
    5. Cálculo de similitud de rangos (Spearman) entre los detectores.
    """
    print(f"\n[INFO] Inicializando pipeline de experimentación. Semilla: {config.dataset.random_state}")
    
    # 1. Pipeline de Datos
    print("[INFO] Cargando y preprocesando dataset...")
    dataset = build_dataset_loader(config.dataset).load()
    preprocessor = build_numeric_preprocessor(config.preprocessing)
    feature_matrix = np.asarray(preprocessor.fit_transform(dataset.features), dtype=np.float32)
    
    # Extracción de la variable de partición económica (VALP) 
    # Esta variable no ingresa a los modelos; se reserva exclusivamente para proyecciones topológicas.
    price_values = None
    if "VALP" in dataset.features.columns:
        print("[INFO] Variable 'VALP' detectada. Reservando para estratificación topológica downstream.")
        price_values = np.asarray(pd.to_numeric(dataset.features["VALP"], errors="coerce"), dtype=float)

    # 2. Evaluación Baseline: Isolation Forest
    print("[INFO] Ejecutando métricas de partición espacial (Isolation Forest)...")
    iforest_scores = isolation_forest_scores(
        feature_matrix,
        config.model,
        random_state=config.dataset.random_state,
    )
    
    # 3. Evaluación Profunda: Standard Autoencoder (AE)
    print(f"[INFO] Entrenando Autoencoder Estándar (Latent Dim: {config.model.ae_latent_dim})...")
    ae_save_path = None
    if getattr(config, "save_models", False) and getattr(config, "output_dir", None):
        os.makedirs(config.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ae_save_path = os.path.join(config.output_dir, f"ae_latent{config.model.ae_latent_dim}_{ts}.pt")

    ae_scores = score_autoencoder(
        feature_matrix,
        config.model,
        random_state=config.dataset.random_state,
        save_path=ae_save_path,
    )
    ae_reconstruction = ae_scores.reconstruction_mse
    
    # 4. Evaluación Probabilística: Variational Autoencoder (VAE)
    print(f"[INFO] Entrenando Autoencoder Variacional (Latent Dim: {config.model.ae_latent_dim})...")
    vae_save_path = None
    if getattr(config, "save_models", False) and getattr(config, "output_dir", None):
        os.makedirs(config.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        vae_save_path = os.path.join(config.output_dir, f"vae_latent{config.model.ae_latent_dim}_{ts}.pt")

    vae_output = score_variational_autoencoder(
        feature_matrix,
        config.model,
        random_state=config.dataset.random_state,
        save_path=vae_save_path,
    )
    vae_scores = vae_output.elbo_loss

    # 5. Análisis de Acuerdo de Detectores (Spearman Rank Correlation)
    # Evalúa la congruencia monotónica entre paradigmas de detección radicalmente distintos.
    print("[INFO] Calculando coeficientes de correlación de Spearman (Detector Agreement)...")
    correlations = {
        "iforest_vs_ae": spearman_rank_correlation(iforest_scores, ae_reconstruction),
        "iforest_vs_vae": spearman_rank_correlation(iforest_scores, vae_scores),
        "ae_vs_vae": spearman_rank_correlation(ae_reconstruction, vae_scores),
    }
    
    # 6. Agregación Estadística
    print("[INFO] Consolidando distribuciones de errores y métricas ELBO...")
    summaries = {
        "iforest": anomaly_score_summary(iforest_scores),
        "ae_reconstruction_mse": anomaly_score_summary(ae_reconstruction),
        "vae_reconstruction_mse": anomaly_score_summary(vae_output.reconstruction_mse),
        "vae_kl_divergence": anomaly_score_summary(vae_output.kl_divergence),
        "vae_elbo": anomaly_score_summary(vae_scores),
    }

    print("[SUCCESS] Ciclo de experimentación finalizado correctamente.")
    return ExperimentResult(
        config=config,
        correlations=correlations,
        score_summaries=summaries,
        iforest_scores=iforest_scores,
        ae_latent_z=ae_scores.latent_z,
        vae_latent_mu=vae_output.latent_mu,
        vae_latent_z=vae_output.latent_z,
        price_values=price_values,
        rows=int(feature_matrix.shape[0]),
        feature_count=int(feature_matrix.shape[1]),
    )


def compare_pipeline_variants(config: ExperimentConfig) -> list[ExperimentResult]:
    """
    Ejecuta un estudio de ablación paramétrica sobre la dimensión del cuello de botella (bottleneck).
    Permite evaluar el balance empírico entre la pérdida de compresión (MSE) y 
    la capacidad de discriminación de anomalías.
    """
    print("\n[INFO] Iniciando Estudio de Ablación Dimensional (Bottleneck Sweep)...")
    
    # Definición del espacio de búsqueda para el cuello de botella
    latent_options = sorted(set([8, config.model.ae_latent_dim, 32]))
    results: list[ExperimentResult] = []
    
    for latent_dim in latent_options:
        print(f"\n[INFO] ==================================================")
        print(f"[INFO] Evaluando topología de compresión: p={latent_dim}")
        print(f"[INFO] ==================================================")
        
        # Propagación de la configuración alterada
        variant = replace(config, model=replace(config.model, ae_latent_dim=latent_dim))
        results.append(run_experiment(variant))
        
    print("\n[SUCCESS] Estudio de ablación dimensional concluido exitosamente.")
    return results