"""
Módulo de Preprocesamiento y Detección de Anomalías Baseline.
Implementa pipelines de normalización, imputación estadística y 
modelado de aislamiento estocástico (Isolation Forest).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig, PreprocessingConfig


def build_numeric_preprocessor(config: PreprocessingConfig) -> Pipeline:
    """
    Construye un pipeline de preprocesamiento de datos numéricos.
    Secuencia:
    1. Imputación de valores faltantes mediante la mediana (robusto ante outliers).
    2. Escalado (StandardScaler) para normalización Z-score.
    """
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if config.scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def isolation_forest_scores(
    features: np.ndarray, 
    model_config: ModelConfig, 
    *, 
    random_state: int
) -> np.ndarray:
    """
    Entrena un Isolation Forest para obtener puntajes de anomalía basados en 
    longitud de trayectoria.
    
    Lógica de Inversión: 
    IsolationForest.score_samples devuelve mayores valores para puntos "normales".
    Se aplica negación para alinear la escala con el error de reconstrucción de 
    los Autoencoders (donde a mayor valor, mayor anomalousness).
    """
    print(f"[INFO] Inicializando Isolation Forest (n_estimators=300, contamination={model_config.iforest_contamination})...")
    
    detector = IsolationForest(
        n_estimators=300,
        contamination=model_config.iforest_contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    
    print("[INFO] Ajustando Isolation Forest al manifold de características...")
    detector.fit(features)
    
    print("[SUCCESS] Cálculo de scores de aislamiento completado.")
    return -detector.score_samples(features)