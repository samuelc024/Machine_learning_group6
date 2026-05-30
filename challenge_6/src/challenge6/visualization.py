"""
Módulo de Proyección Topológica y Visualización del Espacio Latente.
Implementa reducción dimensional (UMAP, t-SNE) para la evaluación de 
modelos profundos, estratificando por puntaje de anomalía y quintiles económicos.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
# Configuración del backend 'Agg' para ejecución sin interfaz gráfica (headless/CI)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from umap import UMAP


def _sample_indices(sample_count: int, limit: int, random_state: int) -> np.ndarray:
    """
    Genera un submuestreo aleatorio estratificado para optimizar la carga computacional
    de las proyecciones topológicas.
    """
    if sample_count <= limit:
        return np.arange(sample_count)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(sample_count, size=limit, replace=False))


def _reduce_umap(matrix: np.ndarray, random_state: int) -> np.ndarray:
    """
    Aplica Uniform Manifold Approximation and Projection (UMAP).
    Preserva tanto la estructura local como la topología global del espacio latente.
    """
    sample_count = int(matrix.shape[0])
    neighbors = max(2, min(15, sample_count - 1))
    reducer = UMAP(
        n_components=2, 
        n_neighbors=neighbors, 
        min_dist=0.1, 
        init="random", 
        random_state=random_state
    )
    return reducer.fit_transform(matrix)


def _reduce_tsne(matrix: np.ndarray, random_state: int) -> np.ndarray:
    """
    Aplica t-Distributed Stochastic Neighbor Embedding (t-SNE).
    Optimizado para maximizar la cohesión espacial de vecindarios locales.
    """
    sample_count = int(matrix.shape[0])
    perplexity = max(2, min(30, (sample_count - 1) // 3))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    return reducer.fit_transform(matrix)


def _plot_projection(
    ax: plt.Axes, 
    coordinates: np.ndarray, 
    color_values: np.ndarray, 
    *, 
    title: str, 
    colorbar_label: str | None = None, 
    categorical: bool = False
) -> None:
    """
    Renderiza una proyección bidimensional específica sobre un eje de Matplotlib.
    Maneja dinámicamente mapas de color continuos (anomalías) y categóricos (quintiles).
    """
    if categorical:
        scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=color_values, cmap="tab10", s=8, alpha=0.8)
        colorbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(colorbar_label or "Quintile")
        colorbar.set_ticks([0, 1, 2, 3, 4])
    else:
        scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=color_values, cmap="magma", s=8, alpha=0.8)
        colorbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(colorbar_label or "Anomaly Score")
    
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(True, linestyle="--", alpha=0.3)


def save_latent_visualizations(
    *,
    latent_name: str,
    latent_matrix: np.ndarray,
    anomaly_scores: np.ndarray,
    price_values: np.ndarray | None,
    output_dir: Path,
    random_state: int,
    max_points: int = 5000,
) -> Path | None:
    """
    Pipeline principal para la generación y exportación de análisis visuales del espacio latente.
    Ejecuta filtrado de valores finitos, submuestreo, reducción dimensional (UMAP/t-SNE) y renderizado.
    """
    print(f"\n[INFO] Inicializando pipeline de visualización topológica para: {latent_name}")
    
    matrix = np.asarray(latent_matrix, dtype=float)
    scores = np.asarray(anomaly_scores, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] == 0:
        print(f"[WARNING] Matriz inválida o vacía para {latent_name}. Abortando visualización.")
        return None

    # Filtrado estricto de valores no finitos (NaN/Inf)
    valid_mask = np.isfinite(scores)
    if price_values is not None:
        prices = np.asarray(price_values, dtype=float)
        valid_mask &= np.isfinite(prices)
    else:
        prices = None

    matrix = matrix[valid_mask]
    scores = scores[valid_mask]
    if prices is not None:
        prices = prices[valid_mask]

    if matrix.shape[0] == 0:
        print(f"[WARNING] Ningún registro válido remanente tras filtrado de NaNs para {latent_name}.")
        return None

    # Estratificación por quintiles manejando colisiones de valores idénticos
    if prices is not None:
        price_series = pd.Series(prices)
        try:
            quintiles = pd.qcut(price_series, 5, labels=False, duplicates="drop").to_numpy(dtype=int)
        except ValueError:
            print("[WARNING] Varianza insuficiente para cálculo de quintiles. Asignando valor constante.")
            quintiles = np.ones(price_series.shape[0], dtype=int)
    else:
        quintiles = None

    # Submuestreo para viabilidad computacional
    keep_indices = _sample_indices(matrix.shape[0], max_points, random_state)
    matrix = matrix[keep_indices]
    scores = scores[keep_indices]
    if quintiles is not None:
        quintiles = quintiles[keep_indices]

    if matrix.shape[0] < 3:
        print(f"[WARNING] Muestra insuficiente (<3) para reducción dimensional en {latent_name}.")
        return None

    print(f"[INFO] Calculando proyección UMAP ({matrix.shape[0]} muestras)...")
    umap_coords = _reduce_umap(matrix, random_state)
    
    print(f"[INFO] Calculando proyección t-SNE ({matrix.shape[0]} muestras)...")
    tsne_coords = _reduce_tsne(matrix, random_state)

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"{latent_name}_latent_space.png"

    # Construcción de la figura en cuadrícula (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    
    _plot_projection(axes[0, 0], umap_coords, scores, title=f"{latent_name} UMAP - Anomaly Score", colorbar_label="Anomaly Score")
    _plot_projection(axes[1, 0], tsne_coords, scores, title=f"{latent_name} t-SNE - Anomaly Score", colorbar_label="Anomaly Score")

    if quintiles is not None:
        _plot_projection(axes[0, 1], umap_coords, quintiles, title=f"{latent_name} UMAP - Price Quintiles", colorbar_label="Price Quintile", categorical=True)
        _plot_projection(axes[1, 1], tsne_coords, quintiles, title=f"{latent_name} t-SNE - Price Quintiles", colorbar_label="Price Quintile", categorical=True)
    else:
        axes[0, 1].axis("off")
        axes[1, 1].axis("off")

    fig.suptitle(f"Topological Latent Projections: {latent_name.upper()}", fontsize=18, fontweight="bold")
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    
    print(f"[SUCCESS] Visualizaciones exportadas correctamente en: {figure_path}")
    return figure_path