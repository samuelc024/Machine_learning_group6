from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def anomaly_score_summary(scores: np.ndarray) -> dict[str, float]:
    vector = np.asarray(scores, dtype=float)
    return {
        "mean": float(np.mean(vector)),
        "std": float(np.std(vector)),
        "p95": float(np.percentile(vector, 95)),
        "p99": float(np.percentile(vector, 99)),
    }


def spearman_rank_correlation(score_a: np.ndarray, score_b: np.ndarray) -> float:
    vector_a = np.asarray(score_a, dtype=float)
    vector_b = np.asarray(score_b, dtype=float)
    if np.all(vector_a == vector_a[0]) or np.all(vector_b == vector_b[0]):
        return 0.0

    corr, _ = spearmanr(vector_a, vector_b)
    if np.isnan(corr):
        return 0.0
    return float(corr)