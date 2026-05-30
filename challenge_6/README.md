# Challenge 6

Pipeline 100% no supervisado para detectar perfiles anómalos en datos socioeconómicos ACS.

## Objetivo

Este proyecto no predice ingresos. Toma la matriz de features, entrena dos redes profundas en PyTorch y compara sus rankings de anomalía contra un baseline de Isolation Forest:

- Autoencoder (AE) con cuello de botella.
- Variational Autoencoder (VAE) con reparametrización.
- Isolation Forest como baseline de ranking.

La comparación se realiza con correlación de Spearman entre puntajes de anomalía.

## Dataset

El loader ACS usa los archivos:

- dataset/psam_husa.csv
- dataset/psam_husb.csv

Ambos tienen el mismo esquema y se concatenan automáticamente.

## Arquitectura

La arquitectura por defecto sigue la forma de cuello de botella pedida por la rúbrica:

- Encoder: input -> 128 -> 64 -> 16
- Decoder: 16 -> 64 -> 128 -> output

El VAE usa la misma base del encoder y aprende media/logvar para aplicar el truco de reparametrización.

## Entrenamiento y Puntajes

- AE: se optimiza MSE de reconstrucción.
- VAE: se optimiza ELBO = reconstrucción + beta * KL.
- Isolation Forest: se calcula score para todas las muestras (sin eliminar filas).

Salida principal:

- Correlación Spearman entre rankings:
  - iforest_vs_ae
  - iforest_vs_vae
  - ae_vs_vae
- Resumen de score por método (mean, std, p95, p99).

La ejecución normal del runner también guarda dos figuras:

- ae_latent_space.png
- vae_latent_space.png

Cada una incluye UMAP y t-SNE del latente correspondiente, coloreados por anomalía e intervalos de precio.

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

## Ejecución rápida

```bash
challenge6-train --max-rows 50000
```

## Comparación de variantes (latente)

```bash
challenge6-train --max-rows 50000 --compare-variants
```

## Ejecución con CSV propio

```bash
challenge6-train --dataset csv --csv-path data\mi_dataset.csv --drop-columns id serial
```