from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DatasetConfig, ExperimentConfig, ModelConfig, PreprocessingConfig
from .training import compare_pipeline_variants, run_experiment
from .visualization import save_latent_visualizations


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Challenge 6 unsupervised anomaly runner")
    parser.add_argument("--dataset", default="acs_housing", choices=["acs_housing", "csv"])
    parser.add_argument("--csv-path")
    parser.add_argument("--target-column")
    parser.add_argument("--drop-columns", nargs="*", default=[])
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-scale", action="store_true")

    parser.add_argument("--iforest-contamination", type=float, default=0.03)
    parser.add_argument("--ae-hidden-dims", nargs=2, type=int, default=[128, 64])
    parser.add_argument("--ae-latent-dim", type=int, default=16)
    parser.add_argument("--ae-epochs", type=int, default=25)
    parser.add_argument("--vae-epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--vae-beta", type=float, default=1.0)

    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--compare-variants", action="store_true")
    parser.add_argument("--analyze-results", type=str, help="Ruta al archivo JSON para realizar el post-análisis")
    return parser


def _build_config(args: argparse.Namespace) -> ExperimentConfig:
    dataset = DatasetConfig(
        kind=args.dataset,
        random_state=args.random_state,
        csv_path=args.csv_path,
        target_column=args.target_column,
        drop_columns=tuple(args.drop_columns),
        max_rows=args.max_rows,
    )
    preprocessing = PreprocessingConfig(scale=not args.no_scale)
    model = ModelConfig(
        iforest_contamination=args.iforest_contamination,
        ae_hidden_dims=(args.ae_hidden_dims[0], args.ae_hidden_dims[1]),
        ae_latent_dim=args.ae_latent_dim,
        ae_epochs=args.ae_epochs,
        vae_epochs=args.vae_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        vae_beta=args.vae_beta,
    )
    return ExperimentConfig(dataset=dataset, preprocessing=preprocessing, model=model)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _result_row(result) -> dict[str, object]:
    row: dict[str, object] = {
        "rows": result.rows,
        "features": result.feature_count,
        "iforest_vs_ae": result.correlations["iforest_vs_ae"],
        "iforest_vs_vae": result.correlations["iforest_vs_vae"],
        "ae_vs_vae": result.correlations["ae_vs_vae"],
        "ae_latent_dim": result.config.model.ae_latent_dim,
        "dataset": result.config.dataset.kind,
    }
    row.update({
        "iforest_mean": result.score_summaries["iforest"]["mean"],
        "ae_mse_mean": result.score_summaries["ae_reconstruction_mse"]["mean"],
        "vae_elbo_mean": result.score_summaries["vae_elbo"]["mean"],
    })
    return row


def _save_plots(result, output_dir: Path) -> dict[str, str]:
    plot_paths: dict[str, str] = {}

    ae_path = save_latent_visualizations(
        latent_name="ae",
        latent_matrix=result.ae_latent_z,
        anomaly_scores=result.iforest_scores,
        price_values=result.price_values,
        output_dir=output_dir,
        random_state=result.config.dataset.random_state,
    )
    if ae_path is not None:
        plot_paths["ae"] = str(ae_path)

    vae_path = save_latent_visualizations(
        latent_name="vae",
        latent_matrix=result.vae_latent_mu,
        anomaly_scores=result.iforest_scores,
        price_values=result.price_values,
        output_dir=output_dir,
        random_state=result.config.dataset.random_state,
    )
    if vae_path is not None:
        plot_paths["vae"] = str(vae_path)

    return plot_paths


def _write_single_result(result, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _timestamp()
    csv_path = output_dir / f"unsupervised_metrics_{timestamp}.csv"
    json_path = output_dir / f"unsupervised_metrics_{timestamp}.json"

    row = _result_row(result)
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"result": row, "full_result": _json_safe(asdict(result))}, handle, indent=2, ensure_ascii=False)

    return csv_path


def _write_comparison_results(results, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _timestamp()
    csv_path = output_dir / f"unsupervised_comparison_{timestamp}.csv"
    frame = pd.DataFrame([_result_row(result) for result in results])
    frame.to_csv(csv_path, index=False)
    return csv_path


def _print_result(result) -> None:
    printable = {
        "rows": result.rows,
        "features": result.feature_count,
        "iforest_vs_ae": result.correlations["iforest_vs_ae"],
        "iforest_vs_vae": result.correlations["iforest_vs_vae"],
        "ae_vs_vae": result.correlations["ae_vs_vae"],
        "iforest_mean": result.score_summaries["iforest"]["mean"],
        "ae_mse_mean": result.score_summaries["ae_reconstruction_mse"]["mean"],
        "vae_elbo_mean": result.score_summaries["vae_elbo"]["mean"],
    }
    print(pd.Series(printable).to_string())


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = _build_config(args)
    output_dir = Path(args.output_dir)
    if args.analyze_results:
        # use relative import to work in local package layouts
        from .analysis import run_post_analysis
        run_post_analysis(Path(args.analyze_results), config)
        return 0
    if args.compare_variants:
        results = compare_pipeline_variants(config)
        metrics_path = _write_comparison_results(results, output_dir)
        frame = pd.DataFrame([_result_row(result) for result in results])
        print(frame.to_string(index=False))
        print(f"saved_metrics={metrics_path}")
        return 0

    result = run_experiment(config)
    plot_paths = _save_plots(result, output_dir)
    _print_result(result)
    metrics_path = _write_single_result(result, output_dir)
    if plot_paths:
        print(pd.Series({f"plot_{name}": path for name, path in plot_paths.items()}).to_string())
    print(f"saved_metrics={metrics_path}")
    return 0