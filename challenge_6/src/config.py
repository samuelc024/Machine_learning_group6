from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetConfig:
    kind: str = "acs_housing"
    random_state: int = 42
    csv_path: str | None = None
    target_column: str | None = None
    drop_columns: tuple[str, ...] = ()
    max_rows: int | None = None


@dataclass(frozen=True)
class PreprocessingConfig:
    scale: bool = True


@dataclass(frozen=True)
class ModelConfig:
    iforest_contamination: float = 0.03
    ae_hidden_dims: tuple[int, int] = (128, 64)
    ae_latent_dim: int = 16
    ae_epochs: int = 25
    vae_epochs: int = 35
    batch_size: int = 512
    learning_rate: float = 1e-3
    vae_beta: float = 1.0


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    # Directorio donde guardar artefactos de experimentos (modelos, métricas).
    # Si es None, no se guardarán modelos automáticamente.
    output_dir: str | None = "runs/modelo_final"
    # Controla si se deben guardar los modelos entrenados (.pt).
    save_models: bool = False