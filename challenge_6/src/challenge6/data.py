from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from .config import DatasetConfig


@dataclass(frozen=True)
class DatasetBundle:
    features: pd.DataFrame
    target: pd.Series | None
    feature_names: list[str]
    target_name: str | None


class DatasetLoader(Protocol):
    def load(self) -> DatasetBundle:
        ...


ACS_HOUSING_SOURCE_FILES = ("psam_husa.csv", "psam_husb.csv")
ACS_HOUSING_TARGET_COLUMN = "HINCP"
ACS_HOUSING_DROP_COLUMNS = ("RT", "SERIALNO", "WGTP") + tuple(f"WGTP{i}" for i in range(1, 81))


def _project_dataset_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "dataset"


def _bundle_from_frame(frame: pd.DataFrame, target_column: str | None) -> DatasetBundle:
    target: pd.Series | None = None
    target_name: str | None = None

    if target_column is not None:
        if target_column not in frame.columns:
            raise ValueError(f"Target column '{target_column}' was not found in the dataset.")
        target = pd.to_numeric(frame[target_column], errors="coerce")
        valid_rows = target.notna()
        frame = frame.loc[valid_rows].reset_index(drop=True)
        target = target.loc[valid_rows].reset_index(drop=True)
        target_name = target_column

    features = frame.drop(columns=[target_column]) if target_column is not None else frame.copy()
    features = features.dropna(axis=1, how="all")
    return DatasetBundle(
        features=features,
        target=target,
        feature_names=list(features.columns),
        target_name=target_name,
    )


@dataclass(frozen=True)
class AcsHousingLoader:
    data_dir: Path | None = None
    max_rows: int | None = None
    target_column: str | None = None
    drop_columns: tuple[str, ...] = ACS_HOUSING_DROP_COLUMNS

    def _load_source(self, path: Path, row_limit: int | None = None) -> pd.DataFrame:
        drop_columns = set(self.drop_columns)
        return pd.read_csv(path, usecols=lambda column: column not in drop_columns, nrows=row_limit)

    def load(self) -> DatasetBundle:
        base_dir = self.data_dir or _project_dataset_dir()
        source_paths = [base_dir / file_name for file_name in ACS_HOUSING_SOURCE_FILES]

        if self.max_rows is None:
            frames = [self._load_source(path) for path in source_paths]
        else:
            sample_rows = max(self.max_rows * 10, self.max_rows)
            base_rows, remainder = divmod(sample_rows, len(source_paths))
            frames = []
            for index, path in enumerate(source_paths):
                row_limit = base_rows + (1 if index < remainder else 0)
                frames.append(self._load_source(path, row_limit=row_limit))

        frame = pd.concat(frames, ignore_index=True)
        bundle = _bundle_from_frame(frame, self.target_column)

        if self.max_rows is not None and len(bundle.features) > self.max_rows:
            features = bundle.features.head(self.max_rows).reset_index(drop=True)
            target = None if bundle.target is None else bundle.target.head(self.max_rows).reset_index(drop=True)
            return DatasetBundle(
                features=features,
                target=target,
                feature_names=list(features.columns),
                target_name=bundle.target_name,
            )

        return bundle


@dataclass(frozen=True)
class CsvTabularLoader:
    csv_path: Path
    target_column: str
    drop_columns: tuple[str, ...] = ()
    max_rows: int | None = None

    def load(self) -> DatasetBundle:
        frame = pd.read_csv(self.csv_path)
        if self.max_rows is not None:
            frame = frame.head(self.max_rows).copy()

        frame = frame.drop(columns=[column for column in self.drop_columns if column in frame.columns])
        return _bundle_from_frame(frame, self.target_column)


def build_dataset_loader(config: DatasetConfig) -> DatasetLoader:
    if config.kind == "acs_housing":
        drop_columns = ACS_HOUSING_DROP_COLUMNS + config.drop_columns
        return AcsHousingLoader(max_rows=config.max_rows, target_column=config.target_column, drop_columns=drop_columns)

    if config.kind == "csv":
        if config.csv_path is None:
            raise ValueError("csv_path is required when dataset kind is 'csv'.")
        if config.target_column is None:
            raise ValueError("target_column is required when dataset kind is 'csv'.")
        return CsvTabularLoader(
            csv_path=Path(config.csv_path),
            target_column=config.target_column,
            drop_columns=config.drop_columns,
            max_rows=config.max_rows,
        )

    raise ValueError(f"Unsupported dataset kind: {config.kind}")