from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingPackConfig:
    source_manifest: str = ""
    output_root: str = ""
    tag: str = ""
    train_start_year: int = 2012
    train_end_year: int = 2023
    validation_year: int = 2024
    test_year: int = 2025
    max_samples_per_role: int = 0
    max_samples_per_date_per_role: int = 0
    feature_dtype: str = "float16"
    stock_chunk_size: int = 64
    resume: bool = True


def build_training_pack(config: TrainingPackConfig) -> dict[str, Any]:
    from daily_research.path_policy.forecast_dataset import build_qdp_training_pack

    source_manifest = Path(str(config.source_manifest or "")).expanduser()
    if not str(source_manifest):
        raise ValueError("--source-manifest is required for build-training-pack.")
    output_root = Path(str(config.output_root)).expanduser() if str(config.output_root or "").strip() else None
    return build_qdp_training_pack(
        source_manifest,
        output_root=output_root,
        tag=str(config.tag or ""),
        train_start_year=int(config.train_start_year),
        train_end_year=int(config.train_end_year),
        validation_year=int(config.validation_year),
        test_year=int(config.test_year),
        max_samples_per_role=int(config.max_samples_per_role),
        max_samples_per_date_per_role=int(config.max_samples_per_date_per_role),
        feature_dtype=str(config.feature_dtype or "float16"),
        stock_chunk_size=int(config.stock_chunk_size),
        resume=bool(config.resume),
    )
