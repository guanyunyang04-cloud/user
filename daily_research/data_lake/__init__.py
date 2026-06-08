from __future__ import annotations

from importlib import import_module
from typing import Any

from daily_research.data_lake.catalog import (
    DEFAULT_DATA_LAKE_ROOT,
    DATA_LAKE_SCHEMA_VERSION,
    LakeDatasetRecord,
    LakeTrainingDatasetRecord,
    ResearchDataLake,
    build_label_completeness_summary,
)

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "audit_gold_dataset": ("daily_research.data_lake.audit_gold_dataset", "audit_gold_dataset"),
    "audit_policy_input_bundle": ("daily_research.data_lake.policy_input_audit", "audit_policy_input_bundle"),
    "build_gold_training_dataset": ("daily_research.data_lake.build_gold_training_dataset", None),
    "build_pool_view": ("daily_research.data_lake.build_pool_view", None),
    "build_research_database": ("daily_research.data_lake.build_research_database", None),
    "build_sector_board_view": ("daily_research.data_lake.build_sector_board_view", None),
    "build_sharded_gold_training_dataset": (
        "daily_research.data_lake.gold_training_builder",
        "build_sharded_gold_training_dataset",
    ),
    "GoldBuildSpec": ("daily_research.data_lake.gold_training_builder", "GoldBuildSpec"),
    "DEFAULT_POLICY_INPUT_LAKE_DATASET_ID": (
        "daily_research.data_lake.policy_input_loader",
        "DEFAULT_POLICY_INPUT_LAKE_DATASET_ID",
    ),
    "load_policy_inputs_from_lake": (
        "daily_research.data_lake.policy_input_loader",
        "load_policy_inputs_from_lake",
    ),
    "PoolViewRecord": ("daily_research.data_lake.pool_views", "PoolViewRecord"),
    "PoolViewSpec": ("daily_research.data_lake.pool_views", "PoolViewSpec"),
    "build_pool_view_from_policy_bundle": (
        "daily_research.data_lake.pool_views",
        "build_pool_view_from_policy_bundle",
    ),
    "load_pool_view": ("daily_research.data_lake.pool_views", "load_pool_view"),
    "resolve_pool_view_for_policy_inputs": (
        "daily_research.data_lake.pool_views",
        "resolve_pool_view_for_policy_inputs",
    ),
    "SectorBoardViewRecord": ("daily_research.data_lake.sector_board_views", "SectorBoardViewRecord"),
    "SectorBoardViewSpec": ("daily_research.data_lake.sector_board_views", "SectorBoardViewSpec"),
    "build_sector_board_view_from_policy_bundle": (
        "daily_research.data_lake.sector_board_views",
        "build_sector_board_view_from_policy_bundle",
    ),
    "load_sector_board_view": ("daily_research.data_lake.sector_board_views", "load_sector_board_view"),
    "resolve_sector_board_view_for_policy_inputs": (
        "daily_research.data_lake.sector_board_views",
        "resolve_sector_board_view_for_policy_inputs",
    ),
}

__all__ = [
    "DATA_LAKE_SCHEMA_VERSION",
    "DEFAULT_DATA_LAKE_ROOT",
    "LakeDatasetRecord",
    "LakeTrainingDatasetRecord",
    "ResearchDataLake",
    "build_label_completeness_summary",
    *_LAZY_EXPORTS.keys(),
]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value
