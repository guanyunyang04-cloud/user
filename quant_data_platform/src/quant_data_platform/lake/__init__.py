from __future__ import annotations

from importlib import import_module
from typing import Any

from quant_data_platform.lake.catalog import (
    DEFAULT_DATA_LAKE_ROOT,
    DATA_LAKE_SCHEMA_VERSION,
    LakeDatasetRecord,
    LakeTrainingDatasetRecord,
    ResearchDataLake,
    build_label_completeness_summary,
)

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "audit_gold_dataset": ("quant_data_platform.lake.audit_gold_dataset", "audit_gold_dataset"),
    "audit_policy_input_bundle": ("quant_data_platform.lake.policy_input_audit", "audit_policy_input_bundle"),
    "build_gold_training_dataset": ("quant_data_platform.lake.build_gold_training_dataset", None),
    "build_pool_view": ("quant_data_platform.lake.build_pool_view", None),
    "build_research_database": ("quant_data_platform.lake.build_research_database", None),
    "build_sector_board_view": ("quant_data_platform.lake.build_sector_board_view", None),
    "build_sharded_gold_training_dataset": (
        "quant_data_platform.lake.gold_training_builder",
        "build_sharded_gold_training_dataset",
    ),
    "build_lake_inventory": ("quant_data_platform.lake.canonical", "build_lake_inventory"),
    "load_canonical_manifest": ("quant_data_platform.lake.canonical", "load_canonical_manifest"),
    "resolve_canonical_dataset_id": ("quant_data_platform.lake.canonical", "resolve_canonical_dataset_id"),
    "write_canonical_manifest": ("quant_data_platform.lake.canonical", "write_canonical_manifest"),
    "GoldBuildSpec": ("quant_data_platform.lake.gold_training_builder", "GoldBuildSpec"),
    "DEFAULT_POLICY_INPUT_LAKE_DATASET_ID": (
        "quant_data_platform.lake.policy_input_loader",
        "DEFAULT_POLICY_INPUT_LAKE_DATASET_ID",
    ),
    "load_policy_inputs_from_lake": (
        "quant_data_platform.lake.policy_input_loader",
        "load_policy_inputs_from_lake",
    ),
    "resolve_policy_input_dataset_id": (
        "quant_data_platform.lake.policy_input_loader",
        "resolve_policy_input_dataset_id",
    ),
    "PoolViewRecord": ("quant_data_platform.lake.pool_views", "PoolViewRecord"),
    "PoolViewSpec": ("quant_data_platform.lake.pool_views", "PoolViewSpec"),
    "build_pool_view_from_policy_bundle": (
        "quant_data_platform.lake.pool_views",
        "build_pool_view_from_policy_bundle",
    ),
    "load_pool_view": ("quant_data_platform.lake.pool_views", "load_pool_view"),
    "resolve_pool_view_for_policy_inputs": (
        "quant_data_platform.lake.pool_views",
        "resolve_pool_view_for_policy_inputs",
    ),
    "SectorBoardViewRecord": ("quant_data_platform.lake.sector_board_views", "SectorBoardViewRecord"),
    "SectorBoardViewSpec": ("quant_data_platform.lake.sector_board_views", "SectorBoardViewSpec"),
    "build_sector_board_view_from_policy_bundle": (
        "quant_data_platform.lake.sector_board_views",
        "build_sector_board_view_from_policy_bundle",
    ),
    "load_sector_board_view": ("quant_data_platform.lake.sector_board_views", "load_sector_board_view"),
    "resolve_sector_board_view_for_policy_inputs": (
        "quant_data_platform.lake.sector_board_views",
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
