from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_data_platform.core.json_io import utc_now, write_json
from quant_data_platform.core.paths import QdpPaths, qdp_paths
from quant_data_platform.features.profiles import DEFAULT_PROFILE


def write_sharded_memmap_plan(
    paths: QdpPaths | None = None,
    *,
    profile: str = DEFAULT_PROFILE,
    max_universe_size: int = 0,
    write: bool = True,
) -> dict[str, Any]:
    resolved = paths or qdp_paths()
    payload = {
        "status": "plan_only",
        "artifact_type": "sharded_memmap_build_plan",
        "profile": str(profile or DEFAULT_PROFILE),
        "max_universe_size": int(max_universe_size or 0),
        "shard_policy": "year_or_year_symbol_block_feature_label_shards",
        "memory_policy": "never_build_full_universe_single_process_memmap",
        "created_at": utc_now(),
    }
    if write:
        path = resolved.memmap_dir / "sharded_memmap_plan.json"
        write_json(path, payload)
        payload["plan_json"] = str(path.as_posix())
    return payload
