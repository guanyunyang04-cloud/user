from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from daily_research.path_policy.qdp_v2_sequence_path_pack import validate_sequence_pack


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESEARCH_STORE_ROOT = Path("daily_research/data/research_store")
DEFAULT_LEGACY_SEQUENCE_PACK_ROOT = DEFAULT_RESEARCH_STORE_ROOT / "sequence_pack"
DEFAULT_RETENTION_POLICY = Path("daily_research/brain/research_store_retention_policy.json")
DEFAULT_PANEL_STORE_ID = "qdp_v2_mainboard_2012_2025_v1"
DELETE_CONFIRMATION = "DELETE_REPLACED_SEQUENCE_PACKS"
LEGACY_PACK_NAMES = (
    "qdp_v2_seq100_path60_full",
    "qdp_v2_seq100_ohlcva_path60_full",
    "qdp_v2_seq100_path60_todayclose_full",
    "qdp_v2_seq100_path20_full",
)


def _workspace_path(path: str | Path) -> Path:
    raw = Path(str(path))
    return raw if raw.is_absolute() else WORKSPACE_ROOT / raw


def _relative(path: str | Path) -> str:
    resolved = _workspace_path(path)
    try:
        return resolved.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    except Exception:
        return resolved.as_posix()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_workspace_path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = _workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return target


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _workspace_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_research_store_retention_policy(
    policy_path: str | Path = DEFAULT_RETENTION_POLICY,
) -> dict[str, Any]:
    path = _workspace_path(policy_path)
    if not path.exists():
        return {}
    payload = _read_json(path)
    if payload.get("artifact_type") not in {None, "research_store_retention_policy"}:
        raise ValueError(f"not a research store retention policy: {path}")
    return payload


def discover_research_store_views(
    store_root: str | Path = DEFAULT_RESEARCH_STORE_ROOT,
) -> dict[str, Path]:
    store_root_path = _workspace_path(store_root)
    views_root = store_root_path / "views"
    discovered: dict[str, Path] = {}
    if not views_root.exists():
        return discovered
    for path in sorted(views_root.glob("*.json")):
        payload = _read_json(path)
        view_id = str(dict(payload.get("artifact_view", {}) or {}).get("view_id", "") or path.stem)
        if view_id in discovered:
            raise ValueError(f"duplicate research store view_id={view_id}: {path} and {discovered[view_id]}")
        discovered[view_id] = path
    return discovered


def _store_relative(path: str | Path, *, store_root: Path) -> str:
    resolved = _workspace_path(path).resolve()
    try:
        return resolved.relative_to(store_root.resolve()).as_posix()
    except ValueError:
        return _relative(resolved)


def _component_id(path: str | Path, *, store_root: Path) -> str:
    resolved = _workspace_path(path).resolve()
    try:
        relative = resolved.relative_to(store_root.resolve())
    except ValueError:
        return ""
    parts = relative.parts
    if not parts:
        return ""
    family = parts[0]
    if family in {"panel_store", "label_store", "sharded_memmap", "training_pack", "sequence_pack"}:
        return "/".join(parts[:2]) if len(parts) >= 2 else family
    if family in {"sample_index", "views"}:
        return relative.as_posix()
    return "/".join(parts[:2]) if len(parts) >= 2 else family


def _runtime_paths_from_view(view: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    for section in ("feature_channels", "label_arrays", "masks"):
        for meta in dict(view.get(section, {}) or {}).values():
            if not isinstance(meta, Mapping):
                continue
            if meta.get("path"):
                paths.append(str(meta["path"]))
            for shard in list(meta.get("shards", []) or []):
                if isinstance(shard, Mapping) and shard.get("path"):
                    paths.append(str(shard["path"]))
    if view.get("sample_index_path"):
        paths.append(str(view["sample_index_path"]))
    artifact_view = dict(view.get("artifact_view", {}) or {})
    for key in ("panel_store_manifest", "source_view"):
        if artifact_view.get(key):
            paths.append(str(artifact_view[key]))
    for value in dict(artifact_view.get("label_sources", {}) or {}).values():
        if value:
            paths.append(str(value))
    return paths


def build_active_view_dependency_graph(
    *,
    store_root: str | Path = DEFAULT_RESEARCH_STORE_ROOT,
    policy_path: str | Path = DEFAULT_RETENTION_POLICY,
    active_view_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    store_root_path = _workspace_path(store_root)
    discovered = discover_research_store_views(store_root_path)
    policy = load_research_store_retention_policy(policy_path)
    selected_ids = sorted({str(item) for item in list(active_view_ids or policy.get("active_view_ids", []) or [])})
    if not selected_ids:
        selected_ids = sorted(discovered)
    views: dict[str, Any] = {}
    active_components: set[str] = set()
    missing_views: list[str] = []
    missing_runtime_paths: list[dict[str, str]] = []
    for view_id in selected_ids:
        view_path = discovered.get(view_id)
        if view_path is None:
            missing_views.append(view_id)
            continue
        payload = _read_json(view_path)
        components = {_component_id(view_path, store_root=store_root_path)}
        runtime_paths: list[dict[str, Any]] = []
        for raw_path in _runtime_paths_from_view(payload):
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = WORKSPACE_ROOT / candidate
            component = _component_id(candidate, store_root=store_root_path)
            if component:
                components.add(component)
            exists = candidate.exists()
            runtime_paths.append(
                {
                    "path": _relative(candidate),
                    "component_id": component,
                    "exists": exists,
                }
            )
            if not exists:
                missing_runtime_paths.append({"view_id": view_id, "path": _relative(candidate)})
        components.discard("")
        active_components.update(components)
        views[view_id] = {
            "path": _relative(view_path),
            "sha256": _sha256_file(view_path),
            "components": sorted(components),
            "runtime_path_count": len(runtime_paths),
        }
    return {
        "schema_version": 1,
        "artifact_type": "research_store_active_view_dependency_graph",
        "store_root": _relative(store_root_path),
        "policy_path": _relative(policy_path) if _workspace_path(policy_path).exists() else "",
        "active_view_ids": selected_ids,
        "views": views,
        "active_component_ids": sorted(active_components),
        "missing_views": missing_views,
        "missing_runtime_paths": missing_runtime_paths,
        "inactive_disk_view_ids": sorted(set(discovered) - set(selected_ids)),
    }


def build_research_store_index(
    *,
    store_root: str | Path = DEFAULT_RESEARCH_STORE_ROOT,
    policy_path: str | Path = DEFAULT_RETENTION_POLICY,
    active_view_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    store_root_path = _workspace_path(store_root)
    graph = build_active_view_dependency_graph(
        store_root=store_root_path,
        policy_path=policy_path,
        active_view_ids=active_view_ids,
    )
    discovered = discover_research_store_views(store_root_path)
    views: dict[str, Any] = {}
    for view_id in graph["active_view_ids"]:
        path = discovered.get(str(view_id))
        if path is None:
            continue
        payload = _read_json(path)
        artifact_view = dict(payload.get("artifact_view", {}) or {})
        views[str(view_id)] = {
            "path": _store_relative(path, store_root=store_root_path),
            "sha256": _sha256_file(path),
            "view_type": str(artifact_view.get("view_type", "research_store_view") or "research_store_view"),
        }
    index = {
        "schema_version": 2,
        "artifact_type": "qdp_v2_research_store_index",
        "created_at": _now(),
        "store_root": _relative(store_root_path),
        "retention_policy": _relative(policy_path) if _workspace_path(policy_path).exists() else "",
        "active_view_ids": list(graph["active_view_ids"]),
        "views": views,
        "active_component_ids": list(graph["active_component_ids"]),
        "inactive_disk_view_ids": list(graph["inactive_disk_view_ids"]),
    }
    index_path = _write_json(store_root_path / "research_store_index.json", index)
    return {**index, "index_path": _relative(index_path)}


def _hardlink_file(source: str | Path, target: str | Path, *, overwrite: bool = False) -> Path:
    src = _workspace_path(source).resolve()
    dst = _workspace_path(target).resolve()
    if not src.exists() and dst.exists() and dst.is_file():
        return dst
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not overwrite:
            return dst
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.move(str(src), str(dst))
    return dst


def _link_memmap_meta(meta: Mapping[str, Any], target_dir: Path, *, overwrite: bool = False) -> dict[str, Any]:
    out = dict(meta)
    if list(meta.get("shards", []) or []):
        shards: list[dict[str, Any]] = []
        shard_dir = target_dir / str(Path(str(meta.get("name", "array"))).name or "shards")
        for shard in list(meta.get("shards", []) or []):
            shard_payload = dict(shard)
            source_path = _workspace_path(str(shard_payload["path"]))
            target_path = _hardlink_file(source_path, shard_dir / source_path.name, overwrite=overwrite)
            shard_payload["path"] = str(target_path.resolve())
            shards.append(shard_payload)
        out["shards"] = shards
        out.pop("path", None)
        return out
    source = _workspace_path(str(meta["path"]))
    target = _hardlink_file(source, target_dir / source.name, overwrite=overwrite)
    out["path"] = str(target.resolve())
    return out


def _load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest.get("artifact_type") != "qdp_v2_sequence_path_pack":
        raise ValueError(f"not a qdp_v2 sequence path pack manifest: {path}")
    return manifest


def _pack_manifest(root: Path, name: str) -> Path:
    return _workspace_path(root / name / "manifest.json")


def _build_panel_store(
    *,
    source_manifest: Mapping[str, Any],
    store_root: Path,
    store_id: str,
    overwrite: bool,
) -> dict[str, Any]:
    panel_root = store_root / "panel_store" / store_id
    feature_channels: dict[str, Any] = {}
    for name, meta in dict(source_manifest.get("feature_channels", {}) or {}).items():
        feature_channels[str(name)] = _link_memmap_meta(meta, panel_root / "panels", overwrite=overwrite)
    masks: dict[str, Any] = {}
    for name, meta in dict(source_manifest.get("masks", {}) or {}).items():
        masks[str(name)] = _link_memmap_meta(meta, panel_root / "masks", overwrite=overwrite)
    payload = {
        "schema_version": 1,
        "artifact_type": "qdp_v2_research_panel_store",
        "store_id": store_id,
        "created_at": _now(),
        "source_manifest": source_manifest.get("manifest_path", ""),
        "date_values": source_manifest.get("date_values", []),
        "symbol_values": source_manifest.get("symbol_values", []),
        "date_count": int(source_manifest.get("date_count", 0) or 0),
        "symbol_count": int(source_manifest.get("symbol_count", 0) or 0),
        "feature_channels": feature_channels,
        "masks": masks,
        "normalization": source_manifest.get("normalization", {}),
        "policy": "shared inputs; experiment views choose lookback, labels and value semantics",
    }
    manifest_path = _write_json(panel_root / "panel_store_manifest.json", payload)
    payload["manifest_path"] = str(manifest_path.resolve())
    _write_json(manifest_path, payload)
    return payload


def _link_label_store(
    *,
    source_manifest: Mapping[str, Any],
    label_names: list[str],
    store_root: Path,
    label_store_id: str,
    overwrite: bool,
) -> dict[str, Any]:
    target_root = store_root / "label_store" / label_store_id
    labels: dict[str, Any] = {}
    source_labels = dict(source_manifest.get("label_arrays", {}) or {})
    for name in label_names:
        if name not in source_labels:
            continue
        labels[name] = _link_memmap_meta(source_labels[name], target_root / name, overwrite=overwrite)
    payload = {
        "schema_version": 1,
        "artifact_type": "qdp_v2_research_label_store",
        "label_store_id": label_store_id,
        "created_at": _now(),
        "source_manifest": source_manifest.get("manifest_path", ""),
        "label_arrays": labels,
    }
    manifest_path = _write_json(target_root / "label_store_manifest.json", payload)
    payload["manifest_path"] = str(manifest_path.resolve())
    _write_json(manifest_path, payload)
    return payload


def _path20_sample_index_on_panel_store(path20_manifest: Mapping[str, Any], panel_store: Mapping[str, Any], target_path: Path) -> Path:
    panel_symbols = list(panel_store.get("symbol_values", []) or [])
    symbol_to_idx = {str(symbol): idx for idx, symbol in enumerate(panel_symbols)}
    frame = pd.read_parquet(str(path20_manifest["sample_index_path"]))
    missing = sorted(set(frame["symbol"].astype(str)) - set(symbol_to_idx))
    if missing:
        raise ValueError(f"path20 symbols missing from panel store: {missing[:10]}")
    frame = frame.copy()
    frame["source_symbol_idx"] = frame["symbol_idx"].astype("int32")
    frame["label_symbol_idx"] = frame["source_symbol_idx"].astype("int32")
    frame["symbol_idx"] = frame["symbol"].astype(str).map(symbol_to_idx).astype("int32")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target_path, index=False)
    return target_path


def _view_manifest(
    *,
    source_manifest: Mapping[str, Any],
    view_id: str,
    panel_store: Mapping[str, Any],
    labels: Mapping[str, Any],
    sample_index_path: Path,
    forward_days: int,
    store_root: Path,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    split_counts = dict(source_manifest.get("sample_count_by_split", {}) or {})
    payload = dict(source_manifest)
    payload.update(
        {
            "artifact_type": "qdp_v2_sequence_path_pack",
            "created_at": _now(),
            "artifact_view": {
                "schema_version": 1,
                "view_id": view_id,
                "view_type": "research_store_view",
                "panel_store_id": panel_store.get("store_id", ""),
                "panel_store_manifest": panel_store.get("manifest_path", ""),
                "label_sources": {
                    name: meta.get("path", meta.get("shards", [{}])[0].get("path", ""))
                    for name, meta in dict(labels).items()
                },
                **dict(extra or {}),
            },
            "lookback_days": int(source_manifest.get("lookback_days", 100) or 100),
            "forward_days": int(forward_days),
            "date_values": panel_store.get("date_values", []),
            "symbol_values": panel_store.get("symbol_values", []),
            "date_count": int(panel_store.get("date_count", 0) or 0),
            "symbol_count": int(panel_store.get("symbol_count", 0) or 0),
            "sample_index_path": str(sample_index_path.resolve()),
            "sample_count": int(source_manifest.get("sample_count", 0) or 0),
            "sample_count_by_split": {str(key): int(value) for key, value in split_counts.items()},
            "feature_channels": panel_store.get("feature_channels", {}),
            "masks": panel_store.get("masks", {}),
            "normalization": panel_store.get("normalization", {}),
            "label_arrays": dict(labels),
        }
    )
    target = store_root / "views" / f"{view_id}.json"
    _write_json(target, payload)
    return payload


def build_unified_store_from_legacy_packs(
    *,
    store_root: str | Path = DEFAULT_RESEARCH_STORE_ROOT,
    legacy_root: str | Path = DEFAULT_LEGACY_SEQUENCE_PACK_ROOT,
    store_id: str = DEFAULT_PANEL_STORE_ID,
    overwrite: bool = False,
) -> dict[str, Any]:
    store_root_path = _workspace_path(store_root)
    legacy_root_path = _workspace_path(legacy_root)
    path60_manifest_path = _pack_manifest(legacy_root_path, "qdp_v2_seq100_path60_full")
    ohlcva_manifest_path = _pack_manifest(legacy_root_path, "qdp_v2_seq100_ohlcva_path60_full")
    todayclose_manifest_path = _pack_manifest(legacy_root_path, "qdp_v2_seq100_path60_todayclose_full")
    path20_manifest_path = _pack_manifest(legacy_root_path, "qdp_v2_seq100_path20_full")
    manifests = {
        "path60": _load_manifest(path60_manifest_path),
        "ohlcva": _load_manifest(ohlcva_manifest_path),
        "todayclose": _load_manifest(todayclose_manifest_path),
        "path20": _load_manifest(path20_manifest_path),
    }
    for key, manifest in manifests.items():
        manifest["manifest_path"] = str(
            {
                "path60": path60_manifest_path,
                "ohlcva": ohlcva_manifest_path,
                "todayclose": todayclose_manifest_path,
                "path20": path20_manifest_path,
            }[key].resolve()
        )
    panel_store = _build_panel_store(
        source_manifest=manifests["path60"],
        store_root=store_root_path,
        store_id=store_id,
        overwrite=overwrite,
    )
    sample_dir = store_root_path / "sample_index"
    path60_sample = _hardlink_file(
        manifests["path60"]["sample_index_path"],
        sample_dir / "seq100_path60_train2012_2023_val2024_test2025.parquet",
        overwrite=overwrite,
    )
    path20_sample = _path20_sample_index_on_panel_store(
        manifests["path20"],
        panel_store,
        sample_dir / f"seq100_path20_train2012_2023_val2024_test2025_on_{store_id}.parquet",
    )
    labels_path60_ohlc = _link_label_store(
        source_manifest=manifests["path60"],
        label_names=["future_ohlc_path", "path_summary"],
        store_root=store_root_path,
        label_store_id="path60_nextopen_ohlc_v1",
        overwrite=overwrite,
    )
    labels_path60_ohlcva = _link_label_store(
        source_manifest=manifests["ohlcva"],
        label_names=["future_ohlcva_path", "path_summary"],
        store_root=store_root_path,
        label_store_id="path60_nextopen_ohlcva_v1",
        overwrite=overwrite,
    )
    labels_todayclose = _link_label_store(
        source_manifest=manifests["todayclose"],
        label_names=["future_ohlcva_path", "path_summary"],
        store_root=store_root_path,
        label_store_id="path60_todayclose_ohlcva_v1",
        overwrite=overwrite,
    )
    labels_path20_ohlc = _link_label_store(
        source_manifest=manifests["path20"],
        label_names=["future_ohlc_path"],
        store_root=store_root_path,
        label_store_id="path20_nextopen_ohlc_v1",
        overwrite=overwrite,
    )
    views = {
        "seq100_path60_nextopen_ohlc": _view_manifest(
            source_manifest=manifests["path60"],
            view_id="seq100_path60_nextopen_ohlc",
            panel_store=panel_store,
            labels=labels_path60_ohlc["label_arrays"],
            sample_index_path=path60_sample,
            forward_days=60,
            store_root=store_root_path,
            extra={"replaces_legacy_pack": "qdp_v2_seq100_path60_full"},
        ),
        "seq100_path60_nextopen_ohlcva": _view_manifest(
            source_manifest=manifests["ohlcva"],
            view_id="seq100_path60_nextopen_ohlcva",
            panel_store=panel_store,
            labels=labels_path60_ohlcva["label_arrays"],
            sample_index_path=path60_sample,
            forward_days=60,
            store_root=store_root_path,
            extra={"replaces_legacy_pack": "qdp_v2_seq100_ohlcva_path60_full"},
        ),
        "seq100_path60_todayclose_ohlcva": _view_manifest(
            source_manifest=manifests["todayclose"],
            view_id="seq100_path60_todayclose_ohlcva",
            panel_store=panel_store,
            labels=labels_todayclose["label_arrays"],
            sample_index_path=path60_sample,
            forward_days=60,
            store_root=store_root_path,
            extra={"replaces_legacy_pack": "qdp_v2_seq100_path60_todayclose_full"},
        ),
        "seq100_path20_nextopen_ohlc_from_path60": _view_manifest(
            source_manifest=manifests["path20"],
            view_id="seq100_path20_nextopen_ohlc_from_path60",
            panel_store=panel_store,
            labels={"future_ohlc_path": labels_path20_ohlc["label_arrays"]["future_ohlc_path"]},
            sample_index_path=path20_sample,
            forward_days=20,
            store_root=store_root_path,
            extra={
                "replaces_legacy_pack": "qdp_v2_seq100_path20_full",
                "label_store_id": "path20_nextopen_ohlc_v1",
                "path_summary": "derived_from_future_path_at_read_time",
                "index_semantics": "symbol_idx points to shared panel store; label_symbol_idx points to path20 label store",
            },
        ),
    }
    index = build_research_store_index(
        store_root=store_root_path,
        active_view_ids=list(views),
    )
    return {
        "schema_version": 2,
        "status": "built",
        "generated_at": _now(),
        "index_path": index["index_path"],
        "view_paths": {name: item["path"] for name, item in dict(index["views"]).items()},
        "legacy_packs_replaced": list(LEGACY_PACK_NAMES),
    }


def _open_memmap(meta: Mapping[str, Any], *, dtype: str) -> np.memmap:
    return np.memmap(str(meta["path"]), dtype=dtype, mode="r", shape=tuple(int(item) for item in meta["shape"]))


def _verify_all_legacy_views_with_path20_equivalence(
    *,
    store_root: str | Path = DEFAULT_RESEARCH_STORE_ROOT,
    legacy_root: str | Path = DEFAULT_LEGACY_SEQUENCE_PACK_ROOT,
    max_checks: int = 128,
) -> dict[str, Any]:
    store_root_path = _workspace_path(store_root)
    legacy_root_path = _workspace_path(legacy_root)
    index = _read_json(store_root_path / "research_store_index.json")
    view_paths: dict[str, Path] = {}
    for name, item in dict(index.get("views", {}) or {}).items():
        raw_path = item.get("path", "") if isinstance(item, Mapping) else item
        path = Path(str(raw_path))
        view_paths[str(name)] = path if path.is_absolute() else store_root_path / path
    for name, path in view_paths.items():
        validate_sequence_pack(path)
    path20_view = _read_json(view_paths["seq100_path20_nextopen_ohlc_from_path60"])
    path60_manifest_path = _pack_manifest(legacy_root_path, "qdp_v2_seq100_path60_full")
    path20_manifest_path = _pack_manifest(legacy_root_path, "qdp_v2_seq100_path20_full")
    if not path60_manifest_path.exists() or not path20_manifest_path.exists():
        view_index = pd.read_parquet(path20_view["sample_index_path"], columns=["symbol_idx", "label_symbol_idx"])
        view_label = _open_memmap(path20_view["label_arrays"]["future_ohlc_path"], dtype="float32")
        if int(view_index["symbol_idx"].max()) >= int(path20_view["symbol_count"]):
            raise ValueError("path20 view symbol_idx exceeds shared panel symbol_count")
        if int(view_index["label_symbol_idx"].max()) >= int(view_label.shape[1]):
            raise ValueError("path20 view label_symbol_idx exceeds path20 label symbol dimension")
        return {
            "schema_version": 1,
            "status": "ok",
            "generated_at": _now(),
            "validated_views": {name: _relative(path) for name, path in view_paths.items()},
            "path20_equivalence_checks": 0,
            "path20_label_source": "moved_label_store_no_legacy_manifest",
            "path20_sample_count": int(len(view_index)),
        }
    path60 = _load_manifest(_pack_manifest(legacy_root_path, "qdp_v2_seq100_path60_full"))
    path20 = _load_manifest(_pack_manifest(legacy_root_path, "qdp_v2_seq100_path20_full"))
    if path20["date_values"] != path60["date_values"][: len(path20["date_values"])]:
        raise ValueError("path20 date values are not a prefix of path60 date values")
    missing_symbols = sorted(set(path20["symbol_values"]) - set(path60["symbol_values"]))
    if missing_symbols:
        raise ValueError(f"path20 symbols missing from path60 panel store: {missing_symbols[:10]}")
    source_index = pd.read_parquet(path20["sample_index_path"], columns=["date_idx", "symbol_idx", "symbol", "trade_date"])
    view_index = pd.read_parquet(
        path20_view["sample_index_path"],
        columns=["date_idx", "symbol_idx", "source_symbol_idx", "label_symbol_idx", "symbol", "trade_date"],
    )
    if len(source_index) != len(view_index):
        raise ValueError(f"path20 sample count mismatch: source={len(source_index)} view={len(view_index)}")
    positions = np.linspace(0, len(source_index) - 1, num=min(max(int(max_checks), 1), len(source_index)), dtype=np.int64)
    symbol_to_idx = {str(symbol): idx for idx, symbol in enumerate(path60["symbol_values"])}
    for channel in ["daily_raw", "daily_state", "intraday_summary", "limit_structure"]:
        source_array = _open_memmap(path20["feature_channels"][channel], dtype="float32")
        view_array = _open_memmap(path20_view["feature_channels"][channel], dtype="float32")
        for pos in positions:
            source_row = source_index.iloc[int(pos)]
            view_row = view_index.iloc[int(pos)]
            date_idx = int(source_row["date_idx"])
            source_symbol_idx = int(source_row["symbol_idx"])
            view_symbol_idx = int(view_row["symbol_idx"])
            if view_symbol_idx != int(symbol_to_idx[str(source_row["symbol"])]):
                raise ValueError(f"symbol remap mismatch at row {pos}")
            source_values = np.asarray(source_array[date_idx, source_symbol_idx, :], dtype=np.float32)
            view_values = np.asarray(view_array[date_idx, view_symbol_idx, :], dtype=np.float32)
            if not np.allclose(source_values, view_values, equal_nan=True):
                raise ValueError(f"path20 feature mismatch at row={pos} channel={channel}")
    view_label = _open_memmap(path20_view["label_arrays"]["future_ohlc_path"], dtype="float32")
    source_label_path = Path(str(path20["label_arrays"]["future_ohlc_path"]["path"]))
    source_label_available = source_label_path.exists()
    if source_label_available:
        source_label = _open_memmap(path20["label_arrays"]["future_ohlc_path"], dtype="float32")
        for pos in positions:
            source_row = source_index.iloc[int(pos)]
            view_row = view_index.iloc[int(pos)]
            date_idx = int(source_row["date_idx"])
            source_symbol_idx = int(source_row["symbol_idx"])
            label_symbol_idx = int(view_row["label_symbol_idx"])
            source_values = np.asarray(source_label[date_idx, source_symbol_idx, :, :], dtype=np.float32)
            view_values = np.asarray(view_label[date_idx, label_symbol_idx, :20, :], dtype=np.float32)
            if not np.allclose(source_values, view_values, equal_nan=True):
                raise ValueError(f"path20 label mismatch at row={pos}")
    else:
        expected_shape = tuple(int(item) for item in path20["label_arrays"]["future_ohlc_path"]["shape"])
        if tuple(int(item) for item in view_label.shape) != expected_shape:
            raise ValueError(f"path20 moved label shape mismatch: expected={expected_shape} actual={view_label.shape}")
        if not Path(str(path20_view["label_arrays"]["future_ohlc_path"].get("path", "") or "")).exists():
            raise ValueError("path20 source label is missing and the moved path20 label store is unavailable")
    return {
        "schema_version": 1,
        "status": "ok",
        "generated_at": _now(),
        "validated_views": {name: _relative(path) for name, path in view_paths.items()},
        "path20_equivalence_checks": int(len(positions)),
        "path20_label_source": "legacy_source" if source_label_available else "moved_label_store",
        "path20_sample_count": int(len(source_index)),
    }


def verify_unified_store_views(
    *,
    store_root: str | Path = DEFAULT_RESEARCH_STORE_ROOT,
    legacy_root: str | Path = DEFAULT_LEGACY_SEQUENCE_PACK_ROOT,
    policy_path: str | Path = DEFAULT_RETENTION_POLICY,
    max_checks: int = 128,
) -> dict[str, Any]:
    store_root_path = _workspace_path(store_root)
    index_path = store_root_path / "research_store_index.json"
    index = _read_json(index_path) if index_path.exists() else {}
    graph = build_active_view_dependency_graph(store_root=store_root_path, policy_path=policy_path)
    discovered = discover_research_store_views(store_root_path)
    active_ids = [str(item) for item in list(graph.get("active_view_ids", []) or [])]
    indexed_views = dict(index.get("views", {}) or {})
    blockers: list[str] = []
    validations: dict[str, Any] = {}

    if int(index.get("schema_version", 0) or 0) != 2:
        blockers.append("research_store_index_schema_must_be_v2")
    if list(index.get("active_view_ids", []) or []) != active_ids:
        blockers.append("research_store_index_active_view_ids_mismatch")
    if set(indexed_views) != set(active_ids):
        blockers.append("research_store_index_view_set_mismatch")
    for view_id in active_ids:
        view_path = discovered.get(view_id)
        if view_path is None:
            blockers.append(f"missing_active_view:{view_id}")
            continue
        item = indexed_views.get(view_id)
        if not isinstance(item, Mapping):
            blockers.append(f"missing_v2_index_entry:{view_id}")
            continue
        indexed_path = Path(str(item.get("path", "") or ""))
        if not indexed_path.is_absolute():
            indexed_path = store_root_path / indexed_path
        if indexed_path.resolve() != view_path.resolve():
            blockers.append(f"indexed_view_path_mismatch:{view_id}")
            continue
        expected_hash = str(item.get("sha256", "") or "")
        actual_hash = _sha256_file(view_path)
        if not expected_hash or expected_hash != actual_hash:
            blockers.append(f"indexed_view_hash_mismatch:{view_id}")
        validation = validate_sequence_pack(view_path)
        validations[view_id] = validation
        if validation.get("status") != "ok":
            blockers.extend(f"{view_id}:{item}" for item in list(validation.get("blockers", []) or []))
    blockers.extend(f"missing_runtime_path:{item['view_id']}:{item['path']}" for item in graph["missing_runtime_paths"])
    blockers.extend(f"missing_policy_view:{view_id}" for view_id in graph["missing_views"])

    result: dict[str, Any] = {
        "schema_version": 2,
        "status": "blocked" if blockers else "ok",
        "generated_at": _now(),
        "index_path": _relative(index_path),
        "policy_path": graph.get("policy_path", ""),
        "active_view_ids": active_ids,
        "validated_views": {view_id: _relative(discovered[view_id]) for view_id in active_ids if view_id in discovered},
        "inactive_disk_view_ids": list(graph.get("inactive_disk_view_ids", []) or []),
        "active_component_ids": list(graph.get("active_component_ids", []) or []),
        "dependency_graph": graph.get("views", {}),
        "validations": validations,
        "blockers": blockers,
    }
    if not blockers and "seq100_path20_nextopen_ohlc_from_path60" in active_ids:
        legacy_result = _verify_all_legacy_views_with_path20_equivalence(
            store_root=store_root_path,
            legacy_root=legacy_root,
            max_checks=max_checks,
        )
        for key in ("path20_equivalence_checks", "path20_label_source", "path20_sample_count"):
            if key in legacy_result:
                result[key] = legacy_result[key]
    return result


def delete_replaced_legacy_packs(
    *,
    store_root: str | Path = DEFAULT_RESEARCH_STORE_ROOT,
    legacy_root: str | Path = DEFAULT_LEGACY_SEQUENCE_PACK_ROOT,
    confirm_delete: str = "",
) -> dict[str, Any]:
    if str(confirm_delete or "") != DELETE_CONFIRMATION:
        raise ValueError(f"delete requires --confirm-delete {DELETE_CONFIRMATION}")
    verification = verify_unified_store_views(store_root=store_root, legacy_root=legacy_root)
    if verification.get("status") != "ok":
        raise RuntimeError(f"research store verification blocked deletion: {verification.get('blockers', [])}")
    root = _workspace_path(legacy_root).resolve()
    deleted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for name in LEGACY_PACK_NAMES:
        path = (root / name).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            skipped.append({"name": name, "path": str(path), "reason": "outside_legacy_root"})
            continue
        if not path.exists():
            skipped.append({"name": name, "path": _relative(path), "reason": "missing"})
            continue
        if not path.is_dir():
            skipped.append({"name": name, "path": _relative(path), "reason": "not_directory"})
            continue
        shutil.rmtree(path)
        deleted.append({"name": name, "path": _relative(path)})
    return {
        "schema_version": 1,
        "status": "deleted",
        "generated_at": _now(),
        "verification": verification,
        "deleted_count": len(deleted),
        "deleted": deleted,
        "skipped": skipped,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and manage lightweight daily_research research_store views.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-from-legacy-packs")
    build.add_argument("--store-root", type=Path, default=DEFAULT_RESEARCH_STORE_ROOT)
    build.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_SEQUENCE_PACK_ROOT)
    build.add_argument("--store-id", default=DEFAULT_PANEL_STORE_ID)
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--json", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("--store-root", type=Path, default=DEFAULT_RESEARCH_STORE_ROOT)
    verify.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_SEQUENCE_PACK_ROOT)
    verify.add_argument("--policy-path", type=Path, default=DEFAULT_RETENTION_POLICY)
    verify.add_argument("--max-checks", type=int, default=128)
    verify.add_argument("--json", action="store_true")
    refresh = sub.add_parser("refresh-index")
    refresh.add_argument("--store-root", type=Path, default=DEFAULT_RESEARCH_STORE_ROOT)
    refresh.add_argument("--policy-path", type=Path, default=DEFAULT_RETENTION_POLICY)
    refresh.add_argument("--json", action="store_true")
    delete = sub.add_parser("delete-replaced-legacy-packs")
    delete.add_argument("--store-root", type=Path, default=DEFAULT_RESEARCH_STORE_ROOT)
    delete.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_SEQUENCE_PACK_ROOT)
    delete.add_argument("--confirm-delete", default="")
    delete.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-from-legacy-packs":
        result = build_unified_store_from_legacy_packs(
            store_root=args.store_root,
            legacy_root=args.legacy_root,
            store_id=str(args.store_id),
            overwrite=bool(args.overwrite),
        )
    elif args.command == "verify":
        result = verify_unified_store_views(
            store_root=args.store_root,
            legacy_root=args.legacy_root,
            policy_path=args.policy_path,
            max_checks=int(args.max_checks),
        )
    elif args.command == "refresh-index":
        result = build_research_store_index(
            store_root=args.store_root,
            policy_path=args.policy_path,
        )
    elif args.command == "delete-replaced-legacy-packs":
        result = delete_replaced_legacy_packs(
            store_root=args.store_root,
            legacy_root=args.legacy_root,
            confirm_delete=str(args.confirm_delete or ""),
        )
    else:
        raise ValueError(f"unsupported command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default) if bool(getattr(args, "json", False)) else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
