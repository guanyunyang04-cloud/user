from __future__ import annotations

import argparse
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
    index = {
        "schema_version": 1,
        "artifact_type": "qdp_v2_research_store_index",
        "created_at": _now(),
        "store_root": str(store_root_path.resolve()),
        "panel_store": panel_store,
        "label_stores": {
            "path60_nextopen_ohlc_v1": labels_path60_ohlc,
            "path60_nextopen_ohlcva_v1": labels_path60_ohlcva,
            "path60_todayclose_ohlcva_v1": labels_todayclose,
            "path20_nextopen_ohlc_v1": labels_path20_ohlc,
        },
        "views": {name: str((store_root_path / "views" / f"{name}.json").resolve()) for name in views},
        "legacy_packs_replaced": list(LEGACY_PACK_NAMES),
    }
    index_path = _write_json(store_root_path / "research_store_index.json", index)
    index["index_path"] = str(index_path.resolve())
    _write_json(index_path, index)
    return {
        "schema_version": 1,
        "status": "built",
        "generated_at": _now(),
        "index_path": str(index_path.resolve()),
        "view_paths": index["views"],
        "legacy_packs_replaced": list(LEGACY_PACK_NAMES),
    }


def _open_memmap(meta: Mapping[str, Any], *, dtype: str) -> np.memmap:
    return np.memmap(str(meta["path"]), dtype=dtype, mode="r", shape=tuple(int(item) for item in meta["shape"]))


def verify_unified_store_views(
    *,
    store_root: str | Path = DEFAULT_RESEARCH_STORE_ROOT,
    legacy_root: str | Path = DEFAULT_LEGACY_SEQUENCE_PACK_ROOT,
    max_checks: int = 128,
) -> dict[str, Any]:
    store_root_path = _workspace_path(store_root)
    legacy_root_path = _workspace_path(legacy_root)
    index = _read_json(store_root_path / "research_store_index.json")
    view_paths = {name: Path(path) for name, path in dict(index.get("views", {}) or {}).items()}
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
        if "path20_nextopen_ohlc_v1" not in dict(index.get("label_stores", {}) or {}):
            raise ValueError("path20 source label is missing and no path20 label store is registered")
    return {
        "schema_version": 1,
        "status": "ok",
        "generated_at": _now(),
        "validated_views": {name: _relative(path) for name, path in view_paths.items()},
        "path20_equivalence_checks": int(len(positions)),
        "path20_label_source": "legacy_source" if source_label_available else "moved_label_store",
        "path20_sample_count": int(len(source_index)),
    }


def delete_replaced_legacy_packs(
    *,
    store_root: str | Path = DEFAULT_RESEARCH_STORE_ROOT,
    legacy_root: str | Path = DEFAULT_LEGACY_SEQUENCE_PACK_ROOT,
    confirm_delete: str = "",
) -> dict[str, Any]:
    if str(confirm_delete or "") != DELETE_CONFIRMATION:
        raise ValueError(f"delete requires --confirm-delete {DELETE_CONFIRMATION}")
    verification = verify_unified_store_views(store_root=store_root, legacy_root=legacy_root)
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
    verify.add_argument("--max-checks", type=int, default=128)
    verify.add_argument("--json", action="store_true")
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
            max_checks=int(args.max_checks),
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
