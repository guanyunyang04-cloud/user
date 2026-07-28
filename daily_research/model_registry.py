from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = WORKSPACE_ROOT / "daily_research/models/registry.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    registry = _read_json(path.resolve())
    if int(registry.get("schema_version", 0)) != 1:
        raise ValueError("unsupported model registry schema")
    bundles = list(registry.get("bundles", []) or [])
    keys = [(str(row.get("model_id", "")), int(row.get("year", 0))) for row in bundles]
    if not bundles or len(keys) != len(set(keys)):
        raise ValueError("model registry bundles must be non-empty and unique")
    return registry


def bundles(
    registry: dict[str, Any],
    *,
    model_id: str | None = None,
    years: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    requested_years = None if years is None else {int(year) for year in years}
    rows = []
    for raw in list(registry.get("bundles", []) or []):
        row = dict(raw)
        if model_id is not None and str(row.get("model_id")) != str(model_id):
            continue
        if requested_years is not None and int(row.get("year", 0)) not in requested_years:
            continue
        rows.append(row)
    return sorted(rows, key=lambda row: (str(row["model_id"]), int(row["year"])))


def resolve_bundle(
    model_id: str,
    year: int,
    *,
    path: Path = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    matches = bundles(load_registry(path), model_id=model_id, years=[year])
    if len(matches) != 1:
        raise KeyError(f"unknown model bundle: {model_id}/{year}")
    result = dict(matches[0])
    result["checkpoint_path"] = str((WORKSPACE_ROOT / str(result["checkpoint"])).resolve())
    result["training_summary_path"] = str(
        (WORKSPACE_ROOT / str(result["training_summary"])).resolve()
    )
    if not Path(result["checkpoint_path"]).is_file():
        raise FileNotFoundError(result["checkpoint_path"])
    if not Path(result["training_summary_path"]).is_file():
        raise FileNotFoundError(result["training_summary_path"])
    return result


def load_checkpoint(
    model_id: str,
    year: int,
    *,
    path: Path = DEFAULT_REGISTRY,
) -> Any:
    import torch

    bundle = resolve_bundle(model_id, year, path=path)
    return torch.load(
        bundle["checkpoint_path"],
        map_location="cpu",
        weights_only=False,
    )


def verify_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    registry = load_registry(path)
    errors: list[str] = []
    verified = []
    for row in bundles(registry):
        key = f"{row['model_id']}/{row['year']}"
        checkpoint = (WORKSPACE_ROOT / str(row["checkpoint"])).resolve()
        summary = (WORKSPACE_ROOT / str(row["training_summary"])).resolve()
        for artifact, label in (
            (checkpoint, "checkpoint"),
            (summary, "training_summary"),
        ):
            if not artifact.is_file():
                errors.append(f"missing:{key}:{label}:{artifact}")
                continue
            if label == "training_summary":
                try:
                    _read_json(artifact)
                except (OSError, json.JSONDecodeError) as exc:
                    errors.append(f"unreadable:{key}:{label}:{exc}")
        verified.append(key)
    return {
        "status": "ok" if not errors else "blocked",
        "registry": str(path.resolve()),
        "bundle_count": len(verified),
        "verified_bundles": verified,
        "errors": errors,
    }
