"""Independent structural validation for the observable oracle audit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_dynamic_oracle_observable_audit as audit
from daily_research.path_policy import seq100_market_replay as replay

VALIDATION_SCHEMA = "seq100_dynamic_oracle_observable_validation/1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _check_frame(
    *,
    root: Path,
    record: Mapping[str, Any],
    required_columns: Sequence[str],
    blocking: defaultdict[str, int],
    name: str,
) -> pd.DataFrame:
    path = Path(str(record["path"]))
    if not path.is_file() or replay.sha256(path) != str(record["sha256"]):
        blocking[f"{name}_hash"] += 1
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if len(frame) != int(record["rows"]):
        blocking[f"{name}_rows"] += 1
    missing = set(required_columns) - set(frame.columns)
    if missing:
        blocking[f"{name}_columns"] += len(missing)
    return frame


def validate(
    *,
    study_path: str | Path = audit.DEFAULT_STUDY_PATH,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    study = audit.load_study(study_path)
    root = audit._resolve(output_root or study["output_root"])
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = audit._read_json(manifest_path)
    blocking: defaultdict[str, int] = defaultdict(int)
    if manifest.get("status") != "completed":
        blocking["manifest_status"] += 1
    if manifest.get("study_id") != audit.STUDY_ID:
        blocking["study_id"] += 1
    audit_payload = dict(manifest.get("audit", {}) or {})
    expected_feature_count = int(dict(study["source"])["expected_feature_count"])
    if int(audit_payload.get("feature_count", -1)) != expected_feature_count:
        blocking["feature_count"] += 1
    if int(audit_payload.get("forbidden_2026_rows", -1)) != 0:
        blocking["forbidden_2026_rows"] += 1
    if int(audit_payload.get("duplicate_input_keys", -1)) != 0:
        blocking["duplicate_input_keys"] += 1
    if int(audit_payload.get("aligned_rows", -1)) != int(
        audit_payload.get("model_input_rows", -2)
    ):
        blocking["alignment_rows"] += 1

    output_records = dict(manifest.get("outputs", {}) or {})
    frames: dict[str, pd.DataFrame] = {}
    required = {
        "feature_metrics": [
            "feature_name",
            "column_index",
            "winner_percentile_effect",
            "mean_daily_linear_ic",
        ],
        "feature_annual_metrics": [
            "year",
            "feature_name",
            "winner_percentile_effect",
            "mean_daily_linear_ic",
        ],
        "family_metrics": ["analytic_family", "feature_count"],
        "matched_neighbors": [
            "date_idx",
            "selected_symbol_idx",
            "selected_advantage",
            "nearest_distance",
            "nearest_failure_distance",
        ],
        "cost_overlap": ["metric", "value"],
        "perturbation_stability": [
            "cost_scenario",
            "perturbation_bps_per_action_value",
            "argmax_guaranteed_stable_fraction",
        ],
    }
    for name, columns in required.items():
        record = output_records.get(name)
        if not isinstance(record, Mapping):
            blocking[f"missing_{name}"] += 1
            continue
        frames[name] = _check_frame(
            root=root,
            record=record,
            required_columns=columns,
            blocking=blocking,
            name=name,
        )

    metrics = frames.get("feature_metrics", pd.DataFrame())
    if len(metrics) != expected_feature_count:
        blocking["feature_metrics_row_count"] += 1
    elif metrics["feature_name"].duplicated().any():
        blocking["feature_metrics_duplicate"] += 1
    if not metrics.empty:
        numeric = metrics.select_dtypes(include=["number"])
        if np.isinf(numeric.to_numpy(np.float64)).any():
            blocking["feature_metrics_infinity"] += 1

    annual = frames.get("feature_annual_metrics", pd.DataFrame())
    formal_years = tuple(int(value) for value in dict(study["period"])["formal_years"])
    if not annual.empty:
        if not set(annual["year"].astype(int)).issubset(set(formal_years)):
            blocking["annual_forbidden_year"] += 1
        if annual.duplicated(["year", "feature_name"]).any():
            blocking["annual_duplicate"] += 1
        if set(annual["feature_name"].astype(str)) != set(
            metrics["feature_name"].astype(str)
        ):
            blocking["annual_feature_alignment"] += 1

    neighbors = frames.get("matched_neighbors", pd.DataFrame())
    if not neighbors.empty:
        for column in (
            "nearest_distance",
            "nearest_failure_distance",
            "nearest_coverage",
            "nearest_failure_coverage",
        ):
            if column in neighbors:
                values = neighbors[column].to_numpy(np.float64)
                if np.isinf(values).any() or (values < 0.0).any():
                    blocking[f"neighbor_{column}"] += 1
        if neighbors["date_idx"].duplicated().any():
            blocking["neighbor_duplicate_date"] += 1
        if not (
            neighbors["selected_advantage"].astype(float)
            >= neighbors["nearest_failure_advantage"].astype(float)
        ).all():
            blocking["neighbor_failure_order"] += 1

    perturbations = frames.get("perturbation_stability", pd.DataFrame())
    if not perturbations.empty:
        for column in (
            "argmax_guaranteed_stable_fraction",
            "buy_vs_cash_guaranteed_stable_fraction",
        ):
            values = perturbations[column].to_numpy(np.float64)
            if np.isinf(values).any() or (values < -1.0e-12).any() or (
                values > 1.0 + 1.0e-12
            ).any():
                blocking[f"perturbation_{column}"] += 1

    result = {
        "schema": VALIDATION_SCHEMA,
        "status": "passed" if not blocking else "failed",
        "study_id": audit.STUDY_ID,
        "blocking": dict(blocking),
        "checked_outputs": sorted(frames),
        "audit_summary": audit_payload,
    }
    _write_json(root / "validation.json", result)
    if blocking:
        raise ValueError(f"dynamic_oracle_observable_validation_failed:{dict(blocking)}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate observable oracle audit outputs.")
    parser.add_argument("--study", default=str(audit.DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    result = validate(study_path=args.study, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
