"""Independent validation for the continuation-value study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_daily_continuation_value as value
from daily_research.path_policy.seq100_exit_policy_audit import (
    CandidateCompleteAuditPack,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = value.DEFAULT_OUTPUT_ROOT
VALIDATION_SCHEMA = "seq100_daily_continuation_value_validation/1"


def _resolve(path: str | Path) -> Path:
    target = Path(path)
    return target if target.is_absolute() else WORKSPACE_ROOT / target


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_resolve(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: str | Path) -> dict[str, Any]:
    target = _resolve(path)
    return {
        "path": str(target.resolve()),
        "size": int(target.stat().st_size),
        "sha256": _sha256(target),
    }


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _independent_sale(
    pack: CandidateCompleteAuditPack,
    date_idx: int,
    symbol_idx: int,
    minimum_offset: int,
    maximum_offset: int,
    maximum_date_idx: int,
) -> tuple[int, float]:
    for offset in range(int(minimum_offset), int(maximum_offset) + 1):
        sale_date = int(date_idx) + offset
        if sale_date > int(maximum_date_idx) or sale_date >= len(pack.date_values):
            break
        price = float(pack.exit_close_raw[sale_date, int(symbol_idx)])
        if (
            bool(pack.exit_sellable[sale_date, int(symbol_idx)])
            and math.isfinite(price)
            and price > 0.0
        ):
            return sale_date, price
    return -1, math.nan


def _sample_outcome_reconciliation(
    outcome_manifest: Mapping[str, Any],
    pack: CandidateCompleteAuditPack,
    *,
    maximum_date_idx: int,
    sample_per_year: int = 1000,
) -> dict[str, Any]:
    rows_checked = 0
    maximum_date_difference = 0
    maximum_price_difference = 0.0
    maximum_value_difference = 0.0
    forbidden_rows = 0
    rng = np.random.default_rng(9137)
    for record in outcome_manifest["years"]:
        frame = pd.read_parquet(
            Path(str(record["path"])),
            columns=[
                "date_idx",
                "symbol_idx",
                "exit_now_date_idx",
                "continue_date_idx",
                "continuation_log_value",
                "outcome_valid",
            ],
        )
        positions = np.arange(len(frame))
        if len(positions) > sample_per_year:
            positions = np.sort(
                rng.choice(positions, size=sample_per_year, replace=False)
            )
        for row in frame.iloc[positions].itertuples(index=False):
            first_idx, first_price = _independent_sale(
                pack,
                int(row.date_idx),
                int(row.symbol_idx),
                1,
                80,
                maximum_date_idx,
            )
            second_idx, second_price = _independent_sale(
                pack,
                int(row.date_idx),
                int(row.symbol_idx),
                2,
                80,
                maximum_date_idx,
            )
            valid = first_idx >= 0 and second_idx >= 0
            if bool(row.outcome_valid) != valid:
                raise ValueError("continuation_value_sample_validity_mismatch")
            if valid:
                expected_value = math.log(second_price / first_price)
                maximum_date_difference = max(
                    maximum_date_difference,
                    abs(first_idx - int(row.exit_now_date_idx)),
                    abs(second_idx - int(row.continue_date_idx)),
                )
                maximum_price_difference = max(
                    maximum_price_difference,
                    abs(
                        first_price
                        - float(pack.exit_close_raw[first_idx, int(row.symbol_idx)])
                    ),
                    abs(
                        second_price
                        - float(pack.exit_close_raw[second_idx, int(row.symbol_idx)])
                    ),
                )
                maximum_value_difference = max(
                    maximum_value_difference,
                    abs(expected_value - float(row.continuation_log_value)),
                )
            rows_checked += 1
            forbidden_rows += int(
                int(row.exit_now_date_idx) > maximum_date_idx
                or int(row.continue_date_idx) > maximum_date_idx
            )
    return {
        "rows_checked": rows_checked,
        "maximum_exit_date_difference": int(maximum_date_difference),
        "maximum_exit_price_difference": float(maximum_price_difference),
        "maximum_continuation_value_difference": float(maximum_value_difference),
        "forbidden_sample_rows": int(forbidden_rows),
    }


def run_validation(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    output_root = _resolve(output_root)
    study_path = (
        output_root.parents[3] / "studies/seq100_daily_continuation_value_v1.json"
    )
    study = value.load_study(study_path)
    outcome_manifest_path = output_root / "outcome_manifest.json"
    prediction_manifest_path = output_root / "prediction_manifest.json"
    execution_manifest_path = output_root / "execution_manifest.json"
    outcome_manifest = _read_json(outcome_manifest_path)
    prediction_manifest = _read_json(prediction_manifest_path)
    execution_manifest = _read_json(execution_manifest_path)
    source = value._source_contract(study)
    pack = CandidateCompleteAuditPack(source["pack_manifest_path"])
    maximum_date_idx = int(
        np.searchsorted(pack.date_values, str(study["source"]["maximum_outcome_date"]))
    )
    if str(pack.date_values[maximum_date_idx]) != str(
        study["source"]["maximum_outcome_date"]
    ):
        raise ValueError("continuation_value_validation_cutoff_missing")

    outcome_rows = int(sum(int(item["rows"]) for item in outcome_manifest["years"]))
    expected_rows = int(source["panel_manifest"]["audit"]["rows"])
    if outcome_rows != expected_rows:
        raise ValueError("continuation_value_validation_outcome_count_mismatch")
    outcome_audit = _sample_outcome_reconciliation(
        outcome_manifest, pack, maximum_date_idx=maximum_date_idx
    )

    prediction_audit = {
        "rows": 0,
        "duplicate_keys": 0,
        "forbidden_rows": 0,
        "causal_update_violations": 0,
        "nonfinite_prediction_rows": 0,
    }
    prediction_paths = {
        int(item["year"]): Path(str(item["path"]))
        for item in prediction_manifest["prediction_years"]
    }
    prediction_columns = list(value.PREDICTION_COLUMNS.values())
    for path in prediction_paths.values():
        frame = pd.read_parquet(
            path,
            columns=[
                "symbol",
                "date_idx",
                "trade_date",
                "maximum_training_resolution_date_idx",
                *prediction_columns,
            ],
        )
        prediction_audit["rows"] += len(frame)
        prediction_audit["duplicate_keys"] += int(
            frame.duplicated(["symbol", "date_idx"]).sum()
        )
        prediction_audit["forbidden_rows"] += int(
            (frame["trade_date"].astype(str) > "2025-12-31").sum()
        )
        prediction_audit["causal_update_violations"] += int(
            (
                frame["maximum_training_resolution_date_idx"].astype(np.int64)
                >= frame["date_idx"].astype(np.int64)
            ).sum()
        )
        prediction_audit["nonfinite_prediction_rows"] += int(
            (~np.isfinite(frame[prediction_columns].to_numpy(np.float64)))
            .any(axis=1)
            .sum()
        )

    execution_audit = execution_manifest["audit"]
    violations = dict(execution_audit["violations"])
    if any(int(item) for item in violations.values()):
        raise ValueError(f"continuation_value_execution_violations:{violations}")

    validation = {
        "schema": VALIDATION_SCHEMA,
        "status": "completed",
        "study_id": value.STUDY_ID,
        "study": _file_record(study_path),
        "outcome_manifest": _file_record(outcome_manifest_path),
        "prediction_manifest": _file_record(prediction_manifest_path),
        "execution_manifest": _file_record(execution_manifest_path),
        "audit": {
            "outcome_rows": outcome_rows,
            "expected_outcome_rows": expected_rows,
            "outcome_sample": outcome_audit,
            "prediction": prediction_audit,
            "execution_violations": violations,
            "maximum_allowed_date_idx": maximum_date_idx,
        },
        "profit_claim_allowed": False,
        "production_policy_selected": False,
        "report_generation_performed": False,
    }
    validation_path = output_root / "validation_manifest.json"
    _write_json(validation_path, validation)
    return validation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_validation(args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
