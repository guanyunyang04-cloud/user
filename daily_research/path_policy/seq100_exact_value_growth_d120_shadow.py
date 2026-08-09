"""Freeze the parallel D120 arm before the original 2026 cohort fills."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from daily_research.path_policy import (
    seq100_exact_value_growth_exit_challenge as challenge,
)
from daily_research.path_policy import seq100_exact_value_growth_shadow as original
from daily_research.path_policy import seq100_stock_distribution as base

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_exact_value_growth_2026_d120_shadow_v1"
SUMMARY_SCHEMA = "seq100_exact_value_growth_2026_d120_shadow_summary/1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/studies/seq100_exact_value_growth_2026_d120_shadow_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_exact_value_growth_2026_d120_shadow_v1"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    partial.unlink(missing_ok=True)
    frame.to_parquet(partial, index=False)
    os.replace(partial, path)


def _payload_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def load_study(
    path: str | Path = DEFAULT_STUDY_PATH,
) -> tuple[dict[str, Any], Path]:
    study_path = base._resolve_path(path)
    study = _read_json(study_path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("study_id_mismatch")
    contract = dict(study["epistemic_contract"])
    required = (
        "candidate_membership_rank_entry_and_sizing_are_identical_to_original_d60_arm",
        "d120_was_selected_using_2012_2025_only",
        "no_2026_fill_mark_or_return_was_observed_before_freeze",
        "original_d60_arm_is_unchanged",
        "future_records_are_append_only",
    )
    if not all(bool(contract.get(key)) for key in required):
        raise ValueError("forward_boundary_contract_invalid")
    freeze = dict(study["freeze"])
    if (
        int(freeze["planned_exit_open_days_from_signal"]) != 120
        or freeze["observation_status"] != "pending_forward_fill"
    ):
        raise ValueError("d120_freeze_contract_invalid")
    if int(study["decision_boundary"]["realized_fill_count"]) != 0:
        raise ValueError("parallel_arm_must_precede_first_fill")
    return study, study_path


def _sources(study: Mapping[str, Any]) -> dict[str, Path]:
    source = dict(study["source"])
    paths: dict[str, Path] = {}
    for key in (
        "original_shadow_study",
        "original_shadow_summary",
        "original_frozen_orders",
        "historical_exit_challenge_summary",
    ):
        path = base._resolve_path(source[key])
        if (
            not path.is_file()
            or base._sha256_file(path).lower() != str(source[f"{key}_sha256"]).lower()
        ):
            raise ValueError(f"source_invalid:{key}")
        paths[key] = path
    original.validate_summary(paths["original_shadow_summary"])
    challenge.validate_summary(paths["historical_exit_challenge_summary"])
    return paths


def initialize_shadow(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study, frozen_path = load_study(study_path)
    sources = _sources(study)
    root = base._resolve_path(output_root)
    summary_path = root / "summary.json"
    fingerprint = _payload_hash(
        {
            "study_sha256": base._sha256_file(frozen_path),
            "source_hashes": {
                key: base._sha256_file(path) for key, path in sources.items()
            },
        }
    )
    if summary_path.is_file() and not force:
        current = _read_json(summary_path)
        if current.get("experiment_fingerprint") != fingerprint:
            raise ValueError("existing_output_fingerprint_mismatch")
        if all(
            base._record_valid(record, verify_hash=True)
            for record in current.get("files", {}).values()
        ):
            return current
    orders = pd.read_parquet(sources["original_frozen_orders"])
    freeze = dict(study["freeze"])
    risk = dict(study["risk"])
    if (
        len(orders) != int(freeze["candidate_count"])
        or not bool(orders["status"].eq("pending_forward_fill").all())
        or bool(orders["entry_price_raw"].notna().any())
        or bool(orders["realized_net_return"].notna().any())
    ):
        raise ValueError("original_arm_no_longer_prefill")
    if not bool(
        orders["planned_entry_date"].eq(freeze["planned_next_open_date"]).all()
    ):
        raise ValueError("planned_entry_date_drift")
    parallel = orders.copy()
    parallel.insert(0, "shadow_arm", "parallel_d120")
    parallel["planned_exit_open_days_from_signal"] = 120
    parallel["planned_exit_rule"] = str(freeze["planned_exit"])
    parallel["source_order_sha256"] = str(
        study["source"]["original_frozen_orders_sha256"]
    )
    output_path = root / "frozen_orders.parquet"
    _write_parquet(output_path, parallel)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "frozen_pending_forward_fill",
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "study_sha256": base._sha256_file(frozen_path),
        "candidate_count": len(parallel),
        "signal_close_date": str(freeze["signal_close_date"]),
        "planned_entry_date": str(freeze["planned_next_open_date"]),
        "planned_exit_open_days_from_signal": 120,
        "target_cash_per_order_cny": float(risk["target_cash_per_order_cny"]),
        "realized_fill_count": 0,
        "realized_exit_count": 0,
        "realized_performance_available": False,
        "original_d60_arm_modified": False,
        "forward_append_only": True,
        "profit_claim_allowed": False,
        "files": {"frozen_orders": base._file_record(output_path)},
    }
    base._write_json(summary_path, summary)
    return summary


def validate_summary(path: str | Path) -> dict[str, Any]:
    summary = _read_json(base._resolve_path(path))
    checks = {
        "schema": summary.get("schema") == SUMMARY_SCHEMA,
        "study": summary.get("study_id") == STUDY_ID,
        "pending": summary.get("status") == "frozen_pending_forward_fill",
        "d120": int(summary.get("planned_exit_open_days_from_signal", -1)) == 120,
        "no_fills": int(summary.get("realized_fill_count", -1)) == 0,
        "original_unchanged": summary.get("original_d60_arm_modified") is False,
        "not_profit_claim": summary.get("profit_claim_allowed") is False,
        "files": all(
            base._record_valid(record, verify_hash=True)
            for record in summary.get("files", {}).values()
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"summary_validation_failed:{checks}")
    return {"status": "ok", "checks": checks}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = initialize_shadow(
        study_path=args.study,
        output_root=args.output_root,
        force=args.force,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
