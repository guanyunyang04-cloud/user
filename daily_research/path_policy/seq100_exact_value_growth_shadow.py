"""Initialize and validate the isolated 2026 forward shadow ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from daily_research.path_policy import seq100_exact_value_growth_account as account
from daily_research.path_policy import seq100_stock_distribution as base

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_exact_value_growth_2026_shadow_v1"
SUMMARY_SCHEMA = "seq100_exact_value_growth_2026_shadow_summary/1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/studies/seq100_exact_value_growth_2026_shadow_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_exact_value_growth_2026_shadow_v1"
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
    if int(study["freeze"].get("candidate_count", -1)) != 10:
        raise ValueError("candidate_count_contract_mismatch")
    candidates = list(study["candidates"])
    if [int(row["rank"]) for row in candidates] != list(range(1, 11)):
        raise ValueError("candidate_rank_contract_mismatch")
    if len({str(row["symbol"]) for row in candidates}) != 10:
        raise ValueError("candidate_symbol_duplicate")
    if study["freeze"].get("observation_status") != "pending_forward_fill":
        raise ValueError("initial_shadow_status_mismatch")
    if int(study["decision_boundary"].get("realized_fill_count", -1)) != 0:
        raise ValueError("initial_shadow_must_have_no_fills")
    if int(study["decision_boundary"].get("realized_exit_count", -1)) != 0:
        raise ValueError("initial_shadow_must_have_no_exits")
    return study, study_path


def _source_contract(study: Mapping[str, Any]) -> dict[str, Path]:
    source = dict(study["source"])
    paths: dict[str, Path] = {}
    for key in (
        "candidate_attachment",
        "method_attachment",
        "historical_account_summary",
    ):
        path = base._resolve_path(source[key])
        expected = str(source[f"{key}_sha256"]).lower()
        if not path.is_file() or base._sha256_file(path).lower() != expected:
            raise ValueError(f"shadow_source_invalid:{key}")
        paths[key] = path
    account.validate_summary(paths["historical_account_summary"])
    return paths


def initialize_shadow(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study, frozen_path = load_study(study_path)
    sources = _source_contract(study)
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
            raise ValueError("existing_shadow_fingerprint_mismatch")
        if all(
            base._record_valid(record, verify_hash=True)
            for record in current.get("files", {}).values()
        ):
            return current
    orders = pd.DataFrame(study["candidates"])
    orders["signal_close_date"] = str(study["freeze"]["signal_close_date"])
    orders["planned_entry_date"] = str(study["freeze"]["planned_next_open_date"])
    orders["target_cash_cny"] = float(study["risk"]["target_cash_per_order_cny"])
    orders["status"] = "pending_forward_fill"
    orders["entry_date"] = pd.NA
    orders["entry_price_raw"] = pd.NA
    orders["shares"] = pd.NA
    orders["exit_date"] = pd.NA
    orders["exit_price_raw"] = pd.NA
    orders["realized_net_return"] = pd.NA
    orders_path = root / "frozen_orders.parquet"
    _write_parquet(orders_path, orders)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "frozen_pending_forward_fill",
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "study_sha256": base._sha256_file(frozen_path),
        "candidate_count": len(orders),
        "signal_close_date": str(study["freeze"]["signal_close_date"]),
        "planned_entry_date": str(study["freeze"]["planned_next_open_date"]),
        "target_cash_per_order_cny": float(study["risk"]["target_cash_per_order_cny"]),
        "first_cohort_target_gross_fraction": float(
            study["risk"]["first_cohort_target_gross_fraction"]
        ),
        "realized_fill_count": 0,
        "realized_exit_count": 0,
        "realized_performance_available": False,
        "historical_inputs_modified": False,
        "forward_append_only": True,
        "profit_claim_allowed": False,
        "files": {"frozen_orders": base._file_record(orders_path)},
    }
    base._write_json(summary_path, summary)
    return summary


def validate_summary(path: str | Path) -> dict[str, Any]:
    summary = _read_json(base._resolve_path(path))
    checks = {
        "schema": summary.get("schema") == SUMMARY_SCHEMA,
        "study": summary.get("study_id") == STUDY_ID,
        "pending": summary.get("status") == "frozen_pending_forward_fill",
        "ten_candidates": int(summary.get("candidate_count", -1)) == 10,
        "no_fills": int(summary.get("realized_fill_count", -1)) == 0,
        "no_exits": int(summary.get("realized_exit_count", -1)) == 0,
        "no_performance": summary.get("realized_performance_available") is False,
        "history_unchanged": summary.get("historical_inputs_modified") is False,
        "append_only": summary.get("forward_append_only") is True,
        "files": all(
            base._record_valid(record, verify_hash=True)
            for record in summary.get("files", {}).values()
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"shadow_summary_validation_failed:{checks}")
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
