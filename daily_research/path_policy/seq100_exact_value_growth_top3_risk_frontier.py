"""Posthoc round-number risk frontier for the literal Top3 account."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_exact_value_growth_account as account
from daily_research.path_policy import (
    seq100_exact_value_growth_top3_portfolio as top3,
)
from daily_research.path_policy import (
    seq100_margin_residual_account_feasibility as feasibility,
)
from daily_research.path_policy import seq100_stock_distribution as base
from daily_research.path_policy.seq100_exit_policy_audit import (
    CandidateCompleteAuditPack,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_exact_value_growth_top3_risk_frontier_v1"
SUMMARY_SCHEMA = "seq100_exact_value_growth_top3_risk_frontier_summary/1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/studies/seq100_exact_value_growth_top3_risk_frontier_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT / "daily_research/output/path_policy/studies/"
    "seq100_exact_value_growth_top3_risk_frontier_v1"
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
    epistemic = dict(study["epistemic_contract"])
    if not all(bool(value) for value in epistemic.values()):
        raise ValueError("epistemic_contract_mismatch")
    if int(study["source"]["forbidden_year"]) != base.FORBIDDEN_YEAR:
        raise ValueError("forbidden_year_contract_mismatch")
    policies = [(str(row["name"]), int(row["horizon"])) for row in study["policies"]]
    if policies != [("fixed_d60_max3", 60), ("fixed_d120_max3", 120)]:
        raise ValueError("policy_contract_mismatch")
    if tuple(float(value) for value in study["gross_fraction_grid"]) != (
        0.4,
        0.45,
        0.5,
        0.55,
    ):
        raise ValueError("gross_grid_contract_mismatch")
    account_contract = dict(study["account_contract"])
    if account_contract != {
        "slots": 3,
        "top_k": 3,
        "rank4_or_lower_replacement": False,
        "pyramiding": False,
        "all_other_rules": (
            "inherited unchanged from seq100_exact_value_growth_top3_portfolio_v1"
        ),
    }:
        raise ValueError("account_contract_mismatch")
    if bool(study["decision_boundary"].get("production_claim_allowed", True)):
        raise ValueError("production_claim_forbidden")
    return study, study_path


def _source_contract(
    study: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Path], dict[str, Any]]:
    source = dict(study["source"])
    sources: dict[str, Path] = {}
    for key in ("top3_study", "top3_summary"):
        path = base._resolve_path(source[key])
        expected = str(source[f"{key}_sha256"]).lower()
        if not path.is_file() or base._sha256_file(path).lower() != expected:
            raise ValueError(f"source_invalid:{key}")
        sources[key] = path
    top3.validate_summary(sources["top3_summary"])
    top3_study, loaded = top3.load_study(sources["top3_study"])
    if loaded != sources["top3_study"]:
        raise ValueError("top3_study_path_drift")
    _, account_sources, account_study = top3._source_contract(top3_study)
    if dict(study["evaluation"]["risk_account_gate"]) != dict(
        top3_study["evaluation"]["risk_account_gate"]
    ):
        raise ValueError("risk_gate_drift")
    return sources, top3_study, account_sources, account_study


def _choice(
    rows: Sequence[Mapping[str, Any]], *, near_tie_tolerance: float
) -> dict[str, Any]:
    passing = [dict(row) for row in rows if bool(row["gate_passed"])]
    if not passing:
        return {
            "stable_grid_point_found": False,
            "selected_policy": None,
            "selected_gross_fraction": None,
            "passing_grid_point_count": 0,
        }
    best_growth = max(float(row["annualized_log_growth"]) for row in passing)
    near = [
        row
        for row in passing
        if best_growth - float(row["annualized_log_growth"])
        <= float(near_tie_tolerance)
    ]
    selected = min(
        near,
        key=lambda row: (
            float(row["gross_fraction"]),
            -float(row["annualized_log_growth"]),
            str(row["policy"]),
        ),
    )
    return {
        "stable_grid_point_found": True,
        "selected_policy": str(selected["policy"]),
        "selected_gross_fraction": float(selected["gross_fraction"]),
        "selected_annualized_log_growth": float(selected["annualized_log_growth"]),
        "selected_annualized_compound_return": float(
            selected["annualized_compound_return"]
        ),
        "selected_annualized_volatility": float(selected["annualized_volatility"]),
        "selected_maximum_drawdown": float(selected["maximum_drawdown"]),
        "selected_positive_year_count": int(selected["positive_year_count"]),
        "passing_grid_point_count": len(passing),
        "near_best_grid_point_count": len(near),
    }


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study, frozen_path = load_study(study_path)
    sources, top3_study, account_sources, account_study = _source_contract(study)
    root = base._resolve_path(output_root)
    summary_path = root / "summary.json"
    fingerprint = _payload_hash(
        {
            "study_sha256": base._sha256_file(frozen_path),
            "implementation_sha256": base._sha256_file(Path(__file__)),
            "source_hashes": {
                **{key: base._sha256_file(path) for key, path in sources.items()},
                **{
                    f"account_source__{key}": base._sha256_file(path)
                    for key, path in account_sources.items()
                    if path.is_file()
                },
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

    panel = base.load_panel(
        input_manifest_path=account_sources["input_manifest"],
        label_manifest_path=account_sources["label_manifest"],
    )
    pack = CandidateCompleteAuditPack(account_sources["pack_manifest"])
    feasibility._validate_pack_alignment(panel, pack)
    source_book, _, source_selection_audit = account._account_selection(
        panel,
        pack,
        study=account_study,
        feature_path=account_sources["candidate_features"],
        selection_path=account_sources["selections"],
    )
    adjust_factor, factor_audit = feasibility._load_adjust_factor_panel(
        pack, study=account_study
    )
    boundary = dict(top3_study["common_account_boundary"])
    market = replace(
        account._market(pack, adjust_factor=adjust_factor),
        forward_days=int(boundary["extended_forward_days"]),
        execution_days=int(boundary["extended_execution_days"]),
    )
    signal_amount = feasibility._signal_amount_panel(panel, pack)
    dates = np.asarray(pack.date_values, dtype=str)
    cutoff_positions = np.flatnonzero(
        dates == str(account_study["source"]["maximum_account_mark_date"])
    )
    if len(cutoff_positions) != 1:
        raise ValueError("maximum_account_mark_date_missing")
    cutoff = int(cutoff_positions[0])
    common_last_signal = cutoff - int(boundary["extended_execution_days"])
    book, book_audit = top3._strict_top3_book(
        source_book,
        maximum_date_idx=common_last_signal,
        monthly_plan=False,
        profile="exact_value_growth_strict_top3_risk_frontier",
    )
    years = tuple(int(value) for value in top3_study["evaluation"]["years"])
    files: dict[str, dict[str, Any]] = {}
    result_rows: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    gates: dict[str, dict[str, dict[str, Any]]] = {}
    for policy_row in study["policies"]:
        name = str(policy_row["name"])
        strategy = {
            "name": name,
            "policy": f"fixed_d{int(policy_row['horizon'])}",
        }
        policy = top3._policy(strategy)
        metrics[name] = {}
        gates[name] = {}
        for gross_fraction in study["gross_fraction_grid"]:
            gross_fraction = float(gross_fraction)
            result = top3._run_account(
                market=market,
                book=book,
                policy=policy,
                slots=3,
                study=top3_study,
                signal_amount=signal_amount,
                years=years,
                cutoff=cutoff,
                target_gross_fraction=gross_fraction,
            )
            metric = top3._metric_summary(result[0], result[3], top3_study)
            gate = top3._risk_gate(metric, metric["stability"], top3_study)
            key = f"gross_{round(gross_fraction * 100):02d}"
            metrics[name][key] = metric
            gates[name][key] = gate
            result_rows.append(
                {
                    "policy": name,
                    "horizon": int(policy_row["horizon"]),
                    "gross_fraction": gross_fraction,
                    "annualized_log_growth": float(metric["annualized_log_growth"]),
                    "annualized_compound_return": float(
                        metric["annualized_compound_return_from_log_growth"]
                    ),
                    "annualized_volatility": float(
                        metric["signal_period_annualized_volatility"]
                    ),
                    "maximum_drawdown": float(metric["signal_period_maximum_drawdown"]),
                    "positive_year_count": int(
                        metric["stability"]["positive_year_count"]
                    ),
                    "post_development_positive_year_count": int(
                        metric["stability"]["post_development_positive_year_count"]
                    ),
                    "late_positive_year_count": int(
                        metric["stability"]["late_positive_year_count"]
                    ),
                    "worst_annual_return": float(
                        metric["stability"]["worst_annual_return"]
                    ),
                    "gate_passed": bool(gate["passed"]),
                    "failed_checks": ",".join(
                        check for check, passed in gate["checks"].items() if not passed
                    ),
                }
            )
            prefix = f"{name}__{key}"
            files.update(top3._write_run(root, prefix, result))

    frontier = pd.DataFrame(result_rows).sort_values(
        ["policy", "gross_fraction"], kind="stable"
    )
    frontier_path = root / "risk_frontier.parquet"
    _write_parquet(frontier_path, frontier)
    files["risk_frontier"] = base._file_record(frontier_path)
    decision = _choice(
        result_rows,
        near_tie_tolerance=float(
            study["evaluation"]["near_tie_annualized_log_growth_tolerance"]
        ),
    )
    decision.update(
        {
            "all_results_are_posthoc_adaptive": True,
            "production_claim_allowed": False,
            "forward_confirmation_required": True,
        }
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed_posthoc_top3_risk_frontier",
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "study_sha256": base._sha256_file(frozen_path),
        "implementation_sha256": base._sha256_file(Path(__file__)),
        "source_selection_audit": source_selection_audit,
        "book_audit": book_audit,
        "factor_audit": factor_audit,
        "metrics": metrics,
        "gates": gates,
        "frontier": result_rows,
        "decision": decision,
        "forbidden_2026_read_count": 0,
        "files": files,
    }
    base._write_json(summary_path, summary)
    return summary


def validate_summary(path: str | Path) -> dict[str, Any]:
    summary = _read_json(base._resolve_path(path))
    checks = {
        "schema": summary.get("schema") == SUMMARY_SCHEMA,
        "study": summary.get("study_id") == STUDY_ID,
        "completed": summary.get("status") == "completed_posthoc_top3_risk_frontier",
        "no_2026": int(summary.get("forbidden_2026_read_count", -1)) == 0,
        "not_production": summary.get("decision", {}).get("production_claim_allowed")
        is False,
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
    summary = run_study(
        study_path=args.study,
        output_root=args.output_root,
        force=args.force,
    )
    print(json.dumps(summary["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
