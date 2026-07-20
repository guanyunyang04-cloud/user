"""Capital-speed evaluation for complete Structured input models.

Models always own both ranking and path exit. Fixed exits are the only fallback
when a model's path exit does not improve continuous-account growth.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_finite_capital_backtest as finite
from daily_research.path_policy import seq100_structured_input_ablation as ablation
from daily_research.path_policy.seq100_exit_policy_audit import (
    CandidateCompleteAuditPack,
)


SEMANTIC_VERSION = "capital-speed-joint-model-v1"
TOP_K_VALUES = (1, 3)
SLOTS_BY_TOP_K = {1: (1, 3, 6, 12, 24, 48), 3: (3, 6, 12, 24, 48)}
FIXED_EXIT_DAYS = tuple(range(2, 61))
COST_SCENARIOS = ("base", "double_slippage")
SELECTION_YEARS = (2023, 2024, 2025)
PROGRESS_EVENT_INTERVAL = 50


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=ablation._json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_event(root: Path, payload: Mapping[str, Any]) -> None:
    event = {"timestamp": _now(), **dict(payload)}
    with (root / "monitor_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    _write_json(root / "monitor.json", event)


def _protocol_amendment(study_root: Path) -> dict[str, Any]:
    study_path = study_root.resolve() / "study.json"
    path = study_root.resolve() / "protocol_amendment_joint_model_capital_speed_v5.json"
    payload = {
        "schema_version": 1,
        "artifact_type": "seq100_structured_input_ablation_protocol_amendment",
        "amendment_id": "joint-model-capital-speed-v5",
        "study_id": ablation.STUDY_ID,
        "supersedes_for_stage_and_strategy_selection": [
            "original study contract selection_rule",
            "capital-selection-v2",
            "all-stage-capital-selection-v3",
            "profit-selection-v4",
        ],
        "authorization_timing": (
            "user-authorized after Stage 2 completed and before Stage 3 started"
        ),
        "selection_years": list(SELECTION_YEARS),
        "selection_year_role": "three_symmetric_development_folds_selected_together",
        "selection_rule": {
            "primary_objective": (
                "maximize after-cost annualized log account growth over the common "
                "2023-2025 calendar window"
            ),
            "top_k": list(TOP_K_VALUES),
            "slots_by_top_k": {
                str(key): list(value) for key, value in SLOTS_BY_TOP_K.items()
            },
            "fixed_exit_days": list(FIXED_EXIT_DAYS),
            "own_exit_policies": ["model_plan", "rolling_path"],
            "cost_scenarios": list(COST_SCENARIOS),
            "cross_model_ranking_exit_hybrid_selection_allowed": False,
            "fixed_exit_fallback_allowed": True,
            "winner_reported_globally_and_by_topk_slot": True,
        },
        "interpretation": (
            "A model supplies its own ranking and exit. If its own exit is worse "
            "than a fixed legal exit, that same model may use the fixed exit."
        ),
        "original_study_contract_sha256": str(
            ablation._read_json(study_path)["contract_sha256"]
        ),
        "protected_material_unchanged": {
            "qdp": True,
            "base_pack": True,
            "candidate_keys": True,
            "labels": True,
            "execution_material": True,
            "checkpoints": True,
            "predictions": True,
        },
    }
    if path.is_file():
        if ablation._read_json(path) != payload:
            raise ValueError("joint-model capital-speed amendment drifted")
    else:
        _write_json(path, payload)
    return {"path": str(path), "sha256": ablation._file_sha256(path)}


@dataclass(frozen=True)
class JobSpec:
    model_id: str
    top_k: int
    policy_name: str
    policy_kind: str
    fixed_day: int | None
    slots: int
    cost_scenario: str

    @property
    def job_id(self) -> str:
        return (
            f"{self.model_id}__top{self.top_k}__{self.policy_name}__"
            f"slots{self.slots:02d}__{self.cost_scenario}"
        )


def _job_specs(model_ids: Sequence[str]) -> list[JobSpec]:
    jobs: list[JobSpec] = []
    for model_id in model_ids:
        for top_k in TOP_K_VALUES:
            policies = [
                (f"fixed_d{day}", "fixed", int(day))
                for day in FIXED_EXIT_DAYS
            ]
            policies.extend(
                [
                    ("model_plan", "model_plan", None),
                    ("rolling_path", "rolling", None),
                ]
            )
            for policy_name, policy_kind, fixed_day in policies:
                for slots in SLOTS_BY_TOP_K[int(top_k)]:
                    for cost_scenario in COST_SCENARIOS:
                        jobs.append(
                            JobSpec(
                                model_id=str(model_id),
                                top_k=int(top_k),
                                policy_name=str(policy_name),
                                policy_kind=str(policy_kind),
                                fixed_day=fixed_day,
                                slots=int(slots),
                                cost_scenario=str(cost_scenario),
                            )
                        )
    return jobs


def _model_descriptors(stage: int, study_root: Path) -> dict[str, dict[str, Any]]:
    decisions = ablation._load_stage_decisions(study_root)
    decision = ablation._decision_for_stage(decisions, int(stage))
    if decision is None:
        raise ValueError(f"Stage {stage} diagnostics are incomplete")
    descriptors = [dict(decision["incumbent"]), dict(decision["challenger"])]
    output: dict[str, dict[str, Any]] = {}
    for descriptor in descriptors:
        model_id = str(dict(descriptor["variant"])["variant_id"])
        if model_id in output:
            raise ValueError(f"duplicate capital-speed model id: {model_id}")
        output[model_id] = descriptor
    return output


def _top_k_book(book: Any, *, top_k: int, profile_name: str) -> Any:
    if int(top_k) not in TOP_K_VALUES:
        raise ValueError(f"unsupported Top-K: {top_k}")
    selected = finite.ForecastBook(profile_name)
    for date_idx in book.signal_date_indices:
        day = book.days[int(date_idx)]
        positions = np.argsort(-day.score, kind="mergesort")[: int(top_k)]
        symbols = tuple(int(value) for value in day.symbol_idx[positions])
        selected.days[int(date_idx)] = finite.ForecastDay(
            symbol_idx=day.symbol_idx,
            score=day.score,
            planned_day=day.planned_day,
            top3_symbol_idx=symbols,
        )
    return selected


def _market(descriptor: Mapping[str, Any], study_path: Path) -> Any:
    view = ablation._descriptor_view_path(
        descriptor,
        year=SELECTION_YEARS[0],
        study_path=study_path,
    )
    audit_pack = CandidateCompleteAuditPack(view)
    return finite.BacktestMarket(
        date_values=np.asarray(audit_pack.date_values, dtype=object),
        symbol_values=np.asarray(audit_pack.symbol_values, dtype=object),
        entry_open_raw=audit_pack.entry_open_raw,
        exit_close_raw=audit_pack.exit_close_raw,
        exit_sellable=audit_pack.exit_sellable,
        entry_filled=audit_pack.entry_filled,
        contract=audit_pack.contract,
        terminal_recovery_fraction=float(audit_pack.terminal_recovery_fraction),
        forward_days=int(audit_pack.forward_days),
        execution_days=int(audit_pack.execution_days),
    )


def _policy(spec: JobSpec) -> Any:
    return finite.PolicySpec(
        name=spec.policy_name,
        kind=spec.policy_kind,
        fixed_day=spec.fixed_day,
    )


def _annualized_log_growth(metric: Mapping[str, Any]) -> float:
    total_return = float(metric["signal_period_total_return"])
    sessions = int(metric["signal_session_count"])
    if total_return <= -1.0 or sessions <= 0:
        return float("-inf")
    return float(math.log1p(total_return) * 252.0 / float(sessions))


def _run_job(
    spec: JobSpec,
    *,
    market: Any,
    book: Any,
    signal_dates: Sequence[int],
    guard: Any,
    contract_sha256: str,
) -> dict[str, Any]:
    metric, _equity, trades, annual = finite.simulate_portfolio(
        market=market,
        book=book,
        raw_top3_paths={},
        policy=_policy(spec),
        slots=int(spec.slots),
        cost_scenario=str(spec.cost_scenario),
        first_signal_date_idx=int(signal_dates[0]),
        last_signal_date_idx=int(signal_dates[-1]),
        starting_cash=finite.STARTING_CASH_CNY,
        memory_guard=guard,
    )
    occupied_capital_sessions = (
        float(
            np.sum(
                trades["buy_cash_cny"].to_numpy(dtype=np.float64)
                * trades["occupied_sessions"].to_numpy(dtype=np.float64)
            )
        )
        if not trades.empty
        else 0.0
    )
    net_pnl = float(trades["net_pnl_cny"].sum()) if not trades.empty else 0.0
    payload = {
        "schema_version": 1,
        "artifact_type": "seq100_structured_capital_speed_job",
        "status": "completed",
        "completed_at": _now(),
        "semantic_version": SEMANTIC_VERSION,
        "contract_sha256": str(contract_sha256),
        "job": asdict(spec),
        "job_id": spec.job_id,
        "metric": {
            **metric,
            "annualized_log_growth": _annualized_log_growth(metric),
            "worst_calendar_year_return": float(
                min(float(row["net_return"]) for row in annual)
            ),
            "net_pnl_per_deployed_capital_session": (
                float(net_pnl / occupied_capital_sessions)
                if occupied_capital_sessions > 0.0
                else None
            ),
        },
        "annual_metrics": annual,
    }
    del _equity, trades
    return payload


def _job_valid(path: Path, *, contract_sha256: str, job_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = ablation._read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        str(payload.get("status", "")) == "completed"
        and str(payload.get("contract_sha256", "")) == str(contract_sha256)
        and str(payload.get("job_id", "")) == str(job_id)
    )


def _row_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_id": str(row["model_id"]),
        "top_k": int(row["top_k"]),
        "policy_name": str(row["policy_name"]),
        "policy_kind": str(row["policy_kind"]),
        "fixed_day": (
            int(row["fixed_day"])
            if row.get("fixed_day") is not None
            and not pd.isna(row.get("fixed_day"))
            else None
        ),
        "slot_count": int(row["slot_count"]),
        "cost_scenario": str(row["cost_scenario"]),
    }


def _selection_row(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "annualized_log_growth",
        "signal_period_cagr_trading_days",
        "signal_period_total_return",
        "liquidated_ending_equity_cny",
        "signal_period_maximum_drawdown",
        "worst_calendar_year_return",
        "mean_occupied_sessions",
        "mean_signal_capital_utilization",
        "skipped_no_slot_signal_count",
        "closed_trade_count",
        "net_pnl_per_deployed_capital_session",
    )
    return {
        **_row_identity(row),
        **{
            field: (
                None if pd.isna(row[field]) else float(row[field])
            )
            for field in fields
        },
    }


def _best(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("capital-speed selection frame is empty")
    ordered = frame.sort_values(
        ["annualized_log_growth", "model_id", "top_k", "slot_count", "policy_name"],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    )
    return _selection_row(ordered.iloc[0].to_dict())


def _selection_summary(
    metrics: pd.DataFrame,
    *,
    descriptors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    base = metrics[metrics["cost_scenario"].eq("base")].copy()
    if base.empty:
        raise ValueError("capital-speed base results are missing")
    horizons = sorted(set(base["signal_session_count"].astype(int)))
    years = sorted(
        set(int(value) for value in SELECTION_YEARS)
    )
    if len(horizons) != 1:
        raise ValueError("capital-speed jobs do not share one calendar horizon")
    global_winner = _best(base)
    by_topk_slot: list[dict[str, Any]] = []
    for (top_k, slots), frame in base.groupby(["top_k", "slot_count"], sort=True):
        by_topk_slot.append(_best(frame))
    model_frontier: list[dict[str, Any]] = []
    exit_value: list[dict[str, Any]] = []
    for (model_id, top_k, slots), frame in base.groupby(
        ["model_id", "top_k", "slot_count"], sort=True
    ):
        model_frontier.append(_best(frame))
        fixed = frame[frame["policy_kind"].eq("fixed")]
        own = frame[frame["policy_kind"].isin(["model_plan", "rolling"])]
        best_fixed = _best(fixed)
        best_own = _best(own)
        exit_value.append(
            {
                "model_id": str(model_id),
                "top_k": int(top_k),
                "slot_count": int(slots),
                "best_fixed": best_fixed,
                "best_own_exit": best_own,
                "own_exit_log_growth_delta": float(
                    best_own["annualized_log_growth"]
                    - best_fixed["annualized_log_growth"]
                ),
                "selected_execution": (
                    "own_model_exit"
                    if best_own["annualized_log_growth"]
                    > best_fixed["annualized_log_growth"]
                    else "fixed_exit_fallback"
                ),
            }
        )
    winner_model = str(global_winner["model_id"])
    stress = metrics[
        metrics["cost_scenario"].eq("double_slippage")
        & metrics["model_id"].eq(winner_model)
        & metrics["top_k"].eq(int(global_winner["top_k"]))
        & metrics["slot_count"].eq(int(global_winner["slot_count"]))
        & metrics["policy_name"].eq(str(global_winner["policy_name"]))
    ]
    if len(stress) != 1:
        raise ValueError("global winner stress counterpart is missing")
    return {
        "objective": (
            "maximize base-cost annualized log account growth across the common "
            "2023-2025 calendar window"
        ),
        "selection_years": years,
        "selection_year_role": "symmetric_and_simultaneous",
        "common_signal_calendar_sessions": int(horizons[0]),
        "cross_model_hybrids_selection_allowed": False,
        "global_growth_winner": global_winner,
        "global_growth_winner_stress": _selection_row(stress.iloc[0].to_dict()),
        "winner_by_topk_slot": by_topk_slot,
        "model_policy_frontier": model_frontier,
        "own_exit_value": exit_value,
        "selected_model_descriptor": dict(descriptors[winner_model]),
    }


def _aggregate(
    *,
    stage: int,
    root: Path,
    jobs: Sequence[JobSpec],
    descriptors: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    metric_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    contract_sha256 = str(contract["contract_sha256"])
    for spec in jobs:
        path = root / "jobs" / f"{spec.job_id}.json"
        if not _job_valid(path, contract_sha256=contract_sha256, job_id=spec.job_id):
            raise ValueError(f"capital-speed job is incomplete: {spec.job_id}")
        payload = ablation._read_json(path)
        identity = {
            "stage": int(stage),
            "model_id": spec.model_id,
            "top_k": int(spec.top_k),
            "policy_name": spec.policy_name,
            "policy_kind": spec.policy_kind,
            "fixed_day": spec.fixed_day,
            "slot_count": int(spec.slots),
            "cost_scenario": spec.cost_scenario,
        }
        metric_rows.append({**identity, **dict(payload["metric"])})
        annual_rows.extend(
            {**identity, **dict(row)} for row in payload["annual_metrics"]
        )
    metrics = pd.DataFrame(metric_rows)
    annual = pd.DataFrame(annual_rows)
    metrics_path = root / "account_metrics.csv"
    annual_path = root / "account_annual_metrics.csv"
    ablation._write_csv(metrics_path, metrics)
    ablation._write_csv(annual_path, annual)
    selection = _selection_summary(metrics, descriptors=descriptors)
    summary = {
        "schema_version": 1,
        "artifact_type": "seq100_structured_joint_model_capital_speed_review",
        "status": "completed",
        "completed_at": _now(),
        "semantic_version": SEMANTIC_VERSION,
        "study_id": ablation.STUDY_ID,
        "stage": int(stage),
        "contract_sha256": contract_sha256,
        "selection": selection,
        "selected_model_descriptor": dict(selection["selected_model_descriptor"]),
        "model_descriptors": {
            key: dict(value) for key, value in descriptors.items()
        },
        "protocol_amendment": dict(amendment),
        "job_count": int(len(jobs)),
        "outputs": {
            "account_metrics_csv": str(metrics_path.resolve()),
            "account_metrics_csv_sha256": ablation._file_sha256(metrics_path),
            "account_annual_metrics_csv": str(annual_path.resolve()),
            "account_annual_metrics_csv_sha256": ablation._file_sha256(annual_path),
        },
        "stage3_started": False,
        "qdp_changed": False,
        "base_pack_changed": False,
        "checkpoint_changed": False,
        "provider_called": False,
        "live_state_changed": False,
    }
    _write_json(root / "summary.json", summary)
    decisions = ablation._load_stage_decisions(root.parent)
    stages = list(decisions.get("stages", []) or [])
    for index, raw in enumerate(stages):
        if int(raw["stage"]) != int(stage):
            continue
        row = dict(raw)
        row["decision_scope"] = "joint_model_capital_speed_primary"
        row["capital_speed_review"] = {
            "status": "completed",
            "summary_json": str((root / "summary.json").resolve()),
            "summary_sha256": ablation._file_sha256(root / "summary.json"),
            "selected_model_descriptor": dict(
                selection["selected_model_descriptor"]
            ),
            "global_growth_winner": dict(selection["global_growth_winner"]),
            "cross_model_hybrids_selection_allowed": False,
            "selection_years": list(SELECTION_YEARS),
        }
        legacy_review = dict(row.get("capital_efficiency_review", {}) or {})
        if legacy_review:
            legacy_review["selection_authority"] = False
            legacy_review["role"] = "legacy_cross_component_diagnostic"
            row["capital_efficiency_review"] = legacy_review
        row["interpretation"] = (
            "A complete model is selected by continuous-account capital speed. "
            "Its own exit is used only when it beats that model's best fixed exit."
        )
        stages[index] = row
        break
    else:
        raise ValueError(f"Stage {stage} decision disappeared during aggregation")
    decisions["stages"] = stages
    decisions["updated_at"] = _now()
    _write_json(root.parent / "stage_decisions.json", decisions)
    _append_event(
        root,
        {
            "event": "capital_speed_review_completed",
            "status": "completed",
            "stage": int(stage),
            "job_count": int(len(jobs)),
            "global_growth_winner": selection["global_growth_winner"],
        },
    )
    return summary


def run_capital_speed_review(
    *,
    stage: int,
    study_root: Path = ablation.STUDY_ROOT,
    max_jobs: int = 0,
) -> dict[str, Any]:
    if int(stage) not in {1, 2}:
        raise ValueError("capital-speed review is currently authorized for Stage 1/2")
    ablation.verify_prepared_study(study_root=study_root, deep=False)
    study_path = study_root.resolve() / "study.json"
    descriptors = _model_descriptors(int(stage), study_root.resolve())
    amendment = _protocol_amendment(study_root.resolve())
    root = study_root.resolve() / f"stage{int(stage)}_capital_speed_review"
    jobs_root = root / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)

    books: dict[str, Any] = {}
    top_books: dict[tuple[str, int], Any] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    for model_id, descriptor in descriptors.items():
        book, hashes = ablation._descriptor_forecast_book(
            descriptor,
            study_path=study_path,
            profile_name=model_id,
        )
        books[model_id] = book
        source_hashes[model_id] = hashes
        for top_k in TOP_K_VALUES:
            top_books[(model_id, int(top_k))] = _top_k_book(
                book,
                top_k=int(top_k),
                profile_name=f"{model_id}_top{int(top_k)}",
            )
    signal_dates = tuple(next(iter(books.values())).signal_date_indices)
    if not signal_dates or any(
        tuple(book.signal_date_indices) != signal_dates for book in books.values()
    ):
        raise ValueError("capital-speed model signal calendars differ")
    market = _market(next(iter(descriptors.values())), study_path)
    semantic = {
        "schema_version": 1,
        "semantic_version": SEMANTIC_VERSION,
        "study_contract_sha256": str(ablation._read_json(study_path)["contract_sha256"]),
        "stage": int(stage),
        "selection_years": list(SELECTION_YEARS),
        "model_descriptors": descriptors,
        "source_prediction_sha256": source_hashes,
        "top_k": list(TOP_K_VALUES),
        "slots_by_top_k": {
            str(key): list(value) for key, value in SLOTS_BY_TOP_K.items()
        },
        "fixed_exit_days": list(FIXED_EXIT_DAYS),
        "own_exit_policies": ["model_plan", "rolling_path"],
        "cost_scenarios": list(COST_SCENARIOS),
        "cross_model_hybrids": False,
        "execution_contract": "next_open_legal_close_no_leverage_no_pyramiding",
    }
    contract = {
        "contract": semantic,
        "contract_sha256": ablation._canonical_digest(semantic),
    }
    contract_path = root / "contract.json"
    if contract_path.is_file():
        if ablation._read_json(contract_path) != contract:
            raise ValueError("capital-speed review contract drifted")
    else:
        _write_json(contract_path, contract)

    jobs = _job_specs(tuple(descriptors))
    contract_sha256 = str(contract["contract_sha256"])
    completed = sum(
        _job_valid(
            jobs_root / f"{spec.job_id}.json",
            contract_sha256=contract_sha256,
            job_id=spec.job_id,
        )
        for spec in jobs
    )
    _append_event(
        root,
        {
            "event": "capital_speed_review_started",
            "status": "running",
            "stage": int(stage),
            "completed_jobs": int(completed),
            "total_jobs": int(len(jobs)),
        },
    )
    launched = 0
    guard = finite._MemoryGuard()
    for spec in jobs:
        path = jobs_root / f"{spec.job_id}.json"
        if _job_valid(path, contract_sha256=contract_sha256, job_id=spec.job_id):
            continue
        if int(max_jobs) > 0 and launched >= int(max_jobs):
            break
        payload = _run_job(
            spec,
            market=market,
            book=top_books[(spec.model_id, int(spec.top_k))],
            signal_dates=signal_dates,
            guard=guard,
            contract_sha256=contract_sha256,
        )
        _write_json(path, payload)
        launched += 1
        completed += 1
        if completed % PROGRESS_EVENT_INTERVAL == 0 or completed == len(jobs):
            _append_event(
                root,
                {
                    "event": "capital_speed_review_progress",
                    "status": "running",
                    "stage": int(stage),
                    "completed_jobs": int(completed),
                    "total_jobs": int(len(jobs)),
                    "last_job_id": spec.job_id,
                },
            )
            print(
                f"capital-speed stage={int(stage)} "
                f"completed={int(completed)}/{int(len(jobs))}",
                flush=True,
            )
    if completed != len(jobs):
        _append_event(
            root,
            {
                "event": "capital_speed_review_partial_yielded",
                "status": "partial",
                "stage": int(stage),
                "completed_jobs": int(completed),
                "total_jobs": int(len(jobs)),
                "watcher_remaining": False,
            },
        )
        return {
            "status": "partial",
            "stage": int(stage),
            "completed_jobs": int(completed),
            "total_jobs": int(len(jobs)),
            "monitor": str((root / "monitor.json").resolve()),
        }
    return _aggregate(
        stage=int(stage),
        root=root,
        jobs=jobs,
        descriptors=descriptors,
        contract=contract,
        amendment=amendment,
    )


def verify_capital_speed_review(
    *, stage: int, study_root: Path = ablation.STUDY_ROOT
) -> dict[str, Any]:
    root = study_root.resolve() / f"stage{int(stage)}_capital_speed_review"
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = ablation._read_json(summary_path)
    if str(summary.get("status", "")) != "completed":
        raise ValueError("capital-speed summary is incomplete")
    metrics = pd.read_csv(summary["outputs"]["account_metrics_csv"])
    annual = pd.read_csv(summary["outputs"]["account_annual_metrics_csv"])
    if len(metrics) != int(summary["job_count"]):
        raise ValueError("capital-speed metric count drifted")
    expected_annual = int(summary["job_count"]) * len(SELECTION_YEARS)
    if len(annual) != expected_annual:
        raise ValueError("capital-speed annual metric count drifted")
    if set(annual["year"].astype(int)) != set(SELECTION_YEARS):
        raise ValueError("capital-speed annual year coverage drifted")
    if bool(metrics["model_id"].astype(str).str.contains("rank.*exit", regex=True).any()):
        raise ValueError("cross-model hybrid leaked into capital-speed results")
    return {
        "status": "ok",
        "stage": int(stage),
        "job_count": int(len(metrics)),
        "annual_metric_count": int(len(annual)),
        "selected_model_descriptor": dict(summary["selected_model_descriptor"]),
        "global_growth_winner": dict(summary["selection"]["global_growth_winner"]),
        "stage3_started": False,
    }


def _annual_for_selection(
    annual: pd.DataFrame, selection: Mapping[str, Any]
) -> list[dict[str, Any]]:
    frame = annual[
        annual["model_id"].eq(str(selection["model_id"]))
        & annual["top_k"].eq(int(selection["top_k"]))
        & annual["slot_count"].eq(int(selection["slot_count"]))
        & annual["policy_name"].eq(str(selection["policy_name"]))
    ].sort_values(["cost_scenario", "year"], kind="mergesort")
    return [
        {
            "cost_scenario": str(row["cost_scenario"]),
            "year": int(row["year"]),
            "net_return": float(row["net_return"]),
            "maximum_drawdown": float(row["maximum_drawdown"]),
            "mean_capital_utilization": float(row["mean_capital_utilization"]),
        }
        for row in frame.to_dict(orient="records")
    ]


def _cohort_model_metrics(
    descriptor: Mapping[str, Any], *, study_path: Path
) -> list[dict[str, Any]]:
    fields = (
        "candidate_count",
        "candidate_score_coverage",
        "rank_ic",
        "path_mae",
        "top1_base_alpha",
        "top3_base_alpha",
        "top3_selected_base",
        "exit_regret",
        "top3_average_exit_day",
    )
    rows: list[dict[str, Any]] = []
    for year in SELECTION_YEARS:
        run_dir = ablation._descriptor_run_dir(
            descriptor, year=int(year), study_path=study_path
        )
        evaluation = ablation._read_json(run_dir / "evaluation.json")
        metrics = dict(evaluation["metrics"])
        rows.append(
            {
                "year": int(year),
                **{field: metrics[field] for field in fields},
            }
        )
    return rows


def _neighboring_fixed_rows(
    metrics: pd.DataFrame,
    selection: Mapping[str, Any],
    *,
    radius: int = 3,
) -> list[dict[str, Any]]:
    if str(selection["policy_kind"]) != "fixed":
        return []
    center = int(selection["fixed_day"])
    frame = metrics[
        metrics["model_id"].eq(str(selection["model_id"]))
        & metrics["top_k"].eq(int(selection["top_k"]))
        & metrics["slot_count"].eq(int(selection["slot_count"]))
        & metrics["cost_scenario"].eq("base")
        & metrics["policy_kind"].eq("fixed")
        & metrics["fixed_day"].between(center - int(radius), center + int(radius))
    ].sort_values("fixed_day", kind="mergesort")
    return [
        {
            "fixed_day": int(row["fixed_day"]),
            "cagr": float(row["signal_period_cagr_trading_days"]),
            "maximum_drawdown": float(row["signal_period_maximum_drawdown"]),
            "closed_trade_count": int(row["closed_trade_count"]),
            "net_pnl_per_deployed_capital_session": float(
                row["net_pnl_per_deployed_capital_session"]
            ),
        }
        for row in frame.to_dict(orient="records")
    ]


def _markdown_report(payload: Mapping[str, Any]) -> str:
    stage1 = dict(payload["stage1"])
    stage2 = dict(payload["stage2"])
    top3 = dict(stage2["top3_slots3_winner"])
    global_winner = dict(stage2["global_growth_winner"])
    lines = [
        "# Structured Input Capital Speed: Pre-Stage-3",
        "",
        "2023, 2024, and 2025 are symmetric rolling development folds and are selected together. Cross-model ranking/exit hybrids are excluded.",
        "",
        "## Main result",
        "",
        f"- Selected complete model: `{dict(payload['selected_complete_model'])['variant']['variant_id']}`.",
        f"- Maximum growth: Top{global_winner['top_k']}, {global_winner['slot_count']} slot, `{global_winner['policy_name']}`, CAGR {global_winner['signal_period_cagr_trading_days']:.4%}, drawdown {global_winner['signal_period_maximum_drawdown']:.4%}, {int(global_winner['closed_trade_count'])} trades.",
        f"- Practical Top3/3 result: `{top3['model_id']}` with `{top3['policy_name']}`, CAGR {top3['signal_period_cagr_trading_days']:.4%}, drawdown {top3['signal_period_maximum_drawdown']:.4%}, {int(top3['closed_trade_count'])} trades.",
        f"- The selected model's own exit wins {stage2['own_exit_win_count']} of {stage2['own_exit_comparison_count']} model/Top-K/slot comparisons; fixed exit wins the rest.",
        "",
        "## Stage comparison",
        "",
        "| Stage | Models | Selected model | Global policy | CAGR | Trades |",
        "|---:|---|---|---|---:|---:|",
    ]
    for stage in (stage1, stage2):
        winner = dict(stage["global_growth_winner"])
        lines.append(
            f"| {stage['stage']} | {' vs '.join(stage['model_ids'])} | {stage['selected_model_id']} | Top{winner['top_k']} / {winner['slot_count']} slot / {winner['policy_name']} | {winner['signal_period_cagr_trading_days']:.4%} | {int(winner['closed_trade_count'])} |"
        )
    lines.extend(
        [
            "",
            "## Stage 2 capacity frontier",
            "",
            "| Top-K | Slots | Model | Policy | CAGR | Drawdown | Trades |",
            "|---:|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in stage2["winner_by_topk_slot"]:
        lines.append(
            f"| {int(row['top_k'])} | {int(row['slot_count'])} | {row['model_id']} | {row['policy_name']} | {float(row['signal_period_cagr_trading_days']):.4%} | {float(row['signal_period_maximum_drawdown']):.4%} | {int(row['closed_trade_count'])} |"
        )
    lines.extend(
        [
            "",
            "## Annual returns",
            "",
            "| Strategy | Cost | 2023 | 2024 | 2025 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for name, rows in (
        ("Maximum growth", stage2["global_annual"]),
        ("Top3/3", stage2["top3_slots3_annual"]),
    ):
        for cost in COST_SCENARIOS:
            yearly = {
                int(row["year"]): float(row["net_return"])
                for row in rows
                if str(row["cost_scenario"]) == cost
            }
            lines.append(
                f"| {name} | {cost} | {yearly[2023]:.4%} | {yearly[2024]:.4%} | {yearly[2025]:.4%} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The turnover-enhanced 180-day model improves the deployable ranking baseline, but its learned exit does not beat its own best fixed exit at any tested Top-K/slot capacity. Stage 3 has not been started.",
            "",
        ]
    )
    return "\n".join(lines)


def build_pre_stage3_summary(
    *, study_root: Path = ablation.STUDY_ROOT
) -> dict[str, Any]:
    root = study_root.resolve()
    study_path = root / "study.json"
    stage_payloads: dict[int, dict[str, Any]] = {}
    for stage in (1, 2):
        verify_capital_speed_review(stage=stage, study_root=root)
        summary = ablation._read_json(
            root / f"stage{stage}_capital_speed_review" / "summary.json"
        )
        metrics = pd.read_csv(summary["outputs"]["account_metrics_csv"])
        annual = pd.read_csv(summary["outputs"]["account_annual_metrics_csv"])
        selection = dict(summary["selection"])
        global_winner = dict(selection["global_growth_winner"])
        top3_slots3 = next(
            dict(row)
            for row in selection["winner_by_topk_slot"]
            if int(row["top_k"]) == 3 and int(row["slot_count"]) == 3
        )
        own_rows = list(selection["own_exit_value"])
        stage_payloads[stage] = {
            "stage": int(stage),
            "model_ids": list(summary["model_descriptors"]),
            "selected_model_id": str(
                dict(summary["selected_model_descriptor"])["variant"]["variant_id"]
            ),
            "global_growth_winner": global_winner,
            "global_growth_winner_stress": dict(
                selection["global_growth_winner_stress"]
            ),
            "global_annual": _annual_for_selection(annual, global_winner),
            "global_neighboring_fixed_days": _neighboring_fixed_rows(
                metrics, global_winner
            ),
            "top3_slots3_winner": top3_slots3,
            "top3_slots3_annual": _annual_for_selection(annual, top3_slots3),
            "top3_slots3_neighboring_fixed_days": _neighboring_fixed_rows(
                metrics, top3_slots3
            ),
            "winner_by_topk_slot": list(selection["winner_by_topk_slot"]),
            "own_exit_win_count": int(
                sum(row["selected_execution"] == "own_model_exit" for row in own_rows)
            ),
            "fixed_exit_win_count": int(
                sum(row["selected_execution"] == "fixed_exit_fallback" for row in own_rows)
            ),
            "own_exit_comparison_count": int(len(own_rows)),
        }
    selected_descriptor = dict(
        ablation._read_json(
            root / "stage2_capital_speed_review" / "summary.json"
        )["selected_model_descriptor"]
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "seq100_structured_capital_speed_pre_stage3_summary",
        "status": "completed",
        "created_at": _now(),
        "study_id": ablation.STUDY_ID,
        "selection_years": list(SELECTION_YEARS),
        "selection_year_role": "symmetric_and_simultaneous",
        "selected_complete_model": selected_descriptor,
        "stage1": stage_payloads[1],
        "stage2": stage_payloads[2],
        "stage2_model_cohort_metrics": {
            model_id: _cohort_model_metrics(descriptor, study_path=study_path)
            for model_id, descriptor in ablation._read_json(
                root / "stage2_capital_speed_review" / "summary.json"
            )["model_descriptors"].items()
        },
        "cross_model_hybrids_selected": False,
        "stage3_started": False,
        "qdp_changed": False,
        "provider_called": False,
        "live_state_changed": False,
    }
    json_path = root / "pre_stage3_capital_speed_summary.json"
    markdown_path = root / "pre_stage3_capital_speed_summary.md"
    _write_json(json_path, payload)
    markdown_path.write_text(_markdown_report(payload), encoding="utf-8")
    ablation._update_study_runtime(
        study_path,
        status="pre_stage3_complete",
        current_stage=3,
        current_task=None,
        selected_complete_model=selected_descriptor,
        pre_stage3_capital_speed_summary=str(json_path.resolve()),
        stage3_started=False,
        error=None,
    )
    ablation._append_monitor_event(
        root,
        {
            "event": "pre_stage3_capital_speed_completed",
            "status": "paused_before_stage3",
            "selected_complete_model": selected_descriptor,
            "summary": str(json_path.resolve()),
            "stage3_started": False,
            "watcher_remaining": False,
            "message": "Capital-speed review completed; Stage 3 not started",
        },
    )
    return payload
