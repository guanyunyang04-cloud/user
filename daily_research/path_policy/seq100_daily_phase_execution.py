"""Causal daily phase beliefs connected to a legal finite-capital account.

This study is deliberately a policy audit, not a new predictor.  It consumes
the frozen daily path-neighbor probabilities, turns the scale-16 phase into a
long-side posterior, and compares a posterior-boundary rolling exit with fixed
holding controls under the authoritative corrected execution pack.
"""

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

from daily_research.path_policy import seq100_daily_path_neighbors as neighbors
from daily_research.path_policy import seq100_finite_capital_backtest as backtest
from daily_research.path_policy.seq100_exit_policy_audit import (
    CandidateCompleteAuditPack,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_daily_phase_execution_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_daily_phase_execution_v1"
)
STUDY_ID = "seq100_daily_phase_execution_v1"
SCHEMA_VERSION = 1
BUILDER_VERSION = 1


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else WORKSPACE_ROOT / value


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_resolve(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: str | Path, *, include_hash: bool = True) -> dict[str, Any]:
    target = _resolve(path)
    record: dict[str, Any] = {
        "path": str(target.resolve()),
        "size": int(target.stat().st_size),
    }
    if include_hash:
        record["sha256"] = _sha256(target)
    return record


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _write_frame(
    path: str | Path, frame: pd.DataFrame, *, parquet: bool = True
) -> None:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    if parquet:
        frame.to_parquet(temporary, index=False, compression="zstd")
    else:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, target)


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study = _read_json(path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("daily_phase_execution_study_id_mismatch")
    source = dict(study["source"])
    if source.get("quality_pool_name") != "quality_liquidity_pit":
        raise ValueError("daily_phase_execution_quality_pool_mismatch")
    if int(dict(study["belief"])["turning_scale"]) != 16:
        raise ValueError("daily_phase_execution_scale_contract_mismatch")
    if float(dict(study["belief"])["phase_boundary_probability"]) != 0.5:
        raise ValueError("daily_phase_execution_boundary_mismatch")
    period = dict(study["period"])
    if str(period["last_account_mark_date"]) != str(source["maximum_outcome_date"]):
        raise ValueError("daily_phase_execution_mark_cutoff_mismatch")
    execution = dict(study["execution"])
    if bool(dict(study["belief"]).get("fixed_holding_target_used", True)):
        raise ValueError("daily_phase_execution_fixed_target_forbidden")
    if int(execution["safety_hard_cap_sessions"]) != 60:
        raise ValueError("daily_phase_execution_hard_cap_mismatch")
    if int(execution["daily_top_k"]) <= 0 or int(execution["position_slots"]) < int(
        execution["daily_top_k"]
    ):
        raise ValueError("daily_phase_execution_slot_contract_invalid")
    return study


def _source_contract(study: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(study["source"])
    neighbor_path = _resolve(str(source["path_neighbor_manifest"]))
    neighbor = _read_json(neighbor_path)
    if neighbor.get("status") != "completed":
        raise ValueError("path_neighbor_source_not_completed")
    neighbor_fingerprint = str(neighbor.get("experiment_fingerprint", ""))
    if neighbor_fingerprint != str(source["expected_path_neighbor_fingerprint"]):
        raise ValueError("path_neighbor_source_fingerprint_mismatch")
    neighbor_root = neighbor_path.parent
    panel_manifest = _read_json(neighbor_root / "panel_manifest.json")
    panels = {
        int(item["year"]): _resolve(str(item["path"]))
        for item in panel_manifest["panels"]
    }
    prediction_paths = {
        int(item["year"]): _resolve(str(item["path"]))
        for item in neighbor["outputs"]["predictions"]
    }
    years = tuple(int(value) for value in dict(study["period"])["belief_years"])
    if set(years) != set(prediction_paths) or not set(years).issubset(panels):
        raise ValueError("phase_execution_source_year_mismatch")
    pack_path = _resolve(str(source["corrected_pack_manifest"]))
    pack = _read_json(pack_path)
    actual_pack_hash = _sha256(pack_path)
    if actual_pack_hash != str(source["expected_corrected_pack_sha256"]):
        raise ValueError("corrected_execution_pack_fingerprint_mismatch")
    if int(pack.get("candidate_count", -1)) != int(source["expected_candidate_count"]):
        raise ValueError("corrected_execution_pack_candidate_count_mismatch")
    if str(pack.get("end_date", "")) < str(source["maximum_outcome_date"]):
        raise ValueError("corrected_execution_pack_ends_before_outcome_cutoff")
    return {
        "neighbor_manifest_path": neighbor_path,
        "neighbor_manifest": neighbor,
        "neighbor_manifest_sha256": _sha256(neighbor_path),
        "panel_manifest_path": neighbor_root / "panel_manifest.json",
        "panel_manifest": panel_manifest,
        "prediction_paths": prediction_paths,
        "panel_paths": panels,
        "pack_manifest_path": pack_path,
        "pack_manifest": pack,
        "pack_manifest_sha256": actual_pack_hash,
    }


def _belief_query(prediction_path: Path, panel_path: Path) -> str:
    pred = neighbors.atlas._sql_quote(prediction_path)
    panel = neighbors.atlas._sql_quote(panel_path)
    return f"""
        WITH p AS (
            SELECT
                symbol,
                trade_date,
                date_idx,
                signal_year,
                causal_next_type,
                geometry_probability::DOUBLE AS geometry_probability,
                sequence_raw_probability::DOUBLE AS sequence_raw_probability,
                predicted_log_confirmation_wait_geometry::DOUBLE
                    AS predicted_log_confirmation_wait,
                predicted_log_extreme_wait_ahead_geometry::DOUBLE
                    AS predicted_log_extreme_wait_ahead,
                predicted_directional_extreme_log_return_ahead_geometry::DOUBLE
                    AS predicted_directional_extreme_log_return_ahead,
                maximum_training_resolution_date_idx
            FROM read_parquet({pred})
            WHERE turning_scale = 16
        )
        SELECT
            p.symbol,
            p.trade_date,
            p.date_idx,
            p.signal_year,
            p.causal_next_type,
            x.cumret_3::DOUBLE AS cumret_3,
            x.cumret_20::DOUBLE AS cumret_20,
            p.geometry_probability,
            p.sequence_raw_probability,
            (0.9 * p.geometry_probability
                + 0.1 * p.sequence_raw_probability)::DOUBLE
                AS blend01_probability,
            CASE WHEN p.causal_next_type = 'peak'
                THEN p.geometry_probability
                ELSE 1.0 - p.geometry_probability END::DOUBLE
                AS geometry_long_probability,
            CASE WHEN p.causal_next_type = 'peak'
                THEN (0.9 * p.geometry_probability
                    + 0.1 * p.sequence_raw_probability)
                ELSE 1.0 - (0.9 * p.geometry_probability
                    + 0.1 * p.sequence_raw_probability) END::DOUBLE
                AS blend01_long_probability,
            CASE
                WHEN p.causal_next_type = 'peak'
                     AND x.cumret_20 > 0 AND x.cumret_3 < 0
                    THEN 'uptrend_pullback'
                WHEN p.causal_next_type = 'trough'
                     AND x.cumret_20 < 0 AND x.cumret_3 > 0
                    THEN 'downtrend_rebound'
                ELSE 'other'
            END AS entry_setup,
            p.predicted_log_confirmation_wait,
            p.predicted_log_extreme_wait_ahead,
            p.predicted_directional_extreme_log_return_ahead,
            p.maximum_training_resolution_date_idx
        FROM p
        INNER JOIN read_parquet({panel}) x
            USING (symbol, date_idx)
    """


def _audit_belief_frame(
    frame: pd.DataFrame, *, year: int, expected_rows: int, cutoff: str
) -> dict[str, Any]:
    duplicate_rows = int(frame.duplicated(["symbol", "date_idx"]).sum())
    forbidden_rows = int((frame["trade_date"].astype(str) > cutoff).sum())
    bad_probability_rows = int(
        (
            ~np.isfinite(
                frame[
                    ["geometry_long_probability", "blend01_long_probability"]
                ].to_numpy()
            )
        )
        .any(axis=1)
        .sum()
    )
    bad_probability_rows += int(
        (
            (frame[["geometry_long_probability", "blend01_long_probability"]] <= 0.0)
            | (frame[["geometry_long_probability", "blend01_long_probability"]] >= 1.0)
        )
        .any(axis=1)
        .sum()
    )
    bad_setup_rows = int(
        (
            ~frame["entry_setup"].isin(
                ["other", "uptrend_pullback", "downtrend_rebound"]
            )
        ).sum()
    )
    causal_rows = int(
        (
            frame["maximum_training_resolution_date_idx"].astype(np.int64)
            >= frame["date_idx"].astype(np.int64)
        ).sum()
    )
    return {
        "year": int(year),
        "row_count": len(frame),
        "expected_row_count": int(expected_rows),
        "duplicate_rows": duplicate_rows,
        "forbidden_rows": forbidden_rows,
        "nonfinite_or_boundary_probability_rows": bad_probability_rows,
        "invalid_setup_rows": bad_setup_rows,
        "causal_training_resolution_violations": causal_rows,
        "min_date": str(frame["trade_date"].min()) if len(frame) else None,
        "max_date": str(frame["trade_date"].max()) if len(frame) else None,
    }


def build_beliefs(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study_path = _resolve(study_path)
    output_root = _resolve(output_root)
    study = load_study(study_path)
    source = _source_contract(study)
    belief_root = output_root / "daily_beliefs"
    belief_root.mkdir(parents=True, exist_ok=True)
    resource_study = dict(study)
    connection = neighbors.atlas._connect(output_root, resource_study)
    records: list[dict[str, Any]] = []
    try:
        for year in sorted(
            int(value) for value in dict(study["period"])["belief_years"]
        ):
            target = belief_root / f"year={year}" / "part-0000.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".parquet.tmp")
            temporary.unlink(missing_ok=True)
            expected = int(
                source["panel_manifest"]["panels"][year - 2012]["audit"]["row_count"]
            )
            query = _belief_query(
                source["prediction_paths"][year], source["panel_paths"][year]
            )
            connection.execute(
                f"COPY ({query}) TO {neighbors.atlas._sql_quote(temporary)} "
                f"(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE "
                f"{int(dict(study['resources'])['parquet_row_group_size'])})"
            )
            temporary.replace(target)
            frame = pd.read_parquet(target)
            audit = _audit_belief_frame(
                frame,
                year=year,
                expected_rows=expected,
                cutoff=str(dict(study["source"])["maximum_outcome_date"]),
            )
            if audit["row_count"] != audit["expected_row_count"]:
                raise ValueError(f"belief_row_count_mismatch:{audit}")
            if any(
                int(audit[key]) != 0
                for key in (
                    "duplicate_rows",
                    "forbidden_rows",
                    "nonfinite_or_boundary_probability_rows",
                    "invalid_setup_rows",
                    "causal_training_resolution_violations",
                )
            ):
                raise ValueError(f"belief_audit_failed:{audit}")
            records.append({"year": year, **_file_record(target), "audit": audit})
    finally:
        connection.close()
    aggregate = {
        "rows": int(sum(int(record["audit"]["row_count"]) for record in records)),
        "duplicate_rows": int(
            sum(int(record["audit"]["duplicate_rows"]) for record in records)
        ),
        "forbidden_rows": int(
            sum(int(record["audit"]["forbidden_rows"]) for record in records)
        ),
        "causal_training_resolution_violations": int(
            sum(
                int(record["audit"]["causal_training_resolution_violations"])
                for record in records
            )
        ),
    }
    manifest = {
        "schema": "seq100_daily_phase_beliefs/1",
        "status": "completed",
        "study_id": STUDY_ID,
        "builder_version": BUILDER_VERSION,
        "study": _file_record(study_path),
        "source": {
            "path_neighbor_manifest": _file_record(source["neighbor_manifest_path"]),
            "panel_manifest": _file_record(source["panel_manifest_path"]),
            "corrected_pack_manifest": _file_record(source["pack_manifest_path"]),
            "path_neighbor_fingerprint": source["neighbor_manifest"][
                "experiment_fingerprint"
            ],
        },
        "sequence_diagnostic_weight": float(
            dict(study["belief"])["sequence_diagnostic_weight"]
        ),
        "years": records,
        "aggregate_audit": aggregate,
        "training_performed": False,
        "fixed_holding_target_used": False,
        "execution_backtest_performed": False,
    }
    _write_json(output_root / "belief_manifest.json", manifest)
    return manifest


def _open_market(
    pack_path: Path,
) -> tuple[backtest.BacktestMarket, np.ndarray, CandidateCompleteAuditPack]:
    audit_pack = CandidateCompleteAuditPack(pack_path)
    market = backtest.BacktestMarket(
        date_values=np.asarray(audit_pack.date_values, dtype=object),
        symbol_values=np.asarray(audit_pack.symbol_values, dtype=object),
        entry_open_raw=audit_pack.entry_open_raw,
        exit_close_raw=audit_pack.exit_close_raw,
        exit_sellable=audit_pack.exit_sellable,
        entry_filled=audit_pack.entry_filled,
        costs=audit_pack.costs,
        terminal_recovery_fraction=float(audit_pack.terminal_recovery_fraction),
        forward_days=int(audit_pack.forward_days),
        execution_days=int(audit_pack.execution_days),
    )
    daily_raw = dict(_read_json(pack_path))["feature_channels"]["daily_raw"]
    shape = tuple(int(value) for value in daily_raw["shape"])
    raw_panel = np.memmap(
        _resolve(str(daily_raw["path"])), dtype="float32", mode="r", shape=shape
    )
    amount_index = list(daily_raw["columns"]).index("amount")
    return market, raw_panel[:, :, amount_index], audit_pack


def _profile_config(study: Mapping[str, Any], name: str) -> dict[str, Any]:
    for item in study["profiles"]:
        if str(item["name"]) == str(name):
            return dict(item)
    raise KeyError(name)


def _build_book(
    *,
    profile: Mapping[str, Any],
    belief_manifest: Mapping[str, Any],
    market: backtest.BacktestMarket,
    study: Mapping[str, Any],
) -> backtest.ForecastBook:
    execution = dict(study["execution"])
    top_k = int(execution["daily_top_k"])
    book = backtest.ForecastBook(
        str(profile["name"]), top_k=top_k, candidate_scan_k=top_k
    )
    symbol_to_idx = {
        str(symbol): int(index) for index, symbol in enumerate(market.symbol_values)
    }
    probability_column = str(profile["probability_column"])
    setup_scope = str(profile["setup"])
    first_date = int(
        np.searchsorted(market.date_values, str(study["period"]["first_signal_date"]))
    )
    last_signal = int(
        np.searchsorted(market.date_values, str(study["period"]["last_signal_date"]))
    )
    cap_date = int(
        np.searchsorted(
            market.date_values, str(study["period"]["last_account_mark_date"])
        )
    )
    for record in belief_manifest["years"]:
        frame = pd.read_parquet(
            _resolve(str(record["path"])),
            columns=["symbol", "date_idx", probability_column, "entry_setup"],
        )
        frame["symbol_idx"] = frame["symbol"].map(symbol_to_idx)
        if frame["symbol_idx"].isna().any():
            raise ValueError(
                f"belief_symbol_not_in_pack:{profile['name']}:{record['year']}"
            )
        frame["symbol_idx"] = frame["symbol_idx"].astype(np.int32)
        frame["score"] = frame[probability_column].astype(np.float64) - 0.5
        if setup_scope == "combined":
            eligible = frame["entry_setup"].isin(
                ["uptrend_pullback", "downtrend_rebound"]
            )
        else:
            eligible = frame["entry_setup"].eq(setup_scope)
        eligible &= frame["score"] > 0.0
        for date_idx, group in frame.groupby("date_idx", sort=True):
            date_value = int(date_idx)
            symbols = group["symbol_idx"].to_numpy(dtype=np.int32, copy=False)
            scores = group["score"].to_numpy(dtype=np.float64, copy=False)
            selection = eligible.loc[group.index].to_numpy(dtype=bool, copy=True)
            if date_value > last_signal:
                selection[:] = False
            planned = np.full(len(group), 60, dtype=np.int16)
            book.add_day(
                date_idx=date_value,
                symbol_idx=symbols,
                score=scores,
                planned_day=planned,
                selection_mask=selection,
            )
    if not book.days:
        raise ValueError(f"empty_forecast_book:{profile['name']}")
    if min(book.days) > first_date or max(book.days) > cap_date:
        raise ValueError(f"forecast_book_date_range_invalid:{profile['name']}")
    return book


def _policies(study: Mapping[str, Any]) -> tuple[backtest.PolicySpec, ...]:
    policies: list[backtest.PolicySpec] = [
        backtest.PolicySpec(name="rolling_phase_boundary", kind="rolling")
    ]
    for day in dict(study["execution"])["fixed_exit_controls"]:
        policies.append(
            backtest.PolicySpec(
                name=f"fixed_d{int(day)}", kind="fixed", fixed_day=int(day)
            )
        )
    for policy in policies:
        policy.validate()
    return tuple(policies)


def _job_id(profile: str, policy: str, cost: str) -> str:
    return f"{profile}__{policy}__{cost}"


def _trade_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "mean_net_return": 0.0,
            "median_net_return": 0.0,
            "p10_net_return": 0.0,
            "mean_duration": 0.0,
            "max_participation": 0.0,
        }
    returns = frame["net_return_on_buy_cash"].astype(float)
    participation = frame["signal_amount_participation"].astype(float)
    return {
        "trade_count": len(frame),
        "win_rate": float((returns > 0.0).mean()),
        "mean_net_return": float(returns.mean()),
        "median_net_return": float(returns.median()),
        "p10_net_return": float(returns.quantile(0.10)),
        "mean_duration": float(frame["occupied_sessions"].astype(float).mean()),
        "max_participation": float(participation.max()),
    }


def _audit_task_outputs(
    *,
    output_root: Path,
    task_records: Sequence[Mapping[str, Any]],
    market: backtest.BacktestMarket,
    audit_pack: CandidateCompleteAuditPack,
    study: Mapping[str, Any],
) -> dict[str, Any]:
    cutoff_idx = int(
        np.searchsorted(
            market.date_values, str(dict(study["period"])["last_account_mark_date"])
        )
    )
    fraction = float(dict(study["execution"])["maximum_signal_day_amount_fraction"])
    rows = 0
    max_entry_difference = 0.0
    max_participation = 0.0
    violations = {
        "forbidden_2026_rows": 0,
        "entry_t1_violations": 0,
        "exit_t1_violations": 0,
        "entry_fill_violations": 0,
        "entry_price_violations": 0,
        "sellability_violations": 0,
        "capacity_violations": 0,
        "lot_violations": 0,
        "negative_cash_rows": 0,
        "equity_terminal_mismatch": 0,
    }
    task_count = 0
    for record in task_records:
        task_count += 1
        task_dir = _resolve(str(record["path"]))
        trades_path = task_dir / "trades.parquet"
        equity_path = task_dir / "equity.parquet"
        trades = (
            pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
        )
        equity = pd.read_parquet(equity_path)
        rows += len(trades)
        if not equity.empty:
            violations["forbidden_2026_rows"] += int(
                (
                    equity["trade_date"].astype(str)
                    > str(dict(study["source"])["maximum_outcome_date"])
                ).sum()
            )
            violations["negative_cash_rows"] += int(
                (equity["cash"].astype(float) < -1e-6).sum()
            )
            metric = json.loads((task_dir / "metric.json").read_text(encoding="utf-8"))
            terminal = float(equity["equity"].iloc[-1])
            if abs(terminal - float(metric["liquidated_ending_equity_cny"])) > 1e-5:
                violations["equity_terminal_mismatch"] += 1
        for row in trades.to_dict(orient="records"):
            signal_idx = int(row["signal_date_idx"])
            entry_idx = int(row["entry_date_idx"])
            exit_idx = int(row["exit_date_idx"])
            symbol_idx = int(row["symbol_idx"])
            if signal_idx + 1 != entry_idx:
                violations["entry_t1_violations"] += 1
            if exit_idx < entry_idx + 1:
                violations["exit_t1_violations"] += 1
            if entry_idx > cutoff_idx or exit_idx > cutoff_idx:
                violations["forbidden_2026_rows"] += 1
            if not bool(audit_pack.entry_filled[signal_idx, symbol_idx]):
                violations["entry_fill_violations"] += 1
            expected_entry = float(audit_pack.entry_open_raw[entry_idx, symbol_idx])
            difference = abs(float(row["entry_price_raw"]) - expected_entry)
            max_entry_difference = max(max_entry_difference, difference)
            if difference > 1e-4:
                violations["entry_price_violations"] += 1
            if str(row["exit_reason"]) != "terminal_recovery" and not bool(
                audit_pack.exit_sellable[exit_idx, symbol_idx]
            ):
                violations["sellability_violations"] += 1
            if int(row["shares"]) % int(audit_pack.costs.lot_size) != 0:
                violations["lot_violations"] += 1
            participation = float(row["signal_amount_participation"])
            max_participation = max(max_participation, participation)
            if participation > fraction + 1e-8:
                violations["capacity_violations"] += 1
    blocking = {key: value for key, value in violations.items() if value}
    if blocking:
        raise ValueError(f"phase_execution_task_audit_failed:{blocking}")
    return {
        "task_count": int(task_count),
        "trade_rows": int(rows),
        "max_entry_price_difference": float(max_entry_difference),
        "maximum_signal_amount_participation": float(max_participation),
        "violations": violations,
    }


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study_path = _resolve(study_path)
    output_root = _resolve(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    study = load_study(study_path)
    source = _source_contract(study)
    belief_manifest = build_beliefs(study_path=study_path, output_root=output_root)
    pack_path = source["pack_manifest_path"]
    market, signal_amount_panel, audit_pack = _open_market(pack_path)
    period = dict(study["period"])
    first_signal_idx = int(
        np.searchsorted(market.date_values, str(period["first_signal_date"]))
    )
    last_signal_idx = int(
        np.searchsorted(market.date_values, str(period["last_signal_date"]))
    )
    cutoff_idx = int(
        np.searchsorted(market.date_values, str(period["last_account_mark_date"]))
    )
    if last_signal_idx + int(market.execution_days) != cutoff_idx:
        raise ValueError(
            "phase_execution_signal_and_account_cutoff_are_not_80_sessions_apart"
        )
    policies = _policies(study)
    costs = tuple(str(value) for value in dict(study["execution"])["cost_scenarios"])
    profile_configs = [dict(item) for item in study["profiles"]]
    task_root = output_root / "tasks"
    task_root.mkdir(parents=True, exist_ok=True)
    task_records: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    trade_summary_rows: list[dict[str, Any]] = []
    signal_dates = tuple(range(first_signal_idx, last_signal_idx + 1))
    for profile in profile_configs:
        book = _build_book(
            profile=profile,
            belief_manifest=belief_manifest,
            market=market,
            study=study,
        )
        for policy in policies:
            for cost in costs:
                job_id = _job_id(str(profile["name"]), policy.name, cost)
                job_dir = task_root / job_id
                job_dir.mkdir(parents=True, exist_ok=True)
                metric, equity, trades, annual = backtest.simulate_portfolio(
                    market=market,
                    book=book,
                    raw_top3_paths={},
                    policy=policy,
                    slots=int(dict(study["execution"])["position_slots"]),
                    cost_scenario=cost,
                    first_signal_date_idx=first_signal_idx,
                    last_signal_date_idx=last_signal_idx,
                    starting_cash=float(dict(study["execution"])["starting_cash_cny"]),
                    allow_pyramiding=bool(dict(study["execution"])["pyramiding"]),
                    replace_rejected_from_ranked_candidates=bool(
                        dict(study["execution"])[
                            "replacement_after_future_fill_failure"
                        ]
                    ),
                    calendar_years=tuple(
                        int(value) for value in period["belief_years"]
                    ),
                    top_k=int(dict(study["execution"])["daily_top_k"]),
                    entry_signal_date_indices=signal_dates,
                    signal_amount_panel=signal_amount_panel,
                    maximum_signal_amount_fraction=float(
                        dict(study["execution"])["maximum_signal_day_amount_fraction"]
                    ),
                    exit_on_missing_forecast=True,
                )
                metric.update(
                    {
                        "study_id": STUDY_ID,
                        "profile_role": str(profile["role"]),
                        "probability_column": str(profile["probability_column"]),
                        "entry_setup_scope": str(profile["setup"]),
                        "source_path_neighbor_fingerprint": str(
                            source["neighbor_manifest"]["experiment_fingerprint"]
                        ),
                    }
                )
                _write_json(job_dir / "metric.json", metric)
                _write_frame(job_dir / "equity.parquet", equity)
                _write_frame(job_dir / "trades.parquet", trades)
                annual_frame = pd.DataFrame(annual)
                if not annual_frame.empty:
                    annual_frame.insert(0, "study_id", STUDY_ID)
                    annual_frame.insert(1, "profile_role", str(profile["role"]))
                    annual_frame.insert(
                        2, "probability_column", str(profile["probability_column"])
                    )
                    annual_frame.insert(3, "entry_setup_scope", str(profile["setup"]))
                    annual_frame.insert(4, "profile", str(profile["name"]))
                    annual_frame.insert(5, "policy", policy.name)
                    annual_frame.insert(6, "cost_scenario", cost)
                    _write_frame(job_dir / "annual.parquet", annual_frame)
                    annual_rows.extend(annual_frame.to_dict(orient="records"))
                metric_rows.append(metric)
                trade_summary_rows.append(
                    {
                        "study_id": STUDY_ID,
                        "profile": str(profile["name"]),
                        "profile_role": str(profile["role"]),
                        "policy": policy.name,
                        "cost_scenario": cost,
                        **_trade_summary(trades),
                    }
                )
                task_records.append(
                    {
                        "job_id": job_id,
                        "profile": str(profile["name"]),
                        "policy": policy.name,
                        "cost_scenario": cost,
                        "path": str(job_dir.resolve()),
                        "metric": _file_record(job_dir / "metric.json"),
                        "equity": _file_record(job_dir / "equity.parquet"),
                        "trades": _file_record(job_dir / "trades.parquet"),
                    }
                )
        del book

    metrics = pd.DataFrame(metric_rows)
    annual = pd.DataFrame(annual_rows)
    trades_summary = pd.DataFrame(trade_summary_rows)
    comparisons: list[dict[str, Any]] = []
    for (profile, cost), group in metrics.groupby(
        ["profile", "cost_scenario"], sort=True
    ):
        dynamic = group[group["policy"] == "rolling_phase_boundary"]
        if dynamic.empty:
            continue
        dynamic_row = dynamic.iloc[0]
        for _, control in group[group["policy"].str.startswith("fixed_")].iterrows():
            comparisons.append(
                {
                    "profile": profile,
                    "cost_scenario": cost,
                    "dynamic_policy": "rolling_phase_boundary",
                    "control_policy": str(control["policy"]),
                    "delta_liquidated_log_growth": float(
                        dynamic_row["liquidated_log_growth"]
                        - control["liquidated_log_growth"]
                    ),
                    "delta_signal_period_log_growth": float(
                        math.log1p(float(dynamic_row["signal_period_total_return"]))
                        - math.log1p(float(control["signal_period_total_return"]))
                    ),
                    "dynamic_signal_period_maximum_drawdown": float(
                        dynamic_row["signal_period_maximum_drawdown"]
                    ),
                    "control_signal_period_maximum_drawdown": float(
                        control["signal_period_maximum_drawdown"]
                    ),
                }
            )
    comparison_frame = pd.DataFrame(comparisons)
    _write_frame(output_root / "configuration_metrics.parquet", metrics)
    _write_frame(output_root / "annual_metrics.parquet", annual)
    _write_frame(output_root / "trade_summary.parquet", trades_summary)
    _write_frame(output_root / "dynamic_vs_fixed.parquet", comparison_frame)
    execution_audit = _audit_task_outputs(
        output_root=output_root,
        task_records=task_records,
        market=market,
        audit_pack=audit_pack,
        study=study,
    )
    output_files = {
        "belief_manifest": output_root / "belief_manifest.json",
        "configuration_metrics": output_root / "configuration_metrics.parquet",
        "annual_metrics": output_root / "annual_metrics.parquet",
        "trade_summary": output_root / "trade_summary.parquet",
        "dynamic_vs_fixed": output_root / "dynamic_vs_fixed.parquet",
    }
    manifest = {
        "schema": "seq100_daily_phase_execution/1",
        "status": "completed",
        "study_id": STUDY_ID,
        "builder_version": BUILDER_VERSION,
        "study": _file_record(study_path),
        "source": {
            "path_neighbor_manifest": _file_record(source["neighbor_manifest_path"]),
            "path_neighbor_fingerprint": str(
                source["neighbor_manifest"]["experiment_fingerprint"]
            ),
            "corrected_pack_manifest": _file_record(source["pack_manifest_path"]),
            "corrected_pack_candidate_count": int(
                source["pack_manifest"].get("candidate_count", 0)
            ),
        },
        "period": {
            **period,
            "first_signal_idx": first_signal_idx,
            "last_signal_idx": last_signal_idx,
            "last_account_mark_idx": cutoff_idx,
        },
        "belief_manifest": _file_record(output_files["belief_manifest"]),
        "profiles": profile_configs,
        "policies": [backtest.asdict(policy) for policy in policies],
        "cost_scenarios": list(costs),
        "task_count": len(task_records),
        "tasks": task_records,
        "outputs": {key: _file_record(path) for key, path in output_files.items()},
        "audit": {
            "belief": belief_manifest["aggregate_audit"],
            "execution": execution_audit,
        },
        "training_performed": False,
        "fixed_holding_target_used": False,
        "portfolio_selection_performed": True,
        "execution_backtest_performed": True,
        "profit_claim_allowed": False,
        "production_policy_selected": False,
        "report_generation_performed": False,
    }
    _write_json(output_root / "analysis_manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = run_study(study_path=args.study, output_root=args.output)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "study_id": manifest["study_id"],
                "task_count": manifest["task_count"],
                "trade_rows": manifest["audit"]["execution"]["trade_rows"],
                "maximum_signal_amount_participation": manifest["audit"]["execution"][
                    "maximum_signal_amount_participation"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
