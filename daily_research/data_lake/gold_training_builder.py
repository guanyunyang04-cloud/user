from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from daily_research.continuous_policy.allocation_optimizer import (
    attach_unified_allocation_targets,
    build_unified_allocation_summary,
    summarize_unified_allocation_rows,
)
from daily_research.continuous_policy.allocation_teacher import (
    build_allocation_teacher_summary,
    summarize_allocation_teacher_rows,
)
from daily_research.continuous_policy.label_builder import (
    FuturePathMetrics,
    build_action_labels_for_date,
    build_teacher_global_targets,
    build_teacher_policy_frame,
    resolve_label_config,
)
from daily_research.continuous_policy.pipeline_utils import (
    _annotate_label_frame_with_execution_feedback,
    _apply_budget_objective_targets,
    compute_curve_metrics,
    estimate_trading_cost,
    signal_dates_between,
)
from daily_research.continuous_policy.portfolio_simulator import (
    DEFAULT_BUDGET_CALIBRATION,
    DEFAULT_BUDGET_SEMANTICS,
    DEFAULT_EXECUTION_SEMANTICS,
    PortfolioState,
)
from daily_research.continuous_policy.state_builder import (
    PreparedPolicyInputs,
    build_cross_section_state,
    build_daily_state_features,
)
from daily_research.data_lake.catalog import (
    ResearchDataLake,
    _available_strict_end,
    _json_safe,
    _stable_hash,
    _write_json,
)


ProgressCallback = Callable[[str, Mapping[str, Any]], None]


@dataclass(frozen=True)
class GoldBuildSpec:
    universe: str
    benchmark: str
    start_date: str
    end_date: str
    max_universe_size: int
    data_source: str
    label_preset: str
    execution_semantics: str
    budget_semantics: str
    budget_calibration: str
    budget_objective: str
    alpha_prior_source: str
    alpha_prior_score_panel: str = ""
    alpha_prior_target_weight_panel: str = ""
    transaction_cost_bps: float = 3.0
    slippage_bps: float = 7.0
    sell_tax_bps: float = 10.0
    random_seed: int = 7
    skip_multiplier: float = 2.0
    source_market_dataset_id: str = ""
    shard_frequency: str = "quarter"

    def to_spec_dict(self, *, prepared: PreparedPolicyInputs, zone: str, max_forward_horizon: int) -> dict[str, Any]:
        return {
            "dataset": "continuous_policy_training_matrices",
            "pool_name": prepared.pool_name,
            "universe": self.universe,
            "benchmark": prepared.benchmark,
            "data_source": prepared.data_source,
            "start_date": pd.Timestamp(self.start_date).strftime("%Y-%m-%d"),
            "end_date": pd.Timestamp(self.end_date).strftime("%Y-%m-%d"),
            "max_universe_size": int(self.max_universe_size),
            "label_preset": self.label_preset,
            "execution_semantics": self.execution_semantics,
            "budget_semantics": self.budget_semantics,
            "budget_calibration": self.budget_calibration,
            "budget_objective": self.budget_objective,
            "alpha_prior_source": self.alpha_prior_source,
            "alpha_prior_score_panel": self.alpha_prior_score_panel,
            "alpha_prior_target_weight_panel": self.alpha_prior_target_weight_panel,
            "transaction_cost_bps": float(self.transaction_cost_bps),
            "slippage_bps": float(self.slippage_bps),
            "sell_tax_bps": float(self.sell_tax_bps),
            "random_seed": int(self.random_seed),
            "skip_multiplier": float(self.skip_multiplier),
            "source_market_dataset_id": self.source_market_dataset_id,
            "shard_frequency": self.shard_frequency,
            "zone": zone,
            "max_forward_horizon": int(max_forward_horizon),
            "prepared_universe_size": int(len(prepared.universe)),
            "universe_digest": _stable_hash({"universe": [str(item) for item in prepared.universe]}),
            "sharded": True,
        }


def _normalize_zone(zone: str) -> str:
    text = str(zone or "strict_train").strip().lower()
    if text not in {"strict_train", "realtime_research"}:
        raise ValueError(f"Unsupported Gold training dataset zone: {zone!r}")
    return text


def _date_text(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _shard_key(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> str:
    return f"{_date_text(start_dt)}_{_date_text(end_dt)}"


def _split_dates_by_frequency(dates: Sequence[pd.Timestamp], frequency: str) -> list[list[pd.Timestamp]]:
    if not dates:
        return []
    freq = str(frequency or "quarter").strip().lower()
    if freq in {"month", "monthly", "m"}:
        key_fn = lambda dt: (int(dt.year), int(dt.month))
    elif freq in {"year", "annual", "y"}:
        key_fn = lambda dt: (int(dt.year),)
    elif freq in {"all", "single"}:
        key_fn = lambda dt: (0,)
    else:
        key_fn = lambda dt: (int(dt.year), int((dt.month - 1) // 3) + 1)
    groups: list[list[pd.Timestamp]] = []
    current_key: tuple[int, ...] | None = None
    current: list[pd.Timestamp] = []
    for raw_dt in dates:
        dt = pd.Timestamp(raw_dt).normalize()
        key = key_fn(dt)
        if current and key != current_key:
            groups.append(current)
            current = []
        current_key = key
        current.append(dt)
    if current:
        groups.append(current)
    return groups


def _write_empty_shard(sample_path: Path, daily_path: Path) -> None:
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame().to_parquet(sample_path, index=False)
    pd.DataFrame().to_parquet(daily_path, index=False)


def _label_observed(frame: pd.DataFrame, *, strict_end_date: str) -> pd.Series:
    if frame.empty or "date" not in frame.columns or not strict_end_date:
        return pd.Series([False] * len(frame), index=frame.index)
    dates = pd.to_datetime(frame["date"], errors="coerce")
    return dates.le(pd.Timestamp(strict_end_date)).fillna(False)


def _summarize_label_completeness(
    *,
    shard_records: Sequence[Mapping[str, Any]],
    zone: str,
    strict_end_date: str,
    max_forward_horizon: int,
) -> dict[str, Any]:
    sample_rows = int(sum(int(item.get("sample_rows", 0) or 0) for item in shard_records))
    daily_rows = int(sum(int(item.get("daily_rows", 0) or 0) for item in shard_records))
    observed = int(sum(int(item.get("observed_label_rows", 0) or 0) for item in shard_records))
    daily_observed = int(sum(int(item.get("daily_observed_rows", 0) or 0) for item in shard_records))
    unobserved = int(sample_rows - observed)
    daily_unobserved = int(daily_rows - daily_observed)
    resolved_zone = _normalize_zone(zone)
    return {
        "zone": resolved_zone,
        "max_forward_horizon": int(max_forward_horizon),
        "strict_end_date": str(strict_end_date or ""),
        "sample_rows": sample_rows,
        "daily_rows": daily_rows,
        "observed_label_rows": observed,
        "unobserved_label_rows": unobserved,
        "observed_label_row_ratio": float(observed / sample_rows) if sample_rows else 0.0,
        "daily_observed_rows": daily_observed,
        "daily_unobserved_rows": daily_unobserved,
        "is_training_safe": bool(resolved_zone == "strict_train" and sample_rows > 0 and unobserved == 0),
        "sharded": True,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def _build_day_training_rows(
    *,
    prepared: PreparedPolicyInputs,
    future_metrics: FuturePathMetrics,
    signal_dt: pd.Timestamp,
    next_dt: pd.Timestamp | None,
    portfolio: PortfolioState,
    label_config: Any,
    budget_objective: str,
    execution_semantics: str,
    budget_semantics: str,
    budget_calibration: str,
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]], float | None, dict[str, float], dict[str, float], dict[str, float]]:
    state_frame = build_cross_section_state(prepared, date=signal_dt, portfolio_state=portfolio)
    label_frame = build_action_labels_for_date(
        date=signal_dt,
        state_frame=state_frame,
        future_metrics=future_metrics,
        label_config=label_config,
    )
    label_frame = attach_unified_allocation_targets(label_frame)
    unified_summary = build_unified_allocation_summary(label_frame)
    allocation_summary = build_allocation_teacher_summary(label_frame)
    daily_features = build_daily_state_features(state_frame)
    global_targets = build_teacher_global_targets(label_frame, label_config=label_config)
    global_targets, budget_diagnostics = _apply_budget_objective_targets(
        label_frame=label_frame,
        global_targets=global_targets,
        budget_objective=budget_objective,
    )
    teacher_policy = build_teacher_policy_frame(label_frame)
    step_result = portfolio.step(
        date=signal_dt,
        prices=prepared.close.loc[signal_dt],
        policy_frame=teacher_policy,
        global_targets=global_targets,
        source_label="teacher",
        execution_semantics=execution_semantics,
        budget_semantics=budget_semantics,
        budget_calibration=budget_calibration,
    )
    label_frame = _annotate_label_frame_with_execution_feedback(label_frame, step_result.actions)
    realized_return: float | None = None
    if next_dt is not None:
        next_returns = (
            prepared.close.loc[next_dt]
            .div(prepared.close.loc[signal_dt])
            .sub(1.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        gross_return = float(step_result.weights.reindex(next_returns.index).fillna(0.0).mul(next_returns).sum())
        trading_cost = estimate_trading_cost(
            diagnostics=step_result.diagnostics,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            sell_tax_bps=sell_tax_bps,
        )
        realized_return = gross_return - trading_cost
        portfolio.record_realized_return(realized_return)
    daily_row = {"date": _date_text(signal_dt), **daily_features, **global_targets}
    return (
        label_frame,
        daily_row,
        step_result.actions,
        realized_return,
        budget_diagnostics,
        allocation_summary,
        unified_summary,
    )


def _sample_training_rows(frame: pd.DataFrame, *, rng: np.random.Generator, skip_multiplier: float) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    full = frame.replace([np.inf, -np.inf], np.nan)
    interesting = full.loc[full["action_label"].astype(str) != "skip"]
    skip_frame = full.loc[full["action_label"].astype(str) == "skip"]
    skip_cap = min(len(skip_frame), max(int(len(interesting) * float(skip_multiplier)), 4_000))
    if len(skip_frame) > skip_cap > 0:
        skip_frame = skip_frame.sample(n=skip_cap, random_state=int(rng.integers(0, 2**31 - 1)))
    return (
        pd.concat([interesting, skip_frame], ignore_index=True)
        .sort_values(["date", "stock"], ascending=[True, True])
        .reset_index(drop=True)
    )


def build_sharded_gold_training_dataset(
    *,
    lake: ResearchDataLake,
    prepared: PreparedPolicyInputs,
    future_metrics: FuturePathMetrics,
    build_spec: GoldBuildSpec,
    zone: str,
    resume: bool = True,
    refresh: bool = False,
    min_signal_dates: int = 25,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    resolved_zone = _normalize_zone(zone)
    max_forward_horizon = max(int(item) for item in getattr(future_metrics, "horizons", (20,)))
    trim_horizon = max_forward_horizon if resolved_zone == "strict_train" else 0
    dates = signal_dates_between(
        prepared,
        start_date=build_spec.start_date,
        end_date=build_spec.end_date,
        max_forward_horizon=trim_horizon,
    )
    if len(dates) < int(min_signal_dates):
        raise ValueError("Continuous-policy Gold build window is too short after forward-horizon trimming.")
    available_dates = list(prepared.close.index)
    strict_end_date = _available_strict_end(available_dates, max_forward_horizon)
    spec = build_spec.to_spec_dict(prepared=prepared, zone=resolved_zone, max_forward_horizon=max_forward_horizon)
    identity = lake.build_training_dataset_identity(spec=spec, zone=resolved_zone)
    dataset_id = identity["dataset_id"]
    dataset_dir = Path(identity["dataset_dir"])
    if refresh and dataset_dir.exists():
        # Avoid shell deletion and keep this constrained to the computed dataset directory.
        import shutil

        shutil.rmtree(dataset_dir)
    sample_dir = dataset_dir / "sample_frame"
    daily_dir = dataset_dir / "daily_frame"
    summary_dir = dataset_dir / "teacher_summary"
    checkpoint_dir = dataset_dir / "portfolio_checkpoint"
    for path in (sample_dir, daily_dir, summary_dir, checkpoint_dir):
        path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(build_spec.random_seed))
    label_config = resolve_label_config(build_spec.label_preset)
    date_groups = _split_dates_by_frequency(dates, build_spec.shard_frequency)
    portfolio = PortfolioState()
    shard_records: list[dict[str, Any]] = []
    all_teacher_returns: list[float] = []
    all_teacher_return_dates: list[str] = []
    all_action_counts: dict[str, int] = {}
    all_full_action_counts: dict[str, int] = {}
    budget_rows: list[dict[str, float]] = []
    allocation_rows: list[dict[str, float]] = []
    unified_rows: list[dict[str, float]] = []

    previous_checkpoint_path: Path | None = None
    for shard_index, shard_dates in enumerate(date_groups):
        shard_start = shard_dates[0]
        shard_end = shard_dates[-1]
        key = _shard_key(shard_start, shard_end)
        sample_path = sample_dir / f"{key}.parquet"
        daily_path = daily_dir / f"{key}.parquet"
        summary_path = summary_dir / f"{key}.json"
        checkpoint_path = checkpoint_dir / f"{_date_text(shard_end)}.json"
        if resume and sample_path.exists() and daily_path.exists() and summary_path.exists() and checkpoint_path.exists():
            summary = _read_json(summary_path)
            portfolio = PortfolioState.from_snapshot(_read_json(checkpoint_path).get("portfolio_snapshot", {}))
            shard_record = dict(summary.get("shard_record", {}) or {})
            if shard_record:
                shard_records.append(shard_record)
                for action, count in dict(summary.get("teacher_action_distribution", {}) or {}).items():
                    all_action_counts[str(action)] = all_action_counts.get(str(action), 0) + int(count)
                for action, count in dict(summary.get("teacher_full_action_distribution", {}) or {}).items():
                    all_full_action_counts[str(action)] = all_full_action_counts.get(str(action), 0) + int(count)
                for date_value, return_value in zip(summary.get("teacher_return_dates", []) or [], summary.get("teacher_returns", []) or []):
                    all_teacher_return_dates.append(str(date_value))
                    all_teacher_returns.append(float(return_value))
                budget_rows.extend(list(summary.get("budget_diagnostic_rows", []) or []))
                allocation_rows.extend(list(summary.get("allocation_teacher_rows", []) or []))
                unified_rows.extend(list(summary.get("unified_allocation_rows", []) or []))
            previous_checkpoint_path = checkpoint_path
            if progress:
                progress("gold_shard_resume_hit", {"dataset_id": dataset_id, "zone": resolved_zone, "shard": key})
            continue
        if previous_checkpoint_path is not None and previous_checkpoint_path.exists():
            portfolio = PortfolioState.from_snapshot(_read_json(previous_checkpoint_path).get("portfolio_snapshot", {}))

        sample_frames: list[pd.DataFrame] = []
        daily_rows: list[dict[str, Any]] = []
        action_rows: list[dict[str, Any]] = []
        shard_returns: list[float] = []
        shard_return_dates: list[str] = []
        for local_index, signal_dt in enumerate(shard_dates):
            global_index = dates.index(signal_dt)
            next_dt = dates[global_index + 1] if global_index + 1 < len(dates) else None
            (
                label_frame,
                daily_row,
                actions,
                realized_return,
                budget_diagnostics,
                allocation_summary,
                unified_summary,
            ) = _build_day_training_rows(
                prepared=prepared,
                future_metrics=future_metrics,
                signal_dt=signal_dt,
                next_dt=next_dt,
                portfolio=portfolio,
                label_config=label_config,
                budget_objective=build_spec.budget_objective,
                execution_semantics=build_spec.execution_semantics,
                budget_semantics=build_spec.budget_semantics,
                budget_calibration=build_spec.budget_calibration,
                transaction_cost_bps=build_spec.transaction_cost_bps,
                slippage_bps=build_spec.slippage_bps,
                sell_tax_bps=build_spec.sell_tax_bps,
            )
            sample_frames.append(label_frame)
            daily_rows.append(daily_row)
            action_rows.extend(actions)
            budget_rows.append(budget_diagnostics)
            allocation_rows.append(allocation_summary)
            unified_rows.append(unified_summary)
            if realized_return is not None and next_dt is not None:
                shard_returns.append(float(realized_return))
                shard_return_dates.append(_date_text(next_dt))
            if progress and (local_index == 0 or local_index == len(shard_dates) - 1 or (local_index + 1) % 20 == 0):
                progress(
                    "gold_shard_day_progress",
                    {
                        "dataset_id": dataset_id,
                        "zone": resolved_zone,
                        "shard": key,
                        "day_index": local_index + 1,
                        "day_count": len(shard_dates),
                        "date": _date_text(signal_dt),
                    },
                )

        if sample_frames:
            full_sample = pd.concat(sample_frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
            sampled = _sample_training_rows(full_sample, rng=rng, skip_multiplier=build_spec.skip_multiplier)
        else:
            full_sample = pd.DataFrame()
            sampled = pd.DataFrame()
        daily_frame = pd.DataFrame(daily_rows).replace([np.inf, -np.inf], np.nan)
        observed_sample = _label_observed(sampled, strict_end_date=strict_end_date)
        observed_daily = _label_observed(daily_frame, strict_end_date=strict_end_date)
        if resolved_zone == "realtime_research":
            sampled = sampled.copy()
            daily_frame = daily_frame.copy()
            sampled["is_observed"] = observed_sample.to_numpy(dtype=bool)
            daily_frame["is_observed"] = observed_daily.to_numpy(dtype=bool)
        if resolved_zone == "strict_train":
            sampled = sampled.loc[observed_sample].reset_index(drop=True)
            daily_frame = daily_frame.loc[observed_daily].reset_index(drop=True)

        if sampled.empty and daily_frame.empty:
            _write_empty_shard(sample_path, daily_path)
        else:
            sampled.to_parquet(sample_path, index=False)
            daily_frame.to_parquet(daily_path, index=False)
        action_panel = pd.DataFrame(action_rows)
        action_distribution = {
            str(k): int(v)
            for k, v in sampled.get("action_label", pd.Series(dtype=str)).astype(str).value_counts().sort_index().items()
        }
        full_action_distribution = {
            str(k): int(v)
            for k, v in full_sample.get("action_label", pd.Series(dtype=str)).astype(str).value_counts().sort_index().items()
        }
        for action, count in action_distribution.items():
            all_action_counts[action] = all_action_counts.get(action, 0) + int(count)
        for action, count in full_action_distribution.items():
            all_full_action_counts[action] = all_full_action_counts.get(action, 0) + int(count)
        all_teacher_returns.extend(shard_returns)
        all_teacher_return_dates.extend(shard_return_dates)
        shard_record = {
            "shard_index": int(shard_index),
            "shard_key": key,
            "status": "stored",
            "start_date": _date_text(shard_start),
            "end_date": _date_text(shard_end),
            "sample_path": str(sample_path.resolve()),
            "daily_path": str(daily_path.resolve()),
            "teacher_summary_path": str(summary_path.resolve()),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "sample_rows": int(len(sampled)),
            "daily_rows": int(len(daily_frame)),
            "full_sample_rows": int(len(full_sample)),
            "action_rows": int(len(action_panel)),
            "observed_label_rows": int(_label_observed(sampled, strict_end_date=strict_end_date).sum()),
            "unobserved_label_rows": int(len(sampled) - int(_label_observed(sampled, strict_end_date=strict_end_date).sum())),
            "daily_observed_rows": int(_label_observed(daily_frame, strict_end_date=strict_end_date).sum()),
            "daily_unobserved_rows": int(len(daily_frame) - int(_label_observed(daily_frame, strict_end_date=strict_end_date).sum())),
        }
        shard_summary = {
            "dataset_id": dataset_id,
            "fingerprint": identity["fingerprint"],
            "zone": resolved_zone,
            "shard_record": shard_record,
            "teacher_action_distribution": action_distribution,
            "teacher_full_action_distribution": full_action_distribution,
            "teacher_return_dates": shard_return_dates,
            "teacher_returns": shard_returns,
            "budget_diagnostic_rows": budget_rows[-len(daily_rows):] if daily_rows else [],
            "allocation_teacher_rows": allocation_rows[-len(daily_rows):] if daily_rows else [],
            "unified_allocation_rows": unified_rows[-len(daily_rows):] if daily_rows else [],
            "teacher_recent_turnover_mean": float(action_panel["delta_weight"].abs().mean()) if not action_panel.empty and "delta_weight" in action_panel else 0.0,
        }
        _write_json(summary_path, shard_summary)
        _write_json(
            checkpoint_path,
            {
                "dataset_id": dataset_id,
                "fingerprint": identity["fingerprint"],
                "zone": resolved_zone,
                "shard_key": key,
                "date": _date_text(shard_end),
                "portfolio_snapshot": portfolio.snapshot(),
            },
        )
        previous_checkpoint_path = checkpoint_path
        shard_records.append(shard_record)
        if progress:
            progress("gold_shard_stored", shard_record)

    label_summary = _summarize_label_completeness(
        shard_records=shard_records,
        zone=resolved_zone,
        strict_end_date=strict_end_date,
        max_forward_horizon=max_forward_horizon,
    )
    teacher_metrics = compute_curve_metrics(pd.Series(all_teacher_returns, index=all_teacher_return_dates, dtype=float))
    teacher_summary = {
        "label_preset": label_config.name,
        "execution_semantics": build_spec.execution_semantics,
        "budget_semantics": build_spec.budget_semantics,
        "budget_calibration": build_spec.budget_calibration,
        "budget_objective": build_spec.budget_objective,
        "teacher_action_distribution": all_action_counts,
        "teacher_full_action_distribution": all_full_action_counts,
        "teacher_rollout_metrics": teacher_metrics,
        "budget_objective_diagnostics_mean": {
            str(column): float(pd.DataFrame(budget_rows)[column].mean())
            for column in (pd.DataFrame(budget_rows).columns if budget_rows else [])
        },
        "allocation_teacher_summary_mean": summarize_allocation_teacher_rows(allocation_rows),
        "unified_allocation_summary_mean": summarize_unified_allocation_rows(unified_rows),
        "full_sample_rows": int(sum(int(item.get("full_sample_rows", 0) or 0) for item in shard_records)),
        "train_sample_rows": int(label_summary["sample_rows"]),
        "daily_rows": int(label_summary["daily_rows"]),
        "sharded": True,
    }
    record = lake.save_sharded_training_dataset(
        spec=spec,
        zone=resolved_zone,
        shard_records=shard_records,
        teacher_summary=teacher_summary,
        label_completeness_summary=label_summary,
        source_cache={
            "source_market_dataset_id": build_spec.source_market_dataset_id,
            "prepared_summary": prepared.to_summary(),
            "build_spec": spec,
        },
        status="stored",
    )
    manifest = {
        "status": "ok",
        "dataset_id": record.dataset_id,
        "zone": record.zone,
        "fingerprint": record.fingerprint,
        "dataset_root": str(Path(identity["dataset_dir"]).resolve()),
        "shard_count": int(len(shard_records)),
        "completed_shard_count": int(sum(1 for item in shard_records if item.get("status") == "stored")),
        "label_completeness_summary": label_summary,
        "row_counts": record.row_counts,
    }
    _write_json(Path(identity["dataset_dir"]) / "build_manifest.json", manifest)
    if progress:
        progress("gold_dataset_registered", manifest)
    return _json_safe(manifest)
