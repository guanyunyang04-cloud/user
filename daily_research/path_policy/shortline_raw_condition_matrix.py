from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from daily_research.path_policy.shortline_raw_upside_diagnostic import (
    DEFAULT_LABEL_HORIZONS,
    DEFAULT_ROLE_YEARS,
    DEFAULT_ROLES,
    DEFAULT_SHARDED_MANIFEST,
    DEFAULT_TARGET_KINDS,
    _exit_label,
    _json_default,
    _now,
    _open_label_arrays,
    _parse_csv_ints,
    _parse_csv_strings,
    _parse_role_years,
    _role_for_year,
    _safe_div,
    _safe_float,
    _selected_shards,
    _target_array,
    _write_frame,
    _write_json,
)


DEFAULT_FEATURES = (
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "turn",
    "turn_z20",
    "raw_open_gap_1d",
    "raw_intraday_range_1d",
    "distance_to_20d_high",
    "local_drawdown_20d",
    "vol_20d",
    "intraday_last_30m_ret",
    "intraday_close_position",
    "intraday_low_time_frac",
    "intraday_high_before_low",
    "intraday_intraday_range",
    "market_positive_share_1d",
    "market_above_ma20_share",
    "market_amount_expansion_share",
    "industry_ret_5_excess",
    "industry_ret_20_excess",
    "stock_ret_5_minus_industry",
    "stock_ret_20_minus_industry",
)


@dataclass(frozen=True)
class Clause:
    feature: str
    op: str
    threshold_ref: str

    def threshold_value(self, thresholds: Mapping[str, Mapping[str, float]]) -> float:
        ref = str(self.threshold_ref)
        if ref.startswith("q"):
            feature_thresholds = dict(thresholds.get(self.feature, {}) or {})
            if ref not in feature_thresholds:
                raise KeyError(f"missing threshold {ref} for feature {self.feature}")
            return float(feature_thresholds[ref])
        return float(ref)

    def mask(self, values: Mapping[str, np.ndarray], thresholds: Mapping[str, Mapping[str, float]]) -> np.ndarray:
        if self.feature not in values:
            raise KeyError(f"missing feature values for condition: {self.feature}")
        x = np.asarray(values[self.feature], dtype=np.float32)
        value = self.threshold_value(thresholds)
        valid = np.isfinite(x)
        if self.op == "<=":
            return valid & (x <= value)
        if self.op == "<":
            return valid & (x < value)
        if self.op == ">=":
            return valid & (x >= value)
        if self.op == ">":
            return valid & (x > value)
        if self.op == "between_low":
            return valid & (x >= value)
        if self.op == "between_high":
            return valid & (x <= value)
        raise ValueError(f"unsupported clause op: {self.op}")


@dataclass(frozen=True)
class Condition:
    name: str
    mechanism: str
    description: str
    clauses: tuple[Clause, ...]

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(clause.feature for clause in self.clauses))

    def mask(self, values: Mapping[str, np.ndarray], thresholds: Mapping[str, Mapping[str, float]]) -> np.ndarray:
        out: np.ndarray | None = None
        for clause in self.clauses:
            current = clause.mask(values, thresholds)
            out = current if out is None else (out & current)
        if out is None:
            first = next(iter(values.values()))
            return np.ones_like(np.asarray(first), dtype=bool)
        return out

    def definition_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, clause in enumerate(self.clauses):
            rows.append(
                {
                    "condition": self.name,
                    "mechanism": self.mechanism,
                    "description": self.description,
                    "clause_index": int(idx),
                    "feature": clause.feature,
                    "op": clause.op,
                    "threshold_ref": clause.threshold_ref,
                }
            )
        return rows


DEFAULT_CONDITIONS = (
    Condition(
        name="anti_overheat_core",
        mechanism="anti_overheat",
        description="low/medium recent runup, low/medium turnover, and not overheated versus industry",
        clauses=(
            Clause("ret_20d", "<=", "q40"),
            Clause("turn", "<=", "q60"),
            Clause("stock_ret_20_minus_industry", "<=", "q60"),
        ),
    ),
    Condition(
        name="low_runup_late_intraday_low",
        mechanism="pullback_repair",
        description="not high recent runup and intraday low occurs late in the day",
        clauses=(
            Clause("ret_20d", "<=", "q40"),
            Clause("intraday_low_time_frac", ">=", "q70"),
        ),
    ),
    Condition(
        name="low_runup_high_before_low",
        mechanism="pullback_repair",
        description="not high recent runup and high-before-low path flag is present",
        clauses=(
            Clause("ret_20d", "<=", "q40"),
            Clause("intraday_high_before_low", ">=", "0.5"),
        ),
    ),
    Condition(
        name="low_mid_turn_mild_positive_gap",
        mechanism="entry_quality",
        description="low/medium turnover with non-negative but not extreme open gap",
        clauses=(
            Clause("turn", "<=", "q60"),
            Clause("raw_open_gap_1d", ">=", "0.0"),
            Clause("raw_open_gap_1d", "<=", "q75"),
        ),
    ),
    Condition(
        name="pullback_intraday_recovery",
        mechanism="pullback_repair",
        description="drawdown-like state with acceptable intraday close position",
        clauses=(
            Clause("ret_20d", "<=", "q50"),
            Clause("local_drawdown_20d", "<=", "q40"),
            Clause("intraday_close_position", ">=", "q50"),
        ),
    ),
    Condition(
        name="industry_not_collapsing_not_overheat",
        mechanism="industry_context",
        description="stock is not overheated versus industry and industry short return is not in the weakest tail",
        clauses=(
            Clause("stock_ret_20_minus_industry", "<=", "q60"),
            Clause("industry_ret_5_excess", ">=", "q30"),
        ),
    ),
    Condition(
        name="market_confirmed_anti_overheat",
        mechanism="market_context",
        description="not high runup, with market breadth/context not broken",
        clauses=(
            Clause("ret_20d", "<=", "q40"),
            Clause("market_positive_share_1d", ">=", "q50"),
            Clause("market_above_ma20_share", ">=", "q40"),
        ),
    ),
    Condition(
        name="quiet_vol_pullback",
        mechanism="pullback_repair",
        description="low/medium volatility and intraday range with not high recent return",
        clauses=(
            Clause("vol_20d", "<=", "q60"),
            Clause("ret_10d", "<=", "q40"),
            Clause("intraday_intraday_range", "<=", "q60"),
        ),
    ),
    Condition(
        name="mild_gap_late_low",
        mechanism="entry_quality",
        description="mild positive gap with late intraday low",
        clauses=(
            Clause("raw_open_gap_1d", ">=", "0.0"),
            Clause("raw_open_gap_1d", "<=", "q75"),
            Clause("intraday_low_time_frac", ">=", "q70"),
        ),
    ),
    Condition(
        name="anti_chase_late_strength",
        mechanism="anti_overheat",
        description="not crowded, not high recent return, and late-day intraday strength",
        clauses=(
            Clause("turn", "<=", "q50"),
            Clause("ret_10d", "<=", "q40"),
            Clause("intraday_last_30m_ret", ">=", "q60"),
        ),
    ),
)


@dataclass
class FeatureSampler:
    feature: str
    max_samples: int

    def __post_init__(self) -> None:
        self.parts: list[np.ndarray] = []
        self.count = 0
        self.finite_count = 0

    def update(self, values: np.ndarray) -> None:
        x = np.asarray(values, dtype=np.float32).reshape(-1)
        x = x[np.isfinite(x)]
        if len(x) <= 0:
            return
        self.finite_count += int(len(x))
        remaining = int(self.max_samples) - int(self.count)
        if remaining <= 0:
            return
        if len(x) > remaining:
            idx = np.linspace(0, len(x) - 1, num=remaining, dtype=np.int64)
            x = x[idx]
        self.parts.append(np.asarray(x, dtype=np.float32))
        self.count += int(len(x))

    def quantiles(self, qs: Iterable[float]) -> dict[str, float]:
        if not self.parts:
            return {}
        x = np.concatenate(self.parts).astype(np.float64)
        out: dict[str, float] = {}
        for q in qs:
            out[f"q{int(round(float(q) * 100)):02d}"] = float(np.nanquantile(x, float(q)))
        return out


@dataclass
class MetricAccumulator:
    count: int = 0
    target_sum: float = 0.0
    target_sq_sum: float = 0.0
    hit_sum: float = 0.0
    entry_tradeable_sum: float = 0.0
    entry_limit_up_blocked_sum: float = 0.0
    entry_suspended_or_no_open_sum: float = 0.0

    def update(
        self,
        target: np.ndarray,
        mask: np.ndarray,
        *,
        entry_tradeable: np.ndarray | None = None,
        entry_limit_up_buy_blocked: np.ndarray | None = None,
        entry_suspended_or_no_open: np.ndarray | None = None,
    ) -> None:
        y = np.asarray(target, dtype=np.float64)
        selected = np.asarray(mask, dtype=bool) & np.isfinite(y)
        n = int(selected.sum())
        if n <= 0:
            return
        values = y[selected]
        self.count += n
        self.target_sum += float(values.sum())
        self.target_sq_sum += float(np.square(values).sum())
        self.hit_sum += float((values > 0.0).sum())
        if entry_tradeable is not None:
            self.entry_tradeable_sum += float(np.nan_to_num(np.asarray(entry_tradeable, dtype=np.float64)[selected], nan=0.0).sum())
        if entry_limit_up_buy_blocked is not None:
            self.entry_limit_up_blocked_sum += float(
                np.nan_to_num(np.asarray(entry_limit_up_buy_blocked, dtype=np.float64)[selected], nan=0.0).sum()
            )
        if entry_suspended_or_no_open is not None:
            self.entry_suspended_or_no_open_sum += float(
                np.nan_to_num(np.asarray(entry_suspended_or_no_open, dtype=np.float64)[selected], nan=0.0).sum()
            )

    def row(self) -> dict[str, Any]:
        count = float(self.count)
        mean = _safe_div(self.target_sum, count)
        var = _safe_div(self.target_sq_sum, count) - mean * mean if count else float("nan")
        return {
            "count": int(self.count),
            "target_mean": mean,
            "target_std": math.sqrt(max(var, 0.0)) if math.isfinite(var) else float("nan"),
            "hit_rate": _safe_div(self.hit_sum, count),
            "entry_tradeable_rate": _safe_div(self.entry_tradeable_sum, count),
            "entry_limit_up_buy_blocked_rate": _safe_div(self.entry_limit_up_blocked_sum, count),
            "entry_suspended_or_no_open_rate": _safe_div(self.entry_suspended_or_no_open_sum, count),
        }


def _required_features(conditions: Iterable[Condition], requested: tuple[str, ...]) -> tuple[str, ...]:
    values = list(requested)
    for condition in conditions:
        values.extend(condition.features)
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


def _threshold_features(conditions: Iterable[Condition]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for condition in conditions:
        for clause in condition.clauses:
            if str(clause.threshold_ref).startswith("q") and clause.feature not in seen:
                seen.add(clause.feature)
                out.append(clause.feature)
    return tuple(out)


def _build_train_thresholds(
    *,
    shards: list[dict[str, Any]],
    all_features: list[str],
    threshold_features: tuple[str, ...],
    max_samples_per_feature: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    indices = {feature: all_features.index(feature) for feature in threshold_features}
    samplers = {feature: FeatureSampler(feature, int(max_samples_per_feature)) for feature in threshold_features}
    for shard in shards:
        if str(shard.get("_role", "")) != "train":
            continue
        shape = tuple(int(item) for item in list(shard.get("feature_store_shape", []) or []))
        store = np.memmap(str(shard["feature_store_path"]), dtype="float32", mode="r", shape=shape)
        for feature, idx in indices.items():
            samplers[feature].update(np.asarray(store[:, :, int(idx)], dtype=np.float32))
        del store
    quantile_points = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80)
    thresholds = {feature: sampler.quantiles(quantile_points) for feature, sampler in samplers.items()}
    missing = [feature for feature, values in thresholds.items() if not values]
    if missing:
        raise ValueError(f"failed to build train thresholds for features: {missing}")
    coverage = {
        feature: {
            "finite_count": int(sampler.finite_count),
            "sampled_count": int(sampler.count),
            "max_samples": int(sampler.max_samples),
        }
        for feature, sampler in samplers.items()
    }
    return thresholds, coverage


def _condition_values(
    *,
    store: np.memmap,
    feature_indices: Mapping[str, int],
) -> dict[str, np.ndarray]:
    return {feature: np.asarray(store[:, :, int(idx)], dtype=np.float32) for feature, idx in feature_indices.items()}


def _open_optional_label_arrays(label_manifest: Mapping[str, Any]) -> dict[str, np.memmap]:
    names = ("entry_tradeable", "entry_limit_up_buy_blocked", "entry_suspended_or_no_open")
    arrays_meta = dict(label_manifest.get("arrays", {}) or {})
    available = [name for name in names if name in arrays_meta]
    return _open_label_arrays(label_manifest, available) if available else {}


def _update_monthly(
    *,
    monthly: dict[tuple[str, str, int, str, str], MetricAccumulator],
    role: str,
    target_kind: str,
    horizon: int,
    condition: str,
    date_values: list[str],
    target: np.ndarray,
    mask: np.ndarray,
    entry_tradeable: np.ndarray | None,
    entry_limit_up_buy_blocked: np.ndarray | None,
    entry_suspended_or_no_open: np.ndarray | None,
) -> None:
    months = [str(pd.Timestamp(item).strftime("%Y-%m")) for item in date_values]
    for date_idx, month in enumerate(months):
        row_mask = np.asarray(mask[date_idx, :], dtype=bool)
        if int(row_mask.sum()) <= 0:
            continue
        key = (role, target_kind, int(horizon), condition, month)
        if key not in monthly:
            monthly[key] = MetricAccumulator()
        monthly[key].update(
            target[date_idx, :],
            row_mask,
            entry_tradeable=entry_tradeable[date_idx, :] if entry_tradeable is not None else None,
            entry_limit_up_buy_blocked=entry_limit_up_buy_blocked[date_idx, :] if entry_limit_up_buy_blocked is not None else None,
            entry_suspended_or_no_open=entry_suspended_or_no_open[date_idx, :] if entry_suspended_or_no_open is not None else None,
        )


def _metrics_frames(
    *,
    overall: Mapping[tuple[str, str, int, str], MetricAccumulator],
    monthly: Mapping[tuple[str, str, int, str, str], MetricAccumulator],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for (role, target_kind, horizon, condition), acc in overall.items():
        if condition != "__baseline__":
            continue
        baseline_rows[(role, target_kind, int(horizon))] = acc.row()
    metric_rows: list[dict[str, Any]] = []
    for (role, target_kind, horizon, condition), acc in sorted(overall.items()):
        if condition == "__baseline__":
            continue
        row = acc.row()
        baseline = baseline_rows.get((role, target_kind, int(horizon)), {})
        baseline_count = int(baseline.get("count", 0) or 0)
        baseline_mean = _safe_float(baseline.get("target_mean"))
        row.update(
            {
                "role": role,
                "target_kind": target_kind,
                "label_horizon": int(horizon),
                "exit_label": _exit_label(int(horizon)),
                "condition": condition,
                "baseline_count": baseline_count,
                "selected_rate": _safe_div(float(row["count"]), float(baseline_count)),
                "baseline_target_mean": baseline_mean,
                "lift_vs_baseline": row["target_mean"] - baseline_mean if math.isfinite(row["target_mean"]) and math.isfinite(baseline_mean) else float("nan"),
                "baseline_hit_rate": _safe_float(baseline.get("hit_rate")),
            }
        )
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)

    baseline_months: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for (role, target_kind, horizon, condition, month), acc in monthly.items():
        if condition != "__baseline__":
            continue
        baseline_months[(role, target_kind, int(horizon), month)] = acc.row()
    month_rows: list[dict[str, Any]] = []
    for (role, target_kind, horizon, condition, month), acc in sorted(monthly.items()):
        if condition == "__baseline__":
            continue
        row = acc.row()
        baseline = baseline_months.get((role, target_kind, int(horizon), month), {})
        baseline_count = int(baseline.get("count", 0) or 0)
        baseline_mean = _safe_float(baseline.get("target_mean"))
        row.update(
            {
                "role": role,
                "target_kind": target_kind,
                "label_horizon": int(horizon),
                "exit_label": _exit_label(int(horizon)),
                "condition": condition,
                "year_month": month,
                "baseline_count": baseline_count,
                "selected_rate": _safe_div(float(row["count"]), float(baseline_count)),
                "baseline_target_mean": baseline_mean,
                "lift_vs_baseline": row["target_mean"] - baseline_mean if math.isfinite(row["target_mean"]) and math.isfinite(baseline_mean) else float("nan"),
                "baseline_hit_rate": _safe_float(baseline.get("hit_rate")),
            }
        )
        month_rows.append(row)
    monthly_frame = pd.DataFrame(month_rows)
    return metrics, monthly_frame


def _stable_conditions(metrics: pd.DataFrame, monthly: pd.DataFrame, *, min_count: int) -> pd.DataFrame:
    columns = [
        "condition",
        "target_kind",
        "label_horizon",
        "exit_label",
        "train_target_mean",
        "validation_target_mean",
        "test_target_mean",
        "train_lift",
        "validation_lift",
        "test_lift",
        "min_role_target_mean",
        "min_role_lift",
        "all_roles_positive_net",
        "all_roles_positive_lift",
        "validation_test_positive_net",
        "positive_month_rate",
        "positive_lift_month_rate",
        "worst_month_target_mean",
        "worst_month_lift",
    ]
    if metrics.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    required_roles = ("train", "validation", "test")
    for (condition, target_kind, horizon), group in metrics.groupby(["condition", "target_kind", "label_horizon"], sort=True):
        by_role = {str(row["role"]): row for row in group.to_dict("records")}
        if not all(role in by_role for role in required_roles):
            continue
        role_count_ok = all(int(by_role[role].get("count", 0) or 0) >= int(min_count) for role in required_roles)
        if not role_count_ok:
            continue
        target_means = {role: _safe_float(by_role[role].get("target_mean")) for role in required_roles}
        lifts = {role: _safe_float(by_role[role].get("lift_vs_baseline")) for role in required_roles}
        monthly_slice = monthly.loc[
            monthly["condition"].astype(str).eq(str(condition))
            & monthly["target_kind"].astype(str).eq(str(target_kind))
            & monthly["label_horizon"].astype(int).eq(int(horizon))
            & monthly["count"].astype(int).ge(int(min_count))
        ].copy() if not monthly.empty else pd.DataFrame()
        positive_month_rate = float((monthly_slice["target_mean"] > 0.0).mean()) if not monthly_slice.empty else float("nan")
        positive_lift_month_rate = float((monthly_slice["lift_vs_baseline"] > 0.0).mean()) if not monthly_slice.empty else float("nan")
        worst_month_target = float(monthly_slice["target_mean"].min()) if not monthly_slice.empty else float("nan")
        worst_month_lift = float(monthly_slice["lift_vs_baseline"].min()) if not monthly_slice.empty else float("nan")
        rows.append(
            {
                "condition": str(condition),
                "target_kind": str(target_kind),
                "label_horizon": int(horizon),
                "exit_label": _exit_label(int(horizon)),
                "train_target_mean": target_means["train"],
                "validation_target_mean": target_means["validation"],
                "test_target_mean": target_means["test"],
                "train_lift": lifts["train"],
                "validation_lift": lifts["validation"],
                "test_lift": lifts["test"],
                "min_role_target_mean": min(target_means.values()),
                "min_role_lift": min(lifts.values()),
                "all_roles_positive_net": bool(all(value > 0.0 for value in target_means.values())),
                "all_roles_positive_lift": bool(all(value > 0.0 for value in lifts.values())),
                "validation_test_positive_net": bool(target_means["validation"] > 0.0 and target_means["test"] > 0.0),
                "positive_month_rate": positive_month_rate,
                "positive_lift_month_rate": positive_lift_month_rate,
                "worst_month_target_mean": worst_month_target,
                "worst_month_lift": worst_month_lift,
            }
        )
    out = pd.DataFrame(rows, columns=columns)
    if not out.empty:
        out = out.sort_values(
            ["validation_test_positive_net", "all_roles_positive_lift", "min_role_lift", "positive_lift_month_rate", "condition"],
            ascending=[False, False, False, False, True],
            kind="mergesort",
        )
    return out


def _threshold_rows(thresholds: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    rows = []
    for feature, values in sorted(thresholds.items()):
        row = {"feature": feature}
        row.update({key: float(value) for key, value in sorted(values.items())})
        rows.append(row)
    return pd.DataFrame(rows)


def _threshold_coverage_rows(coverage: Mapping[str, Mapping[str, int]]) -> pd.DataFrame:
    rows = []
    for feature, values in sorted(coverage.items()):
        row = {"feature": feature}
        row.update({key: int(value) for key, value in sorted(values.items())})
        rows.append(row)
    return pd.DataFrame(rows)


def _condition_definition_frame(conditions: Iterable[Condition], thresholds: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        for row in condition.definition_rows():
            threshold_ref = str(row["threshold_ref"])
            if threshold_ref.startswith("q"):
                row["threshold_value"] = float(dict(thresholds.get(str(row["feature"]), {}) or {}).get(threshold_ref, float("nan")))
            else:
                row["threshold_value"] = float(threshold_ref)
            rows.append(row)
    return pd.DataFrame(rows)


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Shortline Raw Condition Matrix",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Run tag: `{report.get('run_tag', '')}`",
        f"- Research-only: `{report.get('contract', {}).get('research_only', True)}`",
        f"- Feature source: `{report.get('contract', {}).get('feature_source', '')}`",
        f"- Threshold source: `{report.get('contract', {}).get('threshold_source', '')}`",
        f"- Rows scanned: `{report.get('input', {}).get('scanned_rows', 0)}`",
        f"- Shards scanned: `{report.get('input', {}).get('scanned_shards', 0)}`",
        f"- Condition count: `{report.get('input', {}).get('condition_count', 0)}`",
        "",
        "## Semantics",
        "",
        "- Conditions are built from D-close features only.",
        "- Numeric thresholds are estimated from the train role only, then applied unchanged to validation/test.",
        "- Labels use next-open entry to future-open exits; horizon 1 means D+1 open entry to D+2 open exit.",
        "- Entry tradeability/blocked rates are reported as diagnostics, not used to pre-filter candidates.",
        "",
        "## Stable Condition Candidates",
        "",
    ]
    stable_rows = list(report.get("stable_condition_preview", []) or [])
    if stable_rows:
        lines.append("| condition | target | exit | val_mean | test_mean | min_lift | pos_lift_month_rate |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for row in stable_rows[:20]:
            lines.append(
                "| {condition} | {target_kind} | {exit_label} | {val:.5f} | {test:.5f} | {lift:.5f} | {pm:.3f} |".format(
                    condition=row.get("condition", ""),
                    target_kind=row.get("target_kind", ""),
                    exit_label=row.get("exit_label", ""),
                    val=_safe_float(row.get("validation_target_mean")),
                    test=_safe_float(row.get("test_target_mean")),
                    lift=_safe_float(row.get("min_role_lift")),
                    pm=_safe_float(row.get("positive_lift_month_rate")),
                )
            )
    else:
        lines.append("- No stable condition candidates under current thresholds.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Thresholds: `{report.get('outputs', {}).get('thresholds_csv', '')}`",
            f"- Threshold coverage: `{report.get('outputs', {}).get('threshold_coverage_csv', '')}`",
            f"- Condition definitions: `{report.get('outputs', {}).get('condition_definitions_csv', '')}`",
            f"- Condition metrics: `{report.get('outputs', {}).get('condition_metrics_csv', '')}`",
            f"- Monthly metrics: `{report.get('outputs', {}).get('condition_monthly_csv', '')}`",
            f"- Stable conditions: `{report.get('outputs', {}).get('stable_conditions_csv', '')}`",
            f"- Report JSON: `{report.get('outputs', {}).get('report_json', '')}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_shortline_raw_condition_matrix(
    *,
    sharded_manifest_json: str | Path = DEFAULT_SHARDED_MANIFEST,
    output_root: str | Path | None = None,
    run_tag: str = "shortline_raw_condition_matrix_v1",
    roles: str | Iterable[str] | None = None,
    role_years: str | None = None,
    features: str | Iterable[str] | None = None,
    label_horizons: str | Iterable[int] | None = None,
    target_kinds: str | Iterable[str] | None = None,
    round_trip_cost_bps: float = 20.0,
    max_threshold_samples_per_feature: int = 500_000,
    max_shards_per_role: int = 0,
    min_count_for_stability: int = 500,
) -> dict[str, Any]:
    manifest_path = Path(sharded_manifest_json)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("artifact_type", "")) != "qdp_sharded_memmap":
        raise ValueError(f"requires qdp_sharded_memmap manifest: {manifest_path}")
    resolved_roles = _parse_csv_strings(roles, default=DEFAULT_ROLES)
    resolved_role_years = _parse_role_years(role_years)
    resolved_horizons = _parse_csv_ints(label_horizons, default=DEFAULT_LABEL_HORIZONS)
    resolved_targets = _parse_csv_strings(target_kinds, default=DEFAULT_TARGET_KINDS)
    requested_features = _parse_csv_strings(features, default=DEFAULT_FEATURES)
    conditions = DEFAULT_CONDITIONS
    required_features = _required_features(conditions, requested_features)
    all_features = list(str(item) for item in list(manifest.get("feature_columns", []) or []))
    all_feature_set = set(all_features)
    selected_features = tuple(feature for feature in required_features if feature in all_feature_set)
    missing = [feature for feature in required_features if feature not in all_feature_set]
    condition_missing = sorted({feature for condition in conditions for feature in condition.features if feature not in all_feature_set})
    if condition_missing:
        raise ValueError(f"default conditions require missing features: {condition_missing}")
    if not selected_features:
        raise ValueError(f"none of requested features are in manifest. missing={missing[:10]}")
    output_dir = Path(output_root) if output_root is not None else Path("daily_research/output/path_policy/shortline_raw_condition_matrix") / str(run_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = _selected_shards(manifest, role_years=resolved_role_years, roles=resolved_roles)
    if int(max_shards_per_role) > 0:
        kept: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for shard in shards:
            role = str(shard["_role"])
            counts.setdefault(role, 0)
            if counts[role] >= int(max_shards_per_role):
                continue
            kept.append(shard)
            counts[role] += 1
        shards = kept
    if not shards:
        raise ValueError("no completed shards selected.")
    thresholds, threshold_coverage = _build_train_thresholds(
        shards=shards,
        all_features=all_features,
        threshold_features=_threshold_features(conditions),
        max_samples_per_feature=int(max_threshold_samples_per_feature),
    )
    feature_indices = {feature: all_features.index(feature) for feature in selected_features}
    overall: dict[tuple[str, str, int, str], MetricAccumulator] = {}
    monthly: dict[tuple[str, str, int, str, str], MetricAccumulator] = {}
    progress_path = output_dir / "shortline_raw_condition_matrix_progress.json"
    scanned_rows = 0
    for shard_idx, shard in enumerate(shards, start=1):
        role = str(shard["_role"])
        shape = tuple(int(item) for item in list(shard.get("feature_store_shape", []) or []))
        store = np.memmap(str(shard["feature_store_path"]), dtype="float32", mode="r", shape=shape)
        values = _condition_values(store=store, feature_indices=feature_indices)
        condition_masks = {condition.name: condition.mask(values, thresholds) for condition in conditions}
        scanned_rows += int(shape[0] * shape[1])
        label_manifest = json.loads(Path(str(shard["label_manifest_json"])).read_text(encoding="utf-8"))
        arrays = _open_label_arrays(label_manifest, ("cumulative_return_1to20", "cumulative_excess_return_1to20"))
        optional_arrays = _open_optional_label_arrays(label_manifest)
        entry_tradeable = optional_arrays.get("entry_tradeable")
        entry_limit_up_buy_blocked = optional_arrays.get("entry_limit_up_buy_blocked")
        entry_suspended_or_no_open = optional_arrays.get("entry_suspended_or_no_open")
        date_values = [str(item) for item in list(label_manifest.get("date_values", []) or [])]
        if len(date_values) != int(shape[0]):
            date_values = [f"{int(shard.get('year', 0)):04d}-01-01"] * int(shape[0])
        for target_kind in resolved_targets:
            for horizon in resolved_horizons:
                target = _target_array(
                    arrays,
                    target_kind=str(target_kind),
                    label_horizon=int(horizon),
                    cost_bps=float(round_trip_cost_bps),
                )
                finite = np.isfinite(target)
                baseline_key = (role, str(target_kind), int(horizon), "__baseline__")
                if baseline_key not in overall:
                    overall[baseline_key] = MetricAccumulator()
                overall[baseline_key].update(
                    target,
                    finite,
                    entry_tradeable=entry_tradeable,
                    entry_limit_up_buy_blocked=entry_limit_up_buy_blocked,
                    entry_suspended_or_no_open=entry_suspended_or_no_open,
                )
                _update_monthly(
                    monthly=monthly,
                    role=role,
                    target_kind=str(target_kind),
                    horizon=int(horizon),
                    condition="__baseline__",
                    date_values=date_values,
                    target=target,
                    mask=finite,
                    entry_tradeable=entry_tradeable,
                    entry_limit_up_buy_blocked=entry_limit_up_buy_blocked,
                    entry_suspended_or_no_open=entry_suspended_or_no_open,
                )
                for condition in conditions:
                    mask = condition_masks[condition.name] & finite
                    key = (role, str(target_kind), int(horizon), condition.name)
                    if key not in overall:
                        overall[key] = MetricAccumulator()
                    overall[key].update(
                        target,
                        mask,
                        entry_tradeable=entry_tradeable,
                        entry_limit_up_buy_blocked=entry_limit_up_buy_blocked,
                        entry_suspended_or_no_open=entry_suspended_or_no_open,
                    )
                    _update_monthly(
                        monthly=monthly,
                        role=role,
                        target_kind=str(target_kind),
                        horizon=int(horizon),
                        condition=condition.name,
                        date_values=date_values,
                        target=target,
                        mask=mask,
                        entry_tradeable=entry_tradeable,
                        entry_limit_up_buy_blocked=entry_limit_up_buy_blocked,
                        entry_suspended_or_no_open=entry_suspended_or_no_open,
                    )
        del store
        if shard_idx % 20 == 0 or shard_idx == len(shards):
            _write_json(
                progress_path,
                {
                    "status": "running",
                    "processed_shards": int(shard_idx),
                    "total_shards": int(len(shards)),
                    "scanned_rows": int(scanned_rows),
                },
            )
    metrics, monthly_frame = _metrics_frames(overall=overall, monthly=monthly)
    definitions = _condition_definition_frame(conditions, thresholds)
    threshold_frame = _threshold_rows(thresholds)
    threshold_coverage_frame = _threshold_coverage_rows(threshold_coverage)
    stable = _stable_conditions(metrics, monthly_frame, min_count=int(min_count_for_stability))
    metrics_csv = _write_frame(output_dir / "shortline_raw_condition_metrics.csv", metrics)
    monthly_csv = _write_frame(output_dir / "shortline_raw_condition_monthly.csv", monthly_frame)
    thresholds_csv = _write_frame(output_dir / "shortline_raw_condition_thresholds.csv", threshold_frame)
    threshold_coverage_csv = _write_frame(output_dir / "shortline_raw_condition_threshold_coverage.csv", threshold_coverage_frame)
    definitions_csv = _write_frame(output_dir / "shortline_raw_condition_definitions.csv", definitions)
    stable_csv = _write_frame(output_dir / "shortline_raw_condition_stable_candidates.csv", stable)
    stable_preview = (
        stable.head(40)
        .replace([np.inf, -np.inf], np.nan)
        .where(pd.notna(stable.head(40)), None)
        .to_dict("records")
        if not stable.empty
        else []
    )
    report = {
        "run_tag": str(run_tag),
        "status": "completed" if not metrics.empty else "empty",
        "created_at": _now(),
        "sharded_manifest_json": str(manifest_path.resolve()),
        "contract": {
            "diagnostic_kind": "shortline_raw_condition_matrix_v1",
            "research_only": True,
            "not_model_training": True,
            "not_topk_strategy_selection": True,
            "active_artifact_impact": "unchanged",
            "feature_source": "qdp_sharded_memmap_forecast_feature_store_unnormalized_engineering_features",
            "threshold_source": "train_role_only",
            "label_semantics": "next_open_entry_to_future_open",
            "round_trip_cost_bps": float(round_trip_cost_bps),
            "roles": list(resolved_roles),
            "role_years": {role: list(years) for role, years in resolved_role_years.items()},
            "label_horizons": [int(item) for item in resolved_horizons],
            "target_kinds": list(resolved_targets),
            "min_count_for_stability": int(min_count_for_stability),
        },
        "input": {
            "scanned_shards": int(len(shards)),
            "scanned_rows": int(scanned_rows),
            "feature_count": int(len(selected_features)),
            "condition_count": int(len(conditions)),
            "features": list(selected_features),
            "missing_requested_features": missing,
            "threshold_feature_coverage": threshold_coverage,
            "source_feature_profile": str(manifest.get("feature_profile", "")),
            "source_label_schema": str(manifest.get("label_schema_name", "")),
            "source_label_schema_version": int(manifest.get("label_schema_version", 0) or 0),
        },
        "outputs": {
            "thresholds_csv": thresholds_csv,
            "threshold_coverage_csv": threshold_coverage_csv,
            "condition_definitions_csv": definitions_csv,
            "condition_metrics_csv": metrics_csv,
            "condition_monthly_csv": monthly_csv,
            "stable_conditions_csv": stable_csv,
        },
        "summary": {
            "metric_rows": int(len(metrics)),
            "monthly_rows": int(len(monthly_frame)),
            "stable_condition_rows": int(len(stable)),
            "validation_test_positive_net_rows": int(stable["validation_test_positive_net"].sum()) if not stable.empty else 0,
            "all_roles_positive_lift_rows": int(stable["all_roles_positive_lift"].sum()) if not stable.empty else 0,
        },
        "stable_condition_preview": stable_preview,
    }
    report_json = _write_json(output_dir / "shortline_raw_condition_matrix_report.json", report)
    report["outputs"]["report_json"] = report_json
    _write_markdown(output_dir / "shortline_raw_condition_matrix_report.md", report)
    report["outputs"]["report_md"] = str((output_dir / "shortline_raw_condition_matrix_report.md").resolve())
    _write_json(output_dir / "shortline_raw_condition_matrix_report.json", report)
    _write_json(progress_path, {"status": "completed", "report_json": report_json})
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate train-fixed raw shortline condition matrices.")
    parser.add_argument("--sharded-manifest-json", default=str(DEFAULT_SHARDED_MANIFEST))
    parser.add_argument("--output-root", default="")
    parser.add_argument("--run-tag", default="shortline_raw_condition_matrix_v1")
    parser.add_argument("--roles", default="train,validation,test")
    parser.add_argument("--role-years", default="")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--label-horizons", default="1,2,4")
    parser.add_argument("--target-kinds", default="net_abs,net_excess")
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    parser.add_argument("--max-threshold-samples-per-feature", type=int, default=500_000)
    parser.add_argument("--max-shards-per-role", type=int, default=0)
    parser.add_argument("--min-count-for-stability", type=int, default=500)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = build_shortline_raw_condition_matrix(
        sharded_manifest_json=args.sharded_manifest_json,
        output_root=args.output_root or None,
        run_tag=args.run_tag,
        roles=args.roles,
        role_years=args.role_years or None,
        features=args.features,
        label_horizons=args.label_horizons,
        target_kinds=args.target_kinds,
        round_trip_cost_bps=float(args.round_trip_cost_bps),
        max_threshold_samples_per_feature=int(args.max_threshold_samples_per_feature),
        max_shards_per_role=int(args.max_shards_per_role),
        min_count_for_stability=int(args.min_count_for_stability),
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(report.get("outputs", {}).get("report_json", ""))
    return report


if __name__ == "__main__":
    main()
