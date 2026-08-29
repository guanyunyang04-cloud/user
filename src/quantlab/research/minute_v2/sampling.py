"""Causal stock-attention gate evaluated at every decision minute."""

from __future__ import annotations

import re
from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .contracts import KEY_COLUMNS, MinuteV2Config, MinuteV2Error

EVENT_FLAG_COLUMNS = (
    "event_extreme_return",
    "event_volume_shock",
    "event_industry_leadership",
    "event_vwap_cross",
    "event_liquidity_anchor",
    "event_background_control",
)

# Event files are an index into the full causal feature table.  Keeping the
# gate inputs and decisions here makes the event artifact cheap to scan while
# preserving enough provenance to audit why a row was selected.  Model
# features are joined back by KEY_COLUMNS when a training/replay consumer
# needs them.
EVENT_CONTEXT_COLUMNS = (
    "industry_name",
    "adjust_factor",
    "valid_open",
    "valid_high",
    "valid_low",
    "valid_close",
    "daily_liquidity_rank",
    "stock_return_rank",
    "stock_return_5m_rank",
    "stock_volume_acceleration_rank",
    "stock_amount_rank",
    "industry_strength_rank",
    "industry_stock_return_rank",
    "industry_stock_amount_rank",
    "market_breadth_positive",
    "vwap_deviation",
    "previous_vwap_deviation",
    "return_5m",
)

DEFAULT_RECALL_TOP_K = (1, 3, 5)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
RECALL_TARGET_SEMANTICS = (
    "label_return_*m is measured over decision-grid steps; "
    "label_session_return_*m is measured over same-trade-date raw bars"
)

CANDIDATE_COLUMNS = (
    "candidate_attention",
    "candidate_background_control",
    "candidate_selected",
    "candidate_reason_mask",
)


def event_query(
    *,
    feature_view: str = "minute_features",
    config: MinuteV2Config,
) -> str:
    """Keep causal stock candidates while preserving every decision-time group."""

    config.validate()
    background_percent = int(config.candidate_background_percent)
    liquidity = float(config.minimum_daily_liquidity_rank)
    stock_floor = float(config.candidate_stock_rank_floor)
    industry_floor = float(config.candidate_industry_rank_floor)
    industry_stock_floor = float(config.candidate_industry_stock_rank_floor)
    volume_floor = float(config.candidate_volume_rank_floor)
    return f"""
        WITH flags AS (
            SELECT
                *,
                (
                    stock_return_rank >= {stock_floor}
                    OR stock_return_5m_rank >= {stock_floor}
                ) AS event_extreme_return,
                (
                    stock_volume_acceleration_rank >= {volume_floor}
                    AND stock_amount_rank >= 0.50
                    AND ABS(COALESCE(return_from_previous_close, 0.0)) >= 0.001
                ) AS event_volume_shock,
                (
                    industry_strength_rank >= {industry_floor}
                    AND industry_stock_return_rank >= {industry_stock_floor}
                    AND COALESCE(industry_return_mean, 0.0) > 0.0
                ) AS event_industry_leadership,
                (
                    previous_vwap_deviation IS NOT NULL
                    AND vwap_deviation IS NOT NULL
                    AND SIGN(previous_vwap_deviation) <> SIGN(vwap_deviation)
                    AND ABS(COALESCE(return_5m, 0.0)) >= 0.001
                ) AS event_vwap_cross,
                stock_amount_rank >= 0.90 AS event_liquidity_anchor,
                (HASH(symbol, trade_date, bar_time) % 100) < {background_percent}
                    AS event_background_control
            FROM {feature_view}
        ),
        candidates AS (
            SELECT
                *,
                (
                    daily_liquidity_rank >= {liquidity}
                    AND (
                        event_extreme_return
                        OR event_volume_shock
                        OR event_industry_leadership
                        OR event_vwap_cross
                        OR event_liquidity_anchor
                    )
                ) AS candidate_attention,
                (
                    daily_liquidity_rank >= {liquidity}
                    AND event_background_control
                ) AS candidate_background_control
            FROM flags
        ),
        encoded AS (
            SELECT
                *,
                candidate_attention OR candidate_background_control AS candidate_selected,
                CAST(event_extreme_return AS INTEGER)
                    + 2 * CAST(event_volume_shock AS INTEGER)
                    + 4 * CAST(event_industry_leadership AS INTEGER)
                    + 8 * CAST(event_vwap_cross AS INTEGER)
                    + 16 * CAST(event_liquidity_anchor AS INTEGER)
                    + 32 * CAST(event_background_control AS INTEGER)
                    AS candidate_reason_mask
            FROM candidates
        )
        SELECT
            symbol,
            trade_date,
            bar_time,
            {", ".join(EVENT_CONTEXT_COLUMNS)},
            {", ".join(EVENT_FLAG_COLUMNS)},
            candidate_attention,
            candidate_background_control,
            candidate_selected,
            candidate_reason_mask,
            candidate_reason_mask AS event_mask
        FROM encoded
        WHERE candidate_selected
        ORDER BY trade_date, bar_time, symbol
    """


def build_event_frame(
    features: pd.DataFrame,
    *,
    config: MinuteV2Config | None = None,
    connection: Any | None = None,
) -> pd.DataFrame:
    current = config or MinuteV2Config()
    missing = sorted(set(KEY_COLUMNS).difference(features.columns))
    if missing:
        raise MinuteV2Error(f"minute_v2_event_keys_missing:{','.join(missing)}")
    owned = connection is None
    con = duckdb.connect(":memory:") if owned else connection
    try:
        con.register("minute_features", features)
        result = con.execute(event_query(config=current)).fetchdf()
    finally:
        if owned:
            con.close()
    if result.empty:
        raise MinuteV2Error("minute_v2_candidate_gate_empty")
    if result.duplicated(list(KEY_COLUMNS)).any():
        raise MinuteV2Error("minute_v2_duplicate_event_keys")
    source_sizes = features.groupby(["trade_date", "bar_time"], sort=False).size()
    result_sizes = result.groupby(["trade_date", "bar_time"], sort=False).size()
    if len(result_sizes) != len(source_sizes):
        raise MinuteV2Error(
            f"minute_v2_candidate_time_groups_missing:{len(result_sizes)}:{len(source_sizes)}"
        )
    expected_sizes = source_sizes.reindex(result_sizes.index)
    if result_sizes.gt(expected_sizes).any():
        raise MinuteV2Error("minute_v2_candidate_group_exceeds_universe")
    return result


def _validate_recall_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    require_target: str | None = None,
) -> None:
    missing = sorted(set(KEY_COLUMNS).difference(frame.columns))
    if require_target is not None and require_target not in frame.columns:
        missing.append(require_target)
    if missing:
        raise MinuteV2Error(
            f"minute_v2_recall_{name}_columns_missing:{','.join(sorted(set(missing)))}"
        )
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise MinuteV2Error(f"minute_v2_recall_{name}_duplicate_keys")
    if frame.loc[:, list(KEY_COLUMNS)].isna().any().any():
        raise MinuteV2Error(f"minute_v2_recall_{name}_null_keys")


def _normalise_recall_top_k(top_k: Sequence[int]) -> tuple[int, ...]:
    try:
        if isinstance(top_k, (str, bytes)):
            raise TypeError
        values = list(top_k)
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
            raise TypeError
        requested = tuple(sorted({int(value) for value in values if int(value) > 0}))
    except (OverflowError, TypeError, ValueError) as exc:
        raise MinuteV2Error("minute_v2_recall_top_k_invalid") from exc
    if not requested:
        raise MinuteV2Error("minute_v2_recall_top_k_invalid")
    return requested


def _validate_recall_target_name(target: str) -> str:
    if not isinstance(target, str):
        raise MinuteV2Error("minute_v2_recall_target_invalid")
    value = target
    if not _IDENTIFIER_RE.fullmatch(value) or value in KEY_COLUMNS:
        raise MinuteV2Error("minute_v2_recall_target_invalid")
    return value


def _recall_key_type_signature(path: Path, *, name: str) -> tuple[tuple[str, str], ...]:
    """Return the physical key types before DuckDB can apply implicit casts."""

    try:
        schema = pq.ParquetFile(path).schema_arrow
    except Exception as exc:
        raise MinuteV2Error(
            f"minute_v2_recall_{name}_file_unreadable:{path}"
        ) from exc
    missing = sorted(set(KEY_COLUMNS).difference(schema.names))
    if missing:
        raise MinuteV2Error(
            f"minute_v2_recall_{name}_columns_missing:{','.join(missing)}"
        )
    return tuple((key, str(schema.field(key).type)) for key in KEY_COLUMNS)


def _key_difference(left: pd.DataFrame, right: pd.DataFrame) -> int:
    keys = list(KEY_COLUMNS)
    try:
        matched = left.loc[:, keys].merge(
            right.loc[:, keys],
            how="left",
            on=keys,
            indicator=True,
            validate="one_to_one",
        )
    except (TypeError, ValueError) as exc:
        raise MinuteV2Error("minute_v2_recall_key_types_inconsistent") from exc
    return int((matched["_merge"] == "left_only").sum())


def _validate_recall_key_relationships(
    base: pd.DataFrame,
    candidates: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    missing_outcomes = _key_difference(base, outcomes)
    extra_outcomes = _key_difference(outcomes, base)
    if missing_outcomes or extra_outcomes:
        raise MinuteV2Error(
            "minute_v2_recall_outcomes_coverage_invalid:"
            f"{missing_outcomes}:{extra_outcomes}"
        )
    extra_candidates = _key_difference(candidates, base)
    if extra_candidates:
        raise MinuteV2Error(f"minute_v2_recall_candidates_outside_base:{extra_candidates}")


def _ratio(numerator: float | int, denominator: float | int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _lift(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline <= 0:
        return None
    return float(value / baseline)


def _random_group_hit_probability(
    universe_rows: int,
    candidate_rows: int,
    take: int,
) -> float:
    """Exact same-density null probability of at least one hit.

    The null draws ``candidate_rows`` rows uniformly without replacement from
    a group of ``universe_rows``.  This conditions the benchmark on the gate's
    actual density in every decision-minute group instead of comparing it with
    a looser global Bernoulli approximation.
    """

    universe = int(universe_rows)
    candidates = int(candidate_rows)
    selected = min(max(int(take), 0), universe)
    if universe <= 0 or selected <= 0 or candidates <= 0:
        return 0.0
    if candidates >= universe or selected > universe - candidates:
        return 1.0
    miss = 1.0
    for offset in range(selected):
        miss *= (universe - candidates - offset) / (universe - offset)
    return float(1.0 - miss)


def _recall_metrics_from_universe(
    universe: pd.DataFrame,
    *,
    target_name: str,
    requested_k: tuple[int, ...],
) -> dict[str, Any]:
    """Calculate recall metrics from a finite, key-complete universe."""

    group_columns = ["trade_date", "bar_time"]
    groups = list(universe.groupby(group_columns, sort=False))
    group_count = len(groups)
    candidate_count = int(universe["_candidate_selected"].sum())
    group_sizes = universe.groupby(group_columns, sort=False).size()
    candidate_group_counts = universe.loc[universe["_candidate_selected"]].groupby(
        group_columns, sort=False
    ).size()

    target_values = universe[target_name].astype(float)
    candidate_mask = universe["_candidate_selected"].astype(bool)
    positive_mask = target_values > 0.0
    positive_utility = target_values.clip(lower=0.0)
    negative_mask = target_values < 0.0
    negative_utility = (-target_values).clip(lower=0.0)
    positive_rows = int(positive_mask.sum())
    candidate_positive_rows = int((candidate_mask & positive_mask).sum())
    negative_rows = int(negative_mask.sum())
    candidate_negative_rows = int((candidate_mask & negative_mask).sum())
    total_positive_utility = float(positive_utility.sum())
    candidate_positive_utility = float(positive_utility.loc[candidate_mask].sum())
    total_negative_utility = float(negative_utility.sum())
    candidate_negative_utility = float(negative_utility.loc[candidate_mask].sum())

    random_positive_rows = 0.0
    random_positive_utility = 0.0
    random_negative_rows = 0.0
    random_negative_utility = 0.0
    for _, group in groups:
        universe_rows = len(group)
        candidate_rows = int(group["_candidate_selected"].sum())
        density = candidate_rows / universe_rows
        group_target = group[target_name].astype(float)
        random_positive_rows += int((group_target > 0.0).sum()) * density
        random_positive_utility += float(group_target.clip(lower=0.0).sum()) * density
        random_negative_rows += int((group_target < 0.0).sum()) * density
        random_negative_utility += float((-group_target).clip(lower=0.0).sum()) * density

    positive_recall = _ratio(candidate_positive_rows, positive_rows)
    random_positive_recall = _ratio(random_positive_rows, positive_rows)
    utility_capture = _ratio(candidate_positive_utility, total_positive_utility)
    random_utility_capture = _ratio(random_positive_utility, total_positive_utility)
    negative_recall = _ratio(candidate_negative_rows, negative_rows)
    random_negative_recall = _ratio(random_negative_rows, negative_rows)
    negative_utility_capture = _ratio(candidate_negative_utility, total_negative_utility)
    random_negative_utility_capture = _ratio(
        random_negative_utility, total_negative_utility
    )

    top_k_group_hits: dict[str, int] = {}
    top_k_eligible_groups: dict[str, int] = {}
    top_k_group_hit_rate: dict[str, float | None] = {}
    top_k_row_hits: dict[str, int] = {}
    top_k_rows: dict[str, int] = {}
    top_k_row_recall: dict[str, float | None] = {}
    top_k_random_group_hit_rate: dict[str, float | None] = {}
    top_k_random_row_recall: dict[str, float | None] = {}
    top_k_group_hit_lift_vs_random: dict[str, float | None] = {}
    top_k_row_recall_lift_vs_random: dict[str, float | None] = {}
    for k in requested_k:
        hit_count = 0
        row_hit_count = 0
        eligible = 0
        ranked_rows = 0
        expected_group_hits = 0.0
        expected_row_hits = 0.0
        for _, group in groups:
            universe_rows = len(group)
            if universe_rows == 0:
                continue
            take = min(int(k), universe_rows)
            ranked = group.sort_values(
                [target_name, "symbol"], ascending=[False, True], kind="stable"
            ).head(take)
            candidate_rows = int(group["_candidate_selected"].sum())
            eligible += 1
            ranked_rows += take
            selected_in_top = int(ranked["_candidate_selected"].sum())
            row_hit_count += selected_in_top
            hit_count += int(selected_in_top > 0)
            expected_group_hits += _random_group_hit_probability(
                universe_rows,
                candidate_rows,
                take,
            )
            expected_row_hits += take * candidate_rows / universe_rows
        key = f"top_{k}"
        group_rate = _ratio(hit_count, eligible)
        row_recall = _ratio(row_hit_count, ranked_rows)
        random_group_rate = _ratio(expected_group_hits, eligible)
        random_row_recall = _ratio(expected_row_hits, ranked_rows)
        top_k_group_hits[key] = hit_count
        top_k_eligible_groups[key] = eligible
        top_k_group_hit_rate[key] = group_rate
        top_k_row_hits[key] = row_hit_count
        top_k_rows[key] = ranked_rows
        top_k_row_recall[key] = row_recall
        top_k_random_group_hit_rate[key] = random_group_rate
        top_k_random_row_recall[key] = random_row_recall
        top_k_group_hit_lift_vs_random[key] = _lift(group_rate, random_group_rate)
        top_k_row_recall_lift_vs_random[key] = _lift(row_recall, random_row_recall)

    candidate_values = target_values.loc[candidate_mask]
    non_candidate_values = target_values.loc[~candidate_mask]
    candidate_positive_precision = _ratio(candidate_positive_rows, candidate_count)
    overall_positive_rate = _ratio(positive_rows, len(universe))
    return {
        "finite_outcome_rows": int(len(universe)),
        "candidate_rows": candidate_count,
        "candidate_fraction_of_finite_outcomes": float(candidate_count / len(universe)),
        "group_count": group_count,
        "groups_without_candidates": int(group_count - len(candidate_group_counts)),
        "minimum_candidate_rows_per_group": int(candidate_group_counts.min())
        if len(candidate_group_counts)
        else 0,
        "maximum_candidate_rows_per_group": int(candidate_group_counts.max())
        if len(candidate_group_counts)
        else 0,
        "minimum_universe_rows_per_group": int(group_sizes.min()),
        "maximum_universe_rows_per_group": int(group_sizes.max()),
        "target_mean": float(target_values.mean()),
        "candidate_target_mean": float(candidate_values.mean())
        if len(candidate_values)
        else None,
        "non_candidate_target_mean": float(non_candidate_values.mean())
        if len(non_candidate_values)
        else None,
        "candidate_target_mean_difference": (
            float(candidate_values.mean() - non_candidate_values.mean())
            if len(candidate_values) and len(non_candidate_values)
            else None
        ),
        "positive_outcome_threshold": 0.0,
        "positive_outcome_rows": positive_rows,
        "candidate_positive_outcome_rows": candidate_positive_rows,
        "positive_outcome_row_recall": positive_recall,
        "random_positive_outcome_row_recall": random_positive_recall,
        "positive_outcome_row_recall_lift_vs_random": _lift(
            positive_recall, random_positive_recall
        ),
        "candidate_positive_outcome_precision": candidate_positive_precision,
        "overall_positive_outcome_rate": overall_positive_rate,
        "candidate_positive_outcome_precision_lift_vs_overall": _lift(
            candidate_positive_precision, overall_positive_rate
        ),
        "positive_utility_total": total_positive_utility,
        "candidate_positive_utility": candidate_positive_utility,
        "positive_utility_capture": utility_capture,
        "random_positive_utility_capture": random_utility_capture,
        "positive_utility_capture_lift_vs_random": _lift(
            utility_capture, random_utility_capture
        ),
        "negative_outcome_rows": negative_rows,
        "candidate_negative_outcome_rows": candidate_negative_rows,
        "negative_outcome_row_recall": negative_recall,
        "random_negative_outcome_row_recall": random_negative_recall,
        "negative_outcome_row_recall_lift_vs_random": _lift(
            negative_recall, random_negative_recall
        ),
        "negative_utility_total": total_negative_utility,
        "candidate_negative_utility": candidate_negative_utility,
        "negative_utility_capture": negative_utility_capture,
        "random_negative_utility_capture": random_negative_utility_capture,
        "negative_utility_capture_lift_vs_random": _lift(
            negative_utility_capture, random_negative_utility_capture
        ),
        "top_k_group_hits": top_k_group_hits,
        "top_k_eligible_groups": top_k_eligible_groups,
        "top_k_group_hit_rate": top_k_group_hit_rate,
        "top_k_row_hits": top_k_row_hits,
        "top_k_rows": top_k_rows,
        "top_k_row_recall": top_k_row_recall,
        "top_k_random_group_hit_rate": top_k_random_group_hit_rate,
        "top_k_random_row_recall": top_k_random_row_recall,
        "top_k_group_hit_lift_vs_random": top_k_group_hit_lift_vs_random,
        "top_k_row_recall_lift_vs_random": top_k_row_recall_lift_vs_random,
        # Backward-compatible aliases.  In schema /1 these were called
        # "recall", although they are group hit rates rather than row recall.
        "top_k_hits": dict(top_k_group_hits),
        "top_k_recall": dict(top_k_group_hit_rate),
        "legacy_top_k_recall_definition": (
            "fraction of decision-minute groups with at least one candidate "
            "among the top-k rows; use top_k_row_recall for row recall"
        ),
        "random_baseline_definition": (
            "exact within-group uniform sampling without replacement at each "
            "decision-minute group's observed candidate density"
        ),
    }


def audit_candidate_recall(
    base: pd.DataFrame,
    candidates: pd.DataFrame,
    outcomes: pd.DataFrame | None = None,
    *,
    target: str = "label_return_5m",
    top_k: Sequence[int] = DEFAULT_RECALL_TOP_K,
) -> dict[str, Any]:
    """Measure whether the causal gate retains the eventual best rows.

    ``base`` and ``outcomes`` represent the complete universe.  ``candidates``
    is treated only as a set of primary keys; no outcome column is consulted
    when determining membership.  The function is therefore an audit of the
    gate, not a way to tune it using future returns.
    """

    target_name = _validate_recall_target_name(target)
    _validate_recall_frame(base, name="base")
    _validate_recall_frame(candidates, name="candidates")
    outcome_frame = base if outcomes is None and target_name in base.columns else outcomes
    if outcome_frame is None:
        raise MinuteV2Error("minute_v2_recall_outcomes_required")
    _validate_recall_frame(outcome_frame, name="outcomes", require_target=target_name)
    requested_k = _normalise_recall_top_k(top_k)
    _validate_recall_key_relationships(base, candidates, outcome_frame)

    keys = list(KEY_COLUMNS)
    universe = base.loc[:, keys].merge(
        outcome_frame.loc[:, [*keys, target_name]],
        how="left",
        on=keys,
        validate="one_to_one",
    )
    candidate_keys = candidates.loc[:, keys].drop_duplicates().copy()
    candidate_keys["_candidate_selected"] = True
    universe = universe.merge(candidate_keys, how="left", on=keys, validate="one_to_one")
    universe["_candidate_selected"] = (
        universe["_candidate_selected"].astype("boolean").fillna(False).astype(bool)
    )
    universe[target_name] = pd.to_numeric(universe[target_name], errors="coerce")
    finite = np.isfinite(
        pd.to_numeric(universe[target_name], errors="coerce").to_numpy(dtype=float)
    )
    universe = universe.loc[finite].copy()
    if universe.empty:
        raise MinuteV2Error("minute_v2_recall_no_finite_outcomes")

    candidate_rows_in_base = int(len(candidates))
    metrics = _recall_metrics_from_universe(
        universe,
        target_name=target_name,
        requested_k=requested_k,
    )
    return {
        "schema": "quantlab.minute_v2_candidate_recall/2",
        "target": target_name,
        "base_rows": int(len(base)),
        "outcome_rows": int(len(outcome_frame)),
        "candidate_rows_in_base": candidate_rows_in_base,
        "candidate_fraction_of_base": candidate_rows_in_base / len(base) if len(base) else None,
        "finite_outcome_fraction_of_base": len(universe) / len(base) if len(base) else None,
        **metrics,
        "candidate_membership_source": "primary-key-only; target excluded from gate membership",
        "target_semantics": RECALL_TARGET_SEMANTICS,
    }


def audit_candidate_recall_files(
    base_path: str | Path,
    candidate_path: str | Path,
    outcome_path: str | Path,
    *,
    target: str = "label_return_5m",
    top_k: Sequence[int] = DEFAULT_RECALL_TOP_K,
) -> dict[str, Any]:
    """Run the recall audit from Parquet without materializing feature columns."""

    target_name = _validate_recall_target_name(target)
    requested_k = _normalise_recall_top_k(top_k)
    paths = {
        "base": Path(base_path).resolve(),
        "candidates": Path(candidate_path).resolve(),
        "outcomes": Path(outcome_path).resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise MinuteV2Error(f"minute_v2_recall_{name}_file_missing:{path}")
    key_types_by_file = {
        name: _recall_key_type_signature(path, name=name)
        for name, path in paths.items()
    }
    distinct_key_types = {signature for signature in key_types_by_file.values()}
    if len(distinct_key_types) != 1:
        details = ";".join(
            f"{name}=" + ",".join(f"{key}:{kind}" for key, kind in signature)
            for name, signature in key_types_by_file.items()
        )
        raise MinuteV2Error(f"minute_v2_recall_key_types_inconsistent:{details}")

    connection = duckdb.connect(":memory:")
    try:
        def literal(path: str | Path) -> str:
            return "'" + str(Path(path).resolve()).replace("'", "''") + "'"

        scans = {
            name: f"read_parquet({literal(path)}, union_by_name=true)"
            for name, path in paths.items()
        }
        views = {
            name: f"recall_{name}"
            for name in paths
        }
        for name, view in views.items():
            try:
                connection.execute(
                    f"CREATE OR REPLACE TEMP VIEW {view} AS SELECT * FROM {scans[name]}"
                )
            except duckdb.Error as exc:
                raise MinuteV2Error(
                    f"minute_v2_recall_{name}_file_unreadable:{paths[name]}"
                ) from exc

        key_projection = ",".join(KEY_COLUMNS)
        key_predicate = " OR ".join(f"{key} IS NULL" for key in KEY_COLUMNS)

        def scalar(query: str) -> int:
            value = connection.execute(query).fetchone()
            return int(value[0]) if value and value[0] is not None else 0

        for name, view in views.items():
            try:
                connection.execute(f"SELECT {key_projection} FROM {view} LIMIT 0")
            except duckdb.Error as exc:
                raise MinuteV2Error(
                    f"minute_v2_recall_{name}_columns_missing:{','.join(KEY_COLUMNS)}"
                ) from exc
            null_keys = scalar(f"SELECT count(*) FROM {view} WHERE {key_predicate}")
            if null_keys:
                raise MinuteV2Error(f"minute_v2_recall_{name}_null_keys")
            duplicate_keys = scalar(
                f"SELECT count(*) FROM (SELECT {key_projection} FROM {view} "
                f"GROUP BY {key_projection} HAVING count(*) > 1)"
            )
            if duplicate_keys:
                raise MinuteV2Error(f"minute_v2_recall_{name}_duplicate_keys")

        try:
            connection.execute(
                f"SELECT \"{target_name.replace(chr(34), chr(34) * 2)}\" "
                f"FROM {views['outcomes']} LIMIT 0"
            )
        except duckdb.Error as exc:
            raise MinuteV2Error(
                f"minute_v2_recall_outcomes_columns_missing:{target_name}"
            ) from exc

        missing_outcomes = scalar(
            f"SELECT count(*) FROM {views['base']} b LEFT JOIN {views['outcomes']} o "
            f"USING({key_projection}) WHERE o.symbol IS NULL"
        )
        extra_outcomes = scalar(
            f"SELECT count(*) FROM {views['outcomes']} o LEFT JOIN {views['base']} b "
            f"USING({key_projection}) WHERE b.symbol IS NULL"
        )
        if missing_outcomes or extra_outcomes:
            raise MinuteV2Error(
                "minute_v2_recall_outcomes_coverage_invalid:"
                f"{missing_outcomes}:{extra_outcomes}"
            )
        extra_candidates = scalar(
            f"SELECT count(*) FROM {views['candidates']} c LEFT JOIN {views['base']} b "
            f"USING({key_projection}) WHERE b.symbol IS NULL"
        )
        if extra_candidates:
            raise MinuteV2Error(
                f"minute_v2_recall_candidates_outside_base:{extra_candidates}"
            )

        escaped_target = target_name.replace('"', '""')
        connection.execute(
            f"CREATE OR REPLACE TEMP VIEW recall_outcome_values AS "
            f"SELECT {key_projection}, TRY_CAST(\"{escaped_target}\" AS DOUBLE) AS target "
            f"FROM {views['outcomes']}"
        )
        connection.execute(
            f"CREATE OR REPLACE TEMP VIEW recall_universe AS "
            f"SELECT b.{KEY_COLUMNS[0]}, b.{KEY_COLUMNS[1]}, b.{KEY_COLUMNS[2]}, "
            "o.target, c.symbol IS NOT NULL AS candidate_selected "
            f"FROM {views['base']} b JOIN recall_outcome_values o USING({key_projection}) "
            f"LEFT JOIN {views['candidates']} c USING({key_projection})"
        )
        finite_rows = scalar(
            "SELECT count(*) FROM recall_universe WHERE isfinite(target)"
        )
        if finite_rows == 0:
            raise MinuteV2Error("minute_v2_recall_no_finite_outcomes")
        base_rows = scalar(f"SELECT count(*) FROM {views['base']}")
        outcome_rows = scalar(f"SELECT count(*) FROM {views['outcomes']}")
        candidate_rows_in_base = scalar(f"SELECT count(*) FROM {views['candidates']}")
        aggregate = connection.execute(
            """
            SELECT
                count(*) AS finite_rows,
                count(*) FILTER (WHERE candidate_selected) AS candidate_rows,
                avg(target) AS target_mean,
                avg(target) FILTER (WHERE candidate_selected) AS candidate_target_mean,
                avg(target) FILTER (WHERE NOT candidate_selected) AS non_candidate_target_mean,
                count(*) FILTER (WHERE target > 0.0) AS positive_rows,
                count(*) FILTER (WHERE candidate_selected AND target > 0.0)
                    AS candidate_positive_rows,
                sum(greatest(target, 0.0)) AS positive_utility,
                sum(greatest(target, 0.0)) FILTER (WHERE candidate_selected)
                    AS candidate_positive_utility,
                count(*) FILTER (WHERE target < 0.0) AS negative_rows,
                count(*) FILTER (WHERE candidate_selected AND target < 0.0)
                    AS candidate_negative_rows,
                sum(greatest(-target, 0.0)) AS negative_utility,
                sum(greatest(-target, 0.0)) FILTER (WHERE candidate_selected)
                    AS candidate_negative_utility
            FROM recall_universe
            WHERE isfinite(target)
            """
        ).fetchone()
        if not aggregate:
            raise MinuteV2Error("minute_v2_recall_no_finite_outcomes")
        candidate_rows = int(aggregate[1] or 0)
        target_mean = float(aggregate[2])
        candidate_target_mean = (
            float(aggregate[3]) if aggregate[3] is not None else None
        )
        non_candidate_target_mean = (
            float(aggregate[4]) if aggregate[4] is not None else None
        )
        positive_rows = int(aggregate[5] or 0)
        candidate_positive_rows = int(aggregate[6] or 0)
        positive_utility = float(aggregate[7] or 0.0)
        candidate_positive_utility = float(aggregate[8] or 0.0)
        negative_rows = int(aggregate[9] or 0)
        candidate_negative_rows = int(aggregate[10] or 0)
        negative_utility = float(aggregate[11] or 0.0)
        candidate_negative_utility = float(aggregate[12] or 0.0)
        group_rows = connection.execute(
            """
            SELECT
                count(*) AS universe_rows,
                count(*) FILTER (WHERE candidate_selected) AS candidate_rows,
                count(*) FILTER (WHERE target > 0.0) AS positive_rows,
                sum(greatest(target, 0.0)) AS positive_utility,
                count(*) FILTER (WHERE target < 0.0) AS negative_rows,
                sum(greatest(-target, 0.0)) AS negative_utility
            FROM recall_universe
            WHERE isfinite(target)
            GROUP BY trade_date, bar_time
            """
        ).fetchall()
        group_count = len(group_rows)
        candidate_group_values = [int(row[1]) for row in group_rows if int(row[1]) > 0]
        universe_group_values = [int(row[0]) for row in group_rows]
        random_positive_rows = sum(
            int(row[2]) * int(row[1]) / int(row[0]) for row in group_rows
        )
        random_positive_utility = sum(
            float(row[3] or 0.0) * int(row[1]) / int(row[0]) for row in group_rows
        )
        random_negative_rows = sum(
            int(row[4]) * int(row[1]) / int(row[0]) for row in group_rows
        )
        random_negative_utility = sum(
            float(row[5] or 0.0) * int(row[1]) / int(row[0]) for row in group_rows
        )
        positive_recall = _ratio(candidate_positive_rows, positive_rows)
        random_positive_recall = _ratio(random_positive_rows, positive_rows)
        utility_capture = _ratio(candidate_positive_utility, positive_utility)
        random_utility_capture = _ratio(random_positive_utility, positive_utility)
        negative_recall = _ratio(candidate_negative_rows, negative_rows)
        random_negative_recall = _ratio(random_negative_rows, negative_rows)
        negative_utility_capture = _ratio(candidate_negative_utility, negative_utility)
        random_negative_utility_capture = _ratio(
            random_negative_utility, negative_utility
        )
        candidate_positive_precision = _ratio(candidate_positive_rows, candidate_rows)
        overall_positive_rate = _ratio(positive_rows, finite_rows)
        report: dict[str, Any] = {
            "schema": "quantlab.minute_v2_candidate_recall/2",
            "target": target_name,
            "base_rows": base_rows,
            "outcome_rows": outcome_rows,
            "finite_outcome_rows": finite_rows,
            "candidate_rows": candidate_rows,
            "candidate_rows_in_base": candidate_rows_in_base,
            "candidate_fraction_of_base": float(candidate_rows_in_base / base_rows)
            if base_rows
            else None,
            "finite_outcome_fraction_of_base": float(finite_rows / base_rows)
            if base_rows
            else None,
            "candidate_fraction_of_finite_outcomes": float(candidate_rows / finite_rows),
            "group_count": group_count,
            "groups_without_candidates": int(group_count - len(candidate_group_values)),
            "minimum_candidate_rows_per_group": min(candidate_group_values)
            if candidate_group_values
            else 0,
            "maximum_candidate_rows_per_group": max(candidate_group_values)
            if candidate_group_values
            else 0,
            "minimum_universe_rows_per_group": min(universe_group_values),
            "maximum_universe_rows_per_group": max(universe_group_values),
            "target_mean": target_mean,
            "candidate_target_mean": candidate_target_mean,
            "non_candidate_target_mean": non_candidate_target_mean,
            "candidate_target_mean_difference": (
                candidate_target_mean - non_candidate_target_mean
                if candidate_target_mean is not None
                and non_candidate_target_mean is not None
                else None
            ),
            "positive_outcome_threshold": 0.0,
            "positive_outcome_rows": positive_rows,
            "candidate_positive_outcome_rows": candidate_positive_rows,
            "positive_outcome_row_recall": positive_recall,
            "random_positive_outcome_row_recall": random_positive_recall,
            "positive_outcome_row_recall_lift_vs_random": _lift(
                positive_recall, random_positive_recall
            ),
            "candidate_positive_outcome_precision": candidate_positive_precision,
            "overall_positive_outcome_rate": overall_positive_rate,
            "candidate_positive_outcome_precision_lift_vs_overall": _lift(
                candidate_positive_precision, overall_positive_rate
            ),
            "positive_utility_total": positive_utility,
            "candidate_positive_utility": candidate_positive_utility,
            "positive_utility_capture": utility_capture,
            "random_positive_utility_capture": random_utility_capture,
            "positive_utility_capture_lift_vs_random": _lift(
                utility_capture, random_utility_capture
            ),
            "negative_outcome_rows": negative_rows,
            "candidate_negative_outcome_rows": candidate_negative_rows,
            "negative_outcome_row_recall": negative_recall,
            "random_negative_outcome_row_recall": random_negative_recall,
            "negative_outcome_row_recall_lift_vs_random": _lift(
                negative_recall, random_negative_recall
            ),
            "negative_utility_total": negative_utility,
            "candidate_negative_utility": candidate_negative_utility,
            "negative_utility_capture": negative_utility_capture,
            "random_negative_utility_capture": random_negative_utility_capture,
            "negative_utility_capture_lift_vs_random": _lift(
                negative_utility_capture, random_negative_utility_capture
            ),
            "top_k_group_hits": {},
            "top_k_hits": {},
            "top_k_eligible_groups": {},
            "top_k_group_hit_rate": {},
            "top_k_recall": {},
            "top_k_row_hits": {},
            "top_k_rows": {},
            "top_k_row_recall": {},
            "top_k_random_group_hit_rate": {},
            "top_k_random_row_recall": {},
            "top_k_group_hit_lift_vs_random": {},
            "top_k_row_recall_lift_vs_random": {},
            "legacy_top_k_recall_definition": (
                "fraction of decision-minute groups with at least one candidate "
                "among the top-k rows; use top_k_row_recall for row recall"
            ),
            "random_baseline_definition": (
                "exact within-group uniform sampling without replacement at each "
                "decision-minute group's observed candidate density"
            ),
            "candidate_membership_source": "primary-key-only; target excluded from gate membership",
            "target_semantics": RECALL_TARGET_SEMANTICS,
            "key_types": {key: kind for key, kind in key_types_by_file["base"]},
        }
        for k in requested_k:
            row = connection.execute(
                f"""
                WITH ranked AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY trade_date, bar_time
                        ORDER BY target DESC, symbol
                    ) AS rank_in_group
                    FROM recall_universe
                    WHERE isfinite(target)
                ), groups AS (
                    SELECT trade_date, bar_time,
                           bool_or(candidate_selected) FILTER (WHERE rank_in_group <= {int(k)}) AS hit,
                           count(*) FILTER (WHERE rank_in_group <= {int(k)}) AS top_rows,
                           count(*) FILTER (
                               WHERE rank_in_group <= {int(k)} AND candidate_selected
                           ) AS selected_top_rows
                    FROM ranked
                    GROUP BY trade_date, bar_time
                )
                SELECT count(*) FILTER (WHERE hit), count(*),
                       sum(top_rows), sum(selected_top_rows)
                FROM groups
                """
            ).fetchone()
            hits, eligible, ranked_rows, selected_ranked_rows = row or (0, 0, 0, 0)
            hit_count = int(hits or 0)
            eligible_count = int(eligible or 0)
            ranked_row_count = int(ranked_rows or 0)
            selected_ranked_row_count = int(selected_ranked_rows or 0)
            key = f"top_{k}"
            group_rate = _ratio(hit_count, eligible_count)
            row_recall = _ratio(selected_ranked_row_count, ranked_row_count)
            expected_group_hits = sum(
                _random_group_hit_probability(
                    int(group_row[0]),
                    int(group_row[1]),
                    min(int(k), int(group_row[0])),
                )
                for group_row in group_rows
            )
            expected_row_hits = sum(
                min(int(k), int(group_row[0]))
                * int(group_row[1])
                / int(group_row[0])
                for group_row in group_rows
            )
            random_group_rate = _ratio(expected_group_hits, eligible_count)
            random_row_recall = _ratio(expected_row_hits, ranked_row_count)
            report["top_k_group_hits"][key] = hit_count
            report["top_k_hits"][key] = hit_count
            report["top_k_eligible_groups"][key] = eligible_count
            report["top_k_group_hit_rate"][key] = group_rate
            report["top_k_recall"][key] = group_rate
            report["top_k_row_hits"][key] = selected_ranked_row_count
            report["top_k_rows"][key] = ranked_row_count
            report["top_k_row_recall"][key] = row_recall
            report["top_k_random_group_hit_rate"][key] = random_group_rate
            report["top_k_random_row_recall"][key] = random_row_recall
            report["top_k_group_hit_lift_vs_random"][key] = _lift(
                group_rate, random_group_rate
            )
            report["top_k_row_recall_lift_vs_random"][key] = _lift(
                row_recall, random_row_recall
            )
        return report
    finally:
        connection.close()


__all__ = [
    "CANDIDATE_COLUMNS",
    "DEFAULT_RECALL_TOP_K",
    "EVENT_CONTEXT_COLUMNS",
    "EVENT_FLAG_COLUMNS",
    "audit_candidate_recall",
    "audit_candidate_recall_files",
    "build_event_frame",
    "event_query",
]
