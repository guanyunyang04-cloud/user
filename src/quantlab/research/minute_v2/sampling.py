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

    group_columns = ["trade_date", "bar_time"]
    group_count = int(universe.groupby(group_columns, sort=False).ngroups)
    candidate_count = int(universe["_candidate_selected"].sum())
    candidate_rows_in_base = int(len(candidates))
    group_sizes = universe.groupby(group_columns, sort=False).size()
    recalls: dict[str, float | None] = {}
    hits: dict[str, int] = {}
    eligible_groups: dict[str, int] = {}
    for k in requested_k:
        hit_count = 0
        eligible = 0
        for _, group in universe.groupby(group_columns, sort=False):
            if group.empty:
                continue
            take = min(int(k), len(group))
            ranked = group.sort_values(
                [target_name, "symbol"], ascending=[False, True], kind="stable"
            ).head(take)
            eligible += 1
            hit_count += int(ranked["_candidate_selected"].any())
        key = f"top_{k}"
        hits[key] = hit_count
        eligible_groups[key] = eligible
        recalls[key] = hit_count / eligible if eligible else None

    candidate_group_counts = universe.loc[universe["_candidate_selected"]].groupby(
        group_columns, sort=False
    ).size()
    return {
        "schema": "quantlab.minute_v2_candidate_recall/1",
        "target": target_name,
        "base_rows": int(len(base)),
        "outcome_rows": int(len(outcome_frame)),
        "finite_outcome_rows": int(len(universe)),
        "candidate_rows": candidate_count,
        "candidate_rows_in_base": candidate_rows_in_base,
        "candidate_fraction_of_finite_outcomes": candidate_count / len(universe),
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
        "top_k_hits": hits,
        "top_k_eligible_groups": eligible_groups,
        "top_k_recall": recalls,
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
        candidate_rows = scalar(
            "SELECT count(*) FROM recall_universe "
            "WHERE candidate_selected AND isfinite(target)"
        )
        group_count = scalar(
            "SELECT count(*) FROM (SELECT DISTINCT trade_date,bar_time "
            "FROM recall_universe WHERE isfinite(target))"
        )
        candidate_group_count = scalar(
            "SELECT count(*) FROM (SELECT DISTINCT trade_date,bar_time "
            "FROM recall_universe WHERE isfinite(target) AND candidate_selected)"
        )
        candidate_group_stats = connection.execute(
            "SELECT min(rows), max(rows) FROM ("
            "SELECT trade_date,bar_time,count(*) AS rows FROM recall_universe "
            "WHERE isfinite(target) AND candidate_selected GROUP BY trade_date,bar_time)"
        ).fetchone()
        universe_group_stats = connection.execute(
            "SELECT min(rows), max(rows) FROM ("
            "SELECT trade_date,bar_time,count(*) AS rows FROM recall_universe "
            "WHERE isfinite(target) GROUP BY trade_date,bar_time)"
        ).fetchone()
        report: dict[str, Any] = {
            "schema": "quantlab.minute_v2_candidate_recall/1",
            "target": target_name,
            "base_rows": base_rows,
            "outcome_rows": outcome_rows,
            "finite_outcome_rows": finite_rows,
            "candidate_rows": candidate_rows,
            "candidate_rows_in_base": candidate_rows_in_base,
            "candidate_fraction_of_finite_outcomes": float(candidate_rows / finite_rows),
            "group_count": group_count,
            "groups_without_candidates": int(group_count - candidate_group_count),
            "minimum_candidate_rows_per_group": int(candidate_group_stats[0])
            if candidate_group_stats and candidate_group_stats[0] is not None
            else 0,
            "maximum_candidate_rows_per_group": int(candidate_group_stats[1])
            if candidate_group_stats and candidate_group_stats[1] is not None
            else 0,
            "minimum_universe_rows_per_group": int(universe_group_stats[0])
            if universe_group_stats and universe_group_stats[0] is not None
            else 0,
            "maximum_universe_rows_per_group": int(universe_group_stats[1])
            if universe_group_stats and universe_group_stats[1] is not None
            else 0,
            "top_k_hits": {},
            "top_k_eligible_groups": {},
            "top_k_recall": {},
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
                           bool_or(candidate_selected) FILTER (WHERE rank_in_group <= {int(k)}) AS hit
                    FROM ranked
                    GROUP BY trade_date, bar_time
                )
                SELECT count(*) FILTER (WHERE hit), count(*) FROM groups
                """
            ).fetchone()
            hits, eligible = row or (0, 0)
            hit_count = int(hits or 0)
            eligible_count = int(eligible or 0)
            key = f"top_{k}"
            report["top_k_hits"][key] = hit_count
            report["top_k_eligible_groups"][key] = eligible_count
            report["top_k_recall"][key] = (
                float(hit_count / eligible_count) if eligible_count else None
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
