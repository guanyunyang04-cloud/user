"""Cross-path validation and visual audit for the strict causal Chan parser."""

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

from daily_research.path_policy import seq100_strict_chan_intraday as intraday
from daily_research.path_policy import seq100_strict_chan_parser as parser
from daily_research.path_policy import seq100_strict_chan_validate as validator

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_PATH = (
    WORKSPACE_ROOT / "daily_research" / "studies" / "seq100_strict_chan_audit_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research"
    / "output"
    / "path_policy"
    / "studies"
    / "seq100_strict_chan_audit_v1"
)
AUDIT_SCHEMA_VERSION = "seq100_strict_chan_audit/1"
SUPPORTED_ASSERTION_OPERATORS = {"<", "<=", "==", ">=", ">"}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_audit_spec(path: str | Path = DEFAULT_AUDIT_PATH) -> dict[str, Any]:
    audit_path = Path(path).resolve()
    spec = json.loads(audit_path.read_text(encoding="utf-8"))
    if spec.get("study_id") != "seq100_strict_chan_audit_v1":
        raise ValueError("strict_chan_audit_study_id_mismatch")
    if spec.get("parent_study_id") != parser.STUDY_ID:
        raise ValueError("strict_chan_audit_parent_study_mismatch")
    boundaries = dict(spec.get("boundaries", {}))
    prohibited = (
        "future_path_test_performed",
        "return_test_performed",
        "profit_claim_allowed",
        "account_replay_allowed",
        "model_training_allowed",
    )
    if any(bool(boundaries.get(name)) for name in prohibited):
        raise ValueError("strict_chan_audit_boundary_violation")

    definition = parser.load_definition_spec()
    variant_contract = dict(definition["variants"])
    profiles = dict(spec.get("profiles", {}))
    if not profiles or profiles.get("primary") != {}:
        raise ValueError("strict_chan_audit_primary_profile_invalid")
    for profile_name, overrides in profiles.items():
        if not profile_name or not isinstance(overrides, dict):
            raise ValueError("strict_chan_audit_profile_invalid")
        for variant_name, value in overrides.items():
            if variant_name not in variant_contract:
                raise ValueError(f"strict_chan_audit_unknown_variant:{variant_name}")
            options = set(variant_contract[variant_name]["options"])
            if value not in options:
                raise ValueError(
                    f"strict_chan_audit_variant_value_invalid:{variant_name}:{value}"
                )

    cases = list(spec.get("cases", []))
    case_ids = [str(case.get("case_id", "")) for case in cases]
    if (
        not cases
        or len(case_ids) != len(set(case_ids))
        or any(not item for item in case_ids)
    ):
        raise ValueError("strict_chan_audit_case_ids_invalid")
    for case in cases:
        required = (
            "case_id",
            "category",
            "symbol",
            "start_date",
            "end_date",
            "display_start_date",
            "display_end_date",
            "focal_date",
            "quality_pool_expected",
            "selection_basis",
        )
        if any(name not in case for name in required):
            raise ValueError(
                f"strict_chan_audit_case_missing_field:{case.get('case_id', '')}"
            )
        dates = [
            str(case["start_date"]),
            str(case["display_start_date"]),
            str(case["focal_date"]),
            str(case["display_end_date"]),
            str(case["end_date"]),
        ]
        if dates != sorted(dates):
            raise ValueError(
                f"strict_chan_audit_case_date_order_invalid:{case['case_id']}"
            )
        if dates[0] < "2010-01-01" or dates[-1] > "2025-12-31":
            raise ValueError(
                f"strict_chan_audit_case_date_contract_invalid:{case['case_id']}"
            )
        for assertion in case.get("assertions", []):
            if assertion.get("operator") not in SUPPORTED_ASSERTION_OPERATORS:
                raise ValueError(
                    f"strict_chan_audit_assertion_operator_invalid:{case['case_id']}"
                )
            if not str(assertion.get("metric", "")):
                raise ValueError(
                    f"strict_chan_audit_assertion_metric_invalid:{case['case_id']}"
                )

    pool_manifest = WORKSPACE_ROOT / str(spec["quality_pool_manifest"])
    if not pool_manifest.is_file():
        raise ValueError("strict_chan_audit_quality_pool_manifest_missing")
    spec["_audit_path"] = str(audit_path)
    spec["_quality_pool_manifest_path"] = str(pool_manifest.resolve())
    return spec


def _case_fingerprint(
    case: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    definition_path: str | Path,
) -> str:
    source_paths = (
        Path(__file__).resolve(),
        Path(parser.__file__).resolve(),
        Path(intraday.__file__).resolve(),
        Path(validator.__file__).resolve(),
    )
    payload = {
        "schema": AUDIT_SCHEMA_VERSION,
        "case": dict(case),
        "profiles": dict(spec["profiles"]),
        "prefix_validation": dict(spec["prefix_validation"]),
        "visualization": dict(spec["visualization"]),
        "definition_sha256": _sha256_file(Path(definition_path).resolve()),
        "source_sha256": {path.name: _sha256_file(path) for path in source_paths},
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _quality_pool_membership(
    case: Mapping[str, Any], manifest_path: str | Path
) -> bool:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    focal_year = int(str(case["focal_date"])[:4])
    matches = [
        item for item in manifest["partitions"] if int(item["year"]) == focal_year
    ]
    if len(matches) != 1:
        raise ValueError(f"strict_chan_audit_pool_partition_invalid:{case['case_id']}")
    frame = pd.read_parquet(
        matches[0]["path"],
        columns=["symbol", "trade_date"],
        filters=[
            ("symbol", "==", str(case["symbol"])),
            ("trade_date", "==", str(case["focal_date"])),
        ],
    )
    return not frame.empty


def _display_frame(frame: pd.DataFrame, case: Mapping[str, Any]) -> pd.DataFrame:
    selected = frame[
        frame["trade_date"].between(
            str(case["display_start_date"]), str(case["display_end_date"])
        )
    ].copy()
    if selected.empty:
        raise ValueError(f"strict_chan_audit_display_empty:{case['case_id']}")
    return selected.sort_values(["episode_id", "timestamp"]).reset_index(drop=True)


def _daily_evidence(frame: pd.DataFrame, case: Mapping[str, Any]) -> dict[str, Any]:
    selected = _display_frame(frame, case)
    daily = selected.groupby("trade_date", sort=True).agg(
        adjusted_open=("open", "first"),
        adjusted_high=("high", "max"),
        adjusted_low=("low", "min"),
        adjusted_close=("close", "last"),
        raw_open=("raw_open", "first"),
        raw_high=("raw_high", "max"),
        raw_low=("raw_low", "min"),
        raw_close=("raw_close", "last"),
        adjust_factor=("adjust_factor", "last"),
    )
    log_change = np.log(daily["adjusted_close"]).diff()
    absolute_path = float(log_change.abs().sum())
    log_window = float(
        np.log(daily["adjusted_close"].iloc[-1] / daily["adjusted_close"].iloc[0])
    )
    dates = pd.Series(pd.to_datetime(daily.index), dtype="datetime64[ns]")
    factor_change = daily["adjust_factor"].pct_change().abs()
    adjusted_open_gap = (
        daily["adjusted_open"] / daily["adjusted_close"].shift(1) - 1.0
    ).abs()
    daily_return = daily["adjusted_close"].pct_change()
    one_price = np.isclose(
        daily["raw_high"].to_numpy(dtype=np.float64),
        daily["raw_low"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1.0e-10,
    )
    window_return = float(
        daily["adjusted_close"].iloc[-1] / daily["adjusted_close"].iloc[0] - 1.0
    )
    return {
        "display_usable_days": len(daily),
        "display_start_date_observed": str(daily.index.min()),
        "display_end_date_observed": str(daily.index.max()),
        "display_window_return": window_return,
        "display_abs_window_return": abs(window_return),
        "display_absolute_log_path": absolute_path,
        "display_path_efficiency": (
            abs(log_window) / absolute_path if absolute_path > 0 else math.nan
        ),
        "display_max_calendar_gap_days": int(dates.diff().dt.days.max() or 0),
        "display_max_abs_factor_change": float(factor_change.max() or 0.0),
        "display_max_abs_adjusted_open_gap": float(adjusted_open_gap.max() or 0.0),
        "display_max_daily_return": float(daily_return.max()),
        "display_min_daily_return": float(daily_return.min()),
        "display_one_price_days": int(one_price.sum()),
    }


def _parser_evidence(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "primary_case1_segments": sum(
            str(item["event_type"]) == "segment" and int(item["break_case"]) == 1
            for item in records
        ),
        "primary_case2_segments": sum(
            str(item["event_type"]) == "segment" and int(item["break_case"]) == 2
            for item in records
        ),
        "primary_pending_opened": sum(
            str(item["event_type"]) == "segment_state"
            and str(item["action"]) == "opened"
            for item in records
        ),
        "primary_pending_confirmed": sum(
            str(item["event_type"]) == "segment_state"
            and str(item["action"]) == "confirmed"
            for item in records
        ),
        "primary_pending_invalidated": sum(
            str(item["event_type"]) == "segment_state"
            and str(item["action"]) == "invalidated"
            for item in records
        ),
    }


def _compare(left: float, operator_name: str, right: float) -> bool:
    if operator_name == "<":
        return left < right
    if operator_name == "<=":
        return left <= right
    if operator_name == "==":
        return left == right
    if operator_name == ">=":
        return left >= right
    if operator_name == ">":
        return left > right
    raise ValueError(f"strict_chan_audit_unknown_operator:{operator_name}")


def _evaluate_assertions(
    assertions: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for assertion in assertions:
        metric = str(assertion["metric"])
        if metric not in metrics:
            raise ValueError(f"strict_chan_audit_assertion_metric_missing:{metric}")
        actual = float(metrics[metric])
        expected = float(assertion["value"])
        passed = bool(
            np.isfinite(actual)
            and _compare(actual, str(assertion["operator"]), expected)
        )
        results.append(
            {
                "metric": metric,
                "operator": str(assertion["operator"]),
                "expected": expected,
                "actual": actual,
                "passed": passed,
            }
        )
    return results


def _event_signature(record: Mapping[str, Any]) -> tuple[Any, ...]:
    optional = (
        "direction",
        "kind",
        "side",
        "point_type",
        "level",
        "break_case",
        "action",
        "relation",
        "metric",
    )
    return (
        int(record["episode_id"]),
        str(record["event_type"]),
        str(record["id"]),
        int(record["event_index"]),
        int(record["confirmed_index"]),
        *(record.get(name) for name in optional),
    )


def _event_output_frame(
    profile_records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> pd.DataFrame:
    """Persist heterogeneous event payloads without forcing mixed Arrow types."""
    rows: list[dict[str, Any]] = []
    for records in profile_records.values():
        for record in records:
            rows.append(
                {
                    "case_id": str(record["case_id"]),
                    "category": str(record["category"]),
                    "symbol": str(record["symbol"]),
                    "profile": str(record["profile"]),
                    "episode_id": int(record["episode_id"]),
                    "event_type": str(record["event_type"]),
                    "id": str(record["id"]),
                    "event_index": int(record["event_index"]),
                    "confirmed_index": int(record["confirmed_index"]),
                    "event_time": str(record["event_time"]),
                    "confirmed_time": str(record["confirmed_time"]),
                    "payload_json": json.dumps(
                        dict(record),
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=(
            "case_id",
            "category",
            "symbol",
            "profile",
            "episode_id",
            "event_type",
            "id",
            "event_index",
            "confirmed_index",
            "event_time",
            "confirmed_time",
            "payload_json",
        ),
    )


def _event_metric_rows(
    case: Mapping[str, Any],
    profile_records: Mapping[str, Sequence[Mapping[str, Any]]],
    raw_bars: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile, records in profile_records.items():
        by_type: dict[str, list[Mapping[str, Any]]] = {}
        for record in records:
            by_type.setdefault(str(record["event_type"]), []).append(record)
        for event_type, values in sorted(by_type.items()):
            lag = np.asarray(
                [
                    int(item["confirmed_index"]) - int(item["event_index"])
                    for item in values
                ],
                dtype=np.int64,
            )
            rows.append(
                {
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "symbol": case["symbol"],
                    "profile": profile,
                    "event_type": event_type,
                    "raw_bars": raw_bars,
                    "event_count": len(values),
                    "events_per_1000_raw_bars": len(values) * 1000.0 / max(raw_bars, 1),
                    "confirmation_lag_bars_mean": float(lag.mean()),
                    "confirmation_lag_bars_median": float(np.median(lag)),
                    "confirmation_lag_bars_p90": float(np.quantile(lag, 0.90)),
                    "confirmation_lag_bars_maximum": int(lag.max()),
                }
            )
    return rows


def _disagreement_rows(
    case: Mapping[str, Any],
    profile_records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    primary_by_type: dict[str, set[tuple[Any, ...]]] = {}
    for record in profile_records["primary"]:
        primary_by_type.setdefault(str(record["event_type"]), set()).add(
            _event_signature(record)
        )
    rows: list[dict[str, Any]] = []
    for profile, records in profile_records.items():
        if profile == "primary":
            continue
        comparison_by_type: dict[str, set[tuple[Any, ...]]] = {}
        for record in records:
            comparison_by_type.setdefault(str(record["event_type"]), set()).add(
                _event_signature(record)
            )
        event_types = sorted(set(primary_by_type) | set(comparison_by_type))
        for event_type in event_types:
            primary = primary_by_type.get(event_type, set())
            comparison = comparison_by_type.get(event_type, set())
            intersection = primary & comparison
            union = primary | comparison
            rows.append(
                {
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "symbol": case["symbol"],
                    "profile": profile,
                    "event_type": event_type,
                    "primary_count": len(primary),
                    "comparison_count": len(comparison),
                    "intersection_count": len(intersection),
                    "union_count": len(union),
                    "jaccard": len(intersection) / len(union) if union else 1.0,
                }
            )
    return rows


def _pending_state_rows(
    case: Mapping[str, Any],
    profile_records: Mapping[str, Sequence[Mapping[str, Any]]],
    episode_lengths: Mapping[int, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile, records in profile_records.items():
        state_records = [
            item for item in records if str(item["event_type"]) == "segment_state"
        ]
        by_key: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
        for record in state_records:
            key = (int(record["episode_id"]), str(record["candidate_id"]))
            by_key.setdefault(key, []).append(record)
        for (episode_id, candidate_id), values in sorted(by_key.items()):
            ordered = sorted(
                values,
                key=lambda item: (
                    int(item["confirmed_index"]),
                    str(item["action"]),
                ),
            )
            opened = next(
                (item for item in ordered if str(item["action"]) == "opened"), None
            )
            if opened is None:
                raise ValueError("strict_chan_audit_pending_without_open")
            resolutions = [
                item
                for item in ordered
                if str(item["action"]) in {"confirmed", "invalidated"}
                and int(item["confirmed_index"]) >= int(opened["confirmed_index"])
            ]
            resolution = resolutions[0] if resolutions else None
            end_index = (
                int(resolution["confirmed_index"])
                if resolution is not None
                else int(episode_lengths[episode_id]) - 1
            )
            opened_time = pd.Timestamp(opened["confirmed_time"])
            end_time = (
                pd.Timestamp(resolution["confirmed_time"])
                if resolution is not None
                else pd.NaT
            )
            rows.append(
                {
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "symbol": case["symbol"],
                    "profile": profile,
                    "episode_id": episode_id,
                    "candidate_id": candidate_id,
                    "direction": int(opened["direction"]),
                    "opened_index": int(opened["confirmed_index"]),
                    "opened_time": opened_time.isoformat(),
                    "outcome": (
                        str(resolution["action"])
                        if resolution is not None
                        else "unresolved_at_episode_end"
                    ),
                    "resolution_index": (
                        int(resolution["confirmed_index"])
                        if resolution is not None
                        else pd.NA
                    ),
                    "resolution_time": (
                        end_time.isoformat() if resolution is not None else None
                    ),
                    "pending_bars": max(end_index - int(opened["confirmed_index"]), 0),
                    "pending_calendar_hours": (
                        float((end_time - opened_time).total_seconds() / 3600.0)
                        if resolution is not None
                        else math.nan
                    ),
                }
            )
    return rows


def _prefix_cutoffs(length: int, count: int) -> list[int]:
    if length < 1 or count < 1:
        raise ValueError("strict_chan_audit_prefix_cutoff_arguments_invalid")
    return np.unique(
        np.linspace(1, length, num=min(length, count), dtype=np.int64)
    ).tolist()


def _render_case_chart(
    case: Mapping[str, Any],
    frame: pd.DataFrame,
    primary_records: Sequence[Mapping[str, Any]],
    path: Path,
    visualization: Mapping[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    selected = _display_frame(frame, case)
    start_time = pd.Timestamp(selected["timestamp"].min())
    end_time = pd.Timestamp(selected["timestamp"].max())
    fig, (price_axis, raw_axis) = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.2]},
    )
    calendar_dates = pd.to_datetime(selected["trade_date"])
    large_gap = calendar_dates.diff().dt.days.gt(7)
    adjusted_plot = selected["close"].astype(float).copy()
    raw_plot = selected["raw_close"].astype(float).copy()
    adjusted_plot.loc[large_gap] = np.nan
    raw_plot.loc[large_gap] = np.nan
    price_axis.plot(
        selected["timestamp"],
        adjusted_plot,
        color="#2d3748",
        linewidth=0.75,
        alpha=0.85,
        label="adjusted 5m close",
    )

    records = [
        item
        for item in primary_records
        if pd.Timestamp(item["confirmed_time"]) >= start_time
        and pd.Timestamp(item["event_time"]) <= end_time
    ]
    by_id = {str(item["id"]): item for item in primary_records}

    if visualization.get("show_fractals", True):
        for kind, marker, color, label in (
            (1, "v", "#c53030", "top fractal"),
            (-1, "^", "#16856b", "bottom fractal"),
        ):
            values = [
                item
                for item in records
                if item["event_type"] == "fractal" and int(item["kind"]) == kind
            ]
            if values:
                price_axis.scatter(
                    [pd.Timestamp(item["event_time"]) for item in values],
                    [float(item["extreme_price"]) for item in values],
                    marker=marker,
                    s=12,
                    color=color,
                    alpha=0.45,
                    linewidths=0,
                    label=label,
                    zorder=3,
                )

    first_gap = True
    for position in np.flatnonzero(large_gap.to_numpy(dtype=bool)):
        left = pd.Timestamp(selected.iloc[position - 1]["timestamp"])
        right = pd.Timestamp(selected.iloc[position]["timestamp"])
        price_axis.axvspan(
            left,
            right,
            color="#718096",
            alpha=0.08,
            label="no-trade calendar gap" if first_gap else None,
            zorder=-1,
        )
        raw_axis.axvspan(left, right, color="#718096", alpha=0.08, zorder=-1)
        first_gap = False

    if visualization.get("show_strokes", True):
        for item in records:
            if item["event_type"] != "stroke":
                continue
            color = "#c53030" if int(item["direction"]) > 0 else "#16856b"
            price_axis.plot(
                [pd.Timestamp(item["start_time"]), pd.Timestamp(item["end_time"])],
                [float(item["start_price"]), float(item["end_price"])],
                color=color,
                linewidth=0.8,
                alpha=0.34,
                zorder=2,
            )

    if visualization.get("show_segments", True):
        first_segment = True
        for item in records:
            if item["event_type"] != "segment":
                continue
            color = "#9b2c2c" if int(item["direction"]) > 0 else "#087f8c"
            price_axis.plot(
                [pd.Timestamp(item["start_time"]), pd.Timestamp(item["end_time"])],
                [float(item["start_price"]), float(item["end_price"])],
                color=color,
                linewidth=2.0,
                alpha=0.9,
                label="confirmed segment" if first_segment else None,
                zorder=4,
            )
            first_segment = False

    if visualization.get("show_primary_centers", True):
        maximum_level = int(visualization.get("maximum_center_level", 2))
        first_center = True
        for item in records:
            if item["event_type"] != "center" or int(item["level"]) > maximum_level:
                continue
            component_ids = item.get("component_ids", ())
            if isinstance(component_ids, str):
                continue
            components = [
                by_id.get(str(component_id)) for component_id in component_ids
            ]
            if not components or any(component is None for component in components):
                continue
            left = min(
                pd.Timestamp(component["start_time"]) for component in components
            )
            right = max(pd.Timestamp(component["end_time"]) for component in components)
            left = max(left, start_time)
            right = min(right, end_time)
            if right <= left:
                continue
            low = float(item["core_low"])
            high = float(item["core_high"])
            rectangle = Rectangle(
                (mdates.date2num(left), low),
                mdates.date2num(right) - mdates.date2num(left),
                high - low,
                facecolor="#d69e2e",
                edgecolor="#b7791f",
                linewidth=0.8,
                alpha=0.10 if int(item["level"]) == 1 else 0.18,
                label="center core" if first_center else None,
                zorder=1,
            )
            price_axis.add_patch(rectangle)
            first_center = False

    if visualization.get("show_pending_state_events", True):
        state_colors = {
            "opened": "#dd6b20",
            "confirmed": "#805ad5",
            "invalidated": "#718096",
        }
        used_actions: set[str] = set()
        for item in records:
            if item["event_type"] != "segment_state":
                continue
            action = str(item["action"])
            price_axis.axvline(
                pd.Timestamp(item["confirmed_time"]),
                color=state_colors[action],
                linewidth=0.9,
                alpha=0.65,
                linestyle=":" if action == "opened" else "--",
                label=f"pending {action}" if action not in used_actions else None,
                zorder=0,
            )
            used_actions.add(action)

    if visualization.get("show_trade_point_candidates", True):
        points = [item for item in records if item["event_type"] == "trade_point"]
        if points:
            price_axis.scatter(
                [pd.Timestamp(item["event_time"]) for item in points],
                [float(item["price"]) for item in points],
                marker="D",
                s=28,
                facecolors="none",
                edgecolors="#6b46c1",
                linewidths=1.1,
                label="point candidate",
                zorder=5,
            )

    focal = pd.Timestamp(str(case["focal_date"]))
    price_axis.axvline(
        focal,
        color="#111827",
        linewidth=1.2,
        linestyle="--",
        alpha=0.8,
        label="focal date",
    )
    price_axis.set_xlim(start_time, end_time)
    price_axis.set_ylabel("same-day-factor adjusted price")
    price_axis.grid(True, linewidth=0.35, alpha=0.25)

    adjusted_base = float(selected["close"].iloc[0])
    raw_base = float(selected["raw_close"].iloc[0])
    raw_axis.plot(
        selected["timestamp"],
        adjusted_plot / adjusted_base * 100.0,
        color="#2b6cb0",
        linewidth=0.9,
        label="adjusted close, base=100",
    )
    raw_axis.plot(
        selected["timestamp"],
        raw_plot / raw_base * 100.0,
        color="#c05621",
        linewidth=0.8,
        alpha=0.85,
        label="raw close, base=100",
    )
    raw_axis.axvline(focal, color="#111827", linewidth=1.0, linestyle="--")
    raw_axis.set_ylabel("normalized close")
    raw_axis.grid(True, linewidth=0.35, alpha=0.25)
    raw_axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
    raw_axis.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(raw_axis.xaxis.get_major_locator())
    )

    for axis in (price_axis, raw_axis):
        handles, labels = axis.get_legend_handles_labels()
        unique: dict[str, Any] = {}
        for handle, label in zip(handles, labels, strict=True):
            if label and label not in unique:
                unique[label] = handle
        axis.legend(
            unique.values(),
            unique.keys(),
            loc="upper left",
            fontsize=8,
            ncol=min(4, max(len(unique), 1)),
            frameon=False,
        )
    fig.suptitle(
        f"{case['case_id']} | {case['symbol']} | {case['category']} | primary",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(temporary, dpi=150, format="png")
    plt.close(fig)
    os.replace(temporary, path)


def _completed_case(
    case_root: Path, fingerprint: str, *, require_chart: bool
) -> dict[str, Any] | None:
    summary_path = case_root / "summary.json"
    required = (
        case_root / "input_bars.parquet",
        case_root / "profile_events.parquet",
        case_root / "event_metrics.parquet",
        case_root / "variant_disagreement.parquet",
        case_root / "pending_states.parquet",
        case_root / "prefix_metrics.parquet",
    )
    if require_chart:
        required = (*required, case_root / "chart.png")
    if not summary_path.is_file() or any(not path.is_file() for path in required):
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "completed"
        or summary.get("fingerprint") != fingerprint
    ):
        return None
    return summary


def _run_case(
    case: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    definition_path: str | Path,
    output_root: Path,
    run_prefix_validation: bool,
    render_chart: bool,
    resume: bool,
) -> dict[str, Any]:
    case_root = output_root / "cases" / str(case["case_id"])
    fingerprint = _case_fingerprint(case, spec, definition_path=definition_path)
    if resume:
        completed = _completed_case(case_root, fingerprint, require_chart=render_chart)
        if (
            completed is not None
            and bool(completed.get("prefix_validation_performed"))
            == run_prefix_validation
        ):
            return completed

    loaded = intraday.load_symbol_episodes(
        str(case["symbol"]),
        start_date=str(case["start_date"]),
        end_date=str(case["end_date"]),
        definition_path=definition_path,
        temporary_root=case_root / "duckdb_tmp",
    )
    pool_member = _quality_pool_membership(case, spec["_quality_pool_manifest_path"])
    if pool_member != bool(case["quality_pool_expected"]):
        raise ValueError(f"strict_chan_audit_pool_expectation_failed:{case['case_id']}")

    input_bars = loaded.frame.copy()
    input_bars.insert(0, "case_id", str(case["case_id"]))
    _write_parquet(case_root / "input_bars.parquet", input_bars)

    profiles = dict(spec["profiles"])
    profile_records: dict[str, list[dict[str, Any]]] = {
        profile: [] for profile in profiles
    }
    episode_lengths: dict[int, int] = {}
    for episode_id, frame in loaded.frame.groupby("episode_id", sort=True):
        episode_id = int(episode_id)
        parse_frame = frame.loc[:, parser.REQUIRED_INPUT_COLUMNS].reset_index(drop=True)
        episode_lengths[episode_id] = len(parse_frame)
        for profile, overrides in profiles.items():
            result = parser.parse_strict_chan(
                parse_frame,
                definition_path=definition_path,
                variant_overrides=overrides,
            )
            for record in result.event_records():
                item = dict(record)
                item["case_id"] = str(case["case_id"])
                item["category"] = str(case["category"])
                item["symbol"] = str(case["symbol"])
                item["profile"] = profile
                item["episode_id"] = episode_id
                profile_records[profile].append(item)

    _write_parquet(
        case_root / "profile_events.parquet", _event_output_frame(profile_records)
    )

    evidence = _daily_evidence(loaded.frame, case)
    evidence.update(_parser_evidence(profile_records["primary"]))
    assertion_results = _evaluate_assertions(case.get("assertions", []), evidence)
    failed_assertions = [item for item in assertion_results if not item["passed"]]
    if failed_assertions:
        raise ValueError(
            "strict_chan_audit_case_assertion_failed:"
            + json.dumps(
                {
                    "case_id": case["case_id"],
                    "failures": failed_assertions,
                },
                sort_keys=True,
                default=str,
            )
        )

    event_metrics = pd.DataFrame(
        _event_metric_rows(case, profile_records, len(loaded.frame))
    )
    disagreement = pd.DataFrame(_disagreement_rows(case, profile_records))
    pending_rows = _pending_state_rows(case, profile_records, episode_lengths)
    pending = pd.DataFrame(
        pending_rows,
        columns=(
            "case_id",
            "category",
            "symbol",
            "profile",
            "episode_id",
            "candidate_id",
            "direction",
            "opened_index",
            "opened_time",
            "outcome",
            "resolution_index",
            "resolution_time",
            "pending_bars",
            "pending_calendar_hours",
        ),
    )
    _write_parquet(case_root / "event_metrics.parquet", event_metrics)
    _write_parquet(case_root / "variant_disagreement.parquet", disagreement)
    _write_parquet(case_root / "pending_states.parquet", pending)

    prefix_rows: list[dict[str, Any]] = []
    prefix_details: list[dict[str, Any]] = []
    if run_prefix_validation:
        contract = dict(spec["prefix_validation"])
        maximum_episodes = int(contract["maximum_episodes_per_case"])
        selected_episodes = sorted(
            episode_lengths,
            key=lambda item: (-episode_lengths[item], item),
        )[:maximum_episodes]
        for episode_id in selected_episodes:
            frame = loaded.frame[loaded.frame["episode_id"].eq(episode_id)]
            parse_frame = frame.loc[:, parser.REQUIRED_INPUT_COLUMNS].reset_index(
                drop=True
            )
            cutoffs = _prefix_cutoffs(
                len(parse_frame), int(contract["cutoffs_per_episode"])
            )
            for profile, overrides in profiles.items():
                validation = validator.verify_prefix_invariance(
                    parse_frame,
                    cutoffs=cutoffs,
                    definition_path=definition_path,
                    variant_overrides=overrides,
                )
                prefix_details.append(
                    {
                        "profile": profile,
                        "episode_id": episode_id,
                        "validation": validation,
                    }
                )
                prefix_rows.append(
                    {
                        "case_id": case["case_id"],
                        "category": case["category"],
                        "symbol": case["symbol"],
                        "profile": profile,
                        "episode_id": episode_id,
                        "bars": validation["bars"],
                        "cutoffs": validation["cutoffs"],
                        "event_records_compared": validation["event_records_compared"],
                        "event_fields_compared": validation["event_fields_compared"],
                        "future_confirmed_structure_written_back": validation[
                            "future_confirmed_structure_written_back"
                        ],
                    }
                )
    prefix_frame = pd.DataFrame(
        prefix_rows,
        columns=(
            "case_id",
            "category",
            "symbol",
            "profile",
            "episode_id",
            "bars",
            "cutoffs",
            "event_records_compared",
            "event_fields_compared",
            "future_confirmed_structure_written_back",
        ),
    )
    _write_parquet(case_root / "prefix_metrics.parquet", prefix_frame)
    _write_json(
        case_root / "prefix_validation.json",
        {
            "schema": validator.VALIDATION_SCHEMA_VERSION,
            "performed": run_prefix_validation,
            "validations": prefix_details,
        },
    )

    chart_path = case_root / "chart.png"
    if render_chart:
        _render_case_chart(
            case,
            loaded.frame,
            profile_records["primary"],
            chart_path,
            spec["visualization"],
        )

    primary_metrics = event_metrics[event_metrics["profile"].eq("primary")]
    summary = {
        "schema": AUDIT_SCHEMA_VERSION,
        "status": "completed",
        "study_id": spec["study_id"],
        "parent_study_id": spec["parent_study_id"],
        "case_id": case["case_id"],
        "category": case["category"],
        "symbol": case["symbol"],
        "start_date": case["start_date"],
        "end_date": case["end_date"],
        "display_start_date": case["display_start_date"],
        "display_end_date": case["display_end_date"],
        "focal_date": case["focal_date"],
        "selection_basis": case["selection_basis"],
        "quality_pool_expected": bool(case["quality_pool_expected"]),
        "quality_pool_member_at_focal_date": pool_member,
        "fingerprint": fingerprint,
        "raw_bars": len(loaded.frame),
        "usable_days": int(loaded.audit["usable_adjusted_days"]),
        "missing_positive_days": int(loaded.audit["missing_positive_days"]),
        "episode_count": int(loaded.audit["episode_count"]),
        "primary_event_counts": {
            str(row.event_type): int(row.event_count)
            for row in primary_metrics.itertuples(index=False)
        },
        "case_evidence": evidence,
        "assertions": assertion_results,
        "profiles": list(profiles),
        "prefix_validation_performed": run_prefix_validation,
        "prefix_event_records_compared": int(
            prefix_frame["event_records_compared"].sum()
        )
        if not prefix_frame.empty
        else 0,
        "prefix_event_fields_compared": int(prefix_frame["event_fields_compared"].sum())
        if not prefix_frame.empty
        else 0,
        "future_confirmed_structure_written_back": False,
        "chart_rendered": render_chart,
        "chart_path": str(chart_path.resolve()) if render_chart else None,
        "case_root": str(case_root.resolve()),
        "future_path_test_performed": False,
        "return_test_performed": False,
        "profit_claim": False,
    }
    _write_json(case_root / "input_audit.json", dict(loaded.audit))
    _write_json(case_root / "summary.json", summary)
    return summary


def _aggregate_disagreement(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(["profile", "event_type"], sort=True, observed=True)
    rows: list[dict[str, Any]] = []
    for (profile, event_type), values in grouped:
        union = int(values["union_count"].sum())
        intersection = int(values["intersection_count"].sum())
        rows.append(
            {
                "profile": profile,
                "event_type": event_type,
                "cases": int(values["case_id"].nunique()),
                "primary_count_sum": int(values["primary_count"].sum()),
                "comparison_count_sum": int(values["comparison_count"].sum()),
                "intersection_count_sum": intersection,
                "union_count_sum": union,
                "micro_jaccard": intersection / union if union else 1.0,
                "case_jaccard_median": float(values["jaccard"].median()),
                "case_jaccard_minimum": float(values["jaccard"].min()),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_pending(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=(
                "profile",
                "opened",
                "confirmed",
                "invalidated",
                "unresolved",
                "resolved_fraction",
                "pending_bars_median",
                "pending_bars_p90",
                "pending_bars_maximum",
            )
        )
    rows: list[dict[str, Any]] = []
    for profile, values in frame.groupby("profile", sort=True, observed=True):
        outcome = values["outcome"]
        resolved = outcome.isin(["confirmed", "invalidated"])
        rows.append(
            {
                "profile": profile,
                "opened": len(values),
                "confirmed": int(outcome.eq("confirmed").sum()),
                "invalidated": int(outcome.eq("invalidated").sum()),
                "unresolved": int(outcome.eq("unresolved_at_episode_end").sum()),
                "resolved_fraction": float(resolved.mean()),
                "pending_bars_median": float(values["pending_bars"].median()),
                "pending_bars_p90": float(values["pending_bars"].quantile(0.90)),
                "pending_bars_maximum": int(values["pending_bars"].max()),
            }
        )
    return pd.DataFrame(rows)


def _build_contact_sheet(summaries: Sequence[Mapping[str, Any]], path: Path) -> None:
    from PIL import Image, ImageDraw

    entries = [
        (str(item["case_id"]), Path(str(item["chart_path"])))
        for item in summaries
        if item.get("chart_rendered") and item.get("chart_path")
    ]
    if not entries:
        return
    columns = min(3, len(entries))
    rows = math.ceil(len(entries) / columns)
    tile_width = 720
    tile_height = 405
    label_height = 34
    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * (tile_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for position, (label, image_path) in enumerate(entries):
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((tile_width, tile_height))
        x = (position % columns) * tile_width
        y = (position // columns) * (tile_height + label_height)
        paste_x = x + (tile_width - image.width) // 2
        paste_y = y + label_height + (tile_height - image.height) // 2
        sheet.paste(image, (paste_x, paste_y))
        draw.text((x + 8, y + 9), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    sheet.save(temporary, format="PNG")
    os.replace(temporary, path)


def _visual_review_state(
    path: Path, summaries: Sequence[Mapping[str, Any]]
) -> tuple[str, dict[str, Mapping[str, Any]], str | None]:
    if not path.is_file():
        return "unreviewed", {}, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "stale", {}, "visual_review_unreadable"
    if payload.get("schema") != "seq100_strict_chan_visual_review/1":
        return "stale", {}, "visual_review_schema_mismatch"
    reviews = {
        str(item.get("case_id", "")): item for item in payload.get("reviews", [])
    }
    for summary in summaries:
        case_id = str(summary["case_id"])
        review = reviews.get(case_id)
        if review is None:
            return "stale", {}, f"visual_review_case_missing:{case_id}"
        if str(review.get("fingerprint", "")) != str(summary["fingerprint"]):
            return "stale", {}, f"visual_review_fingerprint_mismatch:{case_id}"
        if review.get("review_status") not in {"passed", "needs_followup"}:
            return "stale", {}, f"visual_review_status_invalid:{case_id}"
    if any(
        bool(reviews[str(item["case_id"])].get("parser_error")) for item in summaries
    ):
        return "issues_found", reviews, None
    if any(
        reviews[str(item["case_id"])]["review_status"] == "needs_followup"
        for item in summaries
    ):
        return "needs_followup", reviews, None
    return "passed", reviews, None


def run_audit(
    *,
    audit_path: str | Path = DEFAULT_AUDIT_PATH,
    definition_path: str | Path = parser.DEFAULT_DEFINITION_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    case_ids: Sequence[str] | None = None,
    run_prefix_validation: bool = True,
    render_charts: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    spec = load_audit_spec(audit_path)
    root = Path(output_root).resolve()
    requested = set(case_ids or [])
    cases = [
        case for case in spec["cases"] if not requested or case["case_id"] in requested
    ]
    unknown = requested - {str(case["case_id"]) for case in cases}
    if unknown:
        raise ValueError(f"strict_chan_audit_unknown_cases:{','.join(sorted(unknown))}")
    if not cases:
        raise ValueError("strict_chan_audit_no_cases_selected")

    summaries = [
        _run_case(
            case,
            spec,
            definition_path=definition_path,
            output_root=root,
            run_prefix_validation=run_prefix_validation,
            render_chart=render_charts,
            resume=resume,
        )
        for case in cases
    ]

    case_roots = [Path(str(item["case_root"])) for item in summaries]
    event_metrics = pd.concat(
        [pd.read_parquet(path / "event_metrics.parquet") for path in case_roots],
        ignore_index=True,
    )
    disagreement = pd.concat(
        [pd.read_parquet(path / "variant_disagreement.parquet") for path in case_roots],
        ignore_index=True,
    )
    pending = pd.concat(
        [pd.read_parquet(path / "pending_states.parquet") for path in case_roots],
        ignore_index=True,
    )
    prefix = pd.concat(
        [pd.read_parquet(path / "prefix_metrics.parquet") for path in case_roots],
        ignore_index=True,
    )
    case_metrics = pd.DataFrame(
        [
            {
                "case_id": item["case_id"],
                "category": item["category"],
                "symbol": item["symbol"],
                "focal_date": item["focal_date"],
                "quality_pool_member_at_focal_date": item[
                    "quality_pool_member_at_focal_date"
                ],
                "raw_bars": item["raw_bars"],
                "usable_days": item["usable_days"],
                "missing_positive_days": item["missing_positive_days"],
                "episode_count": item["episode_count"],
                "prefix_event_records_compared": item["prefix_event_records_compared"],
                "prefix_event_fields_compared": item["prefix_event_fields_compared"],
            }
            for item in summaries
        ]
    )
    aggregate_disagreement = _aggregate_disagreement(disagreement)
    aggregate_pending = _aggregate_pending(pending)
    _write_parquet(root / "case_metrics.parquet", case_metrics)
    _write_parquet(root / "event_metrics.parquet", event_metrics)
    _write_parquet(root / "variant_disagreement.parquet", disagreement)
    _write_parquet(root / "pending_states.parquet", pending)
    _write_parquet(root / "prefix_metrics.parquet", prefix)
    _write_parquet(
        root / "aggregate_variant_disagreement.parquet", aggregate_disagreement
    )
    _write_parquet(root / "aggregate_pending_metrics.parquet", aggregate_pending)

    contact_sheet = root / "visual_audit_contact_sheet.png"
    if render_charts:
        _build_contact_sheet(summaries, contact_sheet)
    review_path = root / "visual_review.json"
    visual_review_status, reviews, visual_review_stale_reason = _visual_review_state(
        review_path, summaries
    )
    review_rows = pd.DataFrame(
        [
            {
                "case_id": item["case_id"],
                "category": item["category"],
                "symbol": item["symbol"],
                "review_status": reviews.get(str(item["case_id"]), {}).get(
                    "review_status", "unreviewed"
                ),
                "parser_error": reviews.get(str(item["case_id"]), {}).get(
                    "parser_error", pd.NA
                ),
                "variant_disagreement_only": reviews.get(str(item["case_id"]), {}).get(
                    "variant_disagreement_only", pd.NA
                ),
                "notes": reviews.get(str(item["case_id"]), {}).get("notes", pd.NA),
                "chart_path": item["chart_path"],
            }
            for item in summaries
        ]
    )
    _write_parquet(root / "visual_review_ledger.parquet", review_rows)

    primary_counts = (
        event_metrics[event_metrics["profile"].eq("primary")]
        .groupby("event_type", sort=True)["event_count"]
        .sum()
    )
    all_cases_selected = len(cases) == len(spec["cases"])
    summary = {
        "schema": AUDIT_SCHEMA_VERSION,
        "status": "completed" if all_cases_selected else "completed_subset",
        "study_id": spec["study_id"],
        "parent_study_id": spec["parent_study_id"],
        "cases_configured": len(spec["cases"]),
        "cases_completed": len(summaries),
        "categories_completed": sorted({str(item["category"]) for item in summaries}),
        "profiles": list(spec["profiles"]),
        "raw_bars": int(case_metrics["raw_bars"].sum()),
        "usable_days": int(case_metrics["usable_days"].sum()),
        "missing_positive_days": int(case_metrics["missing_positive_days"].sum()),
        "episode_count": int(case_metrics["episode_count"].sum()),
        "primary_event_counts": {
            str(name): int(value) for name, value in primary_counts.items()
        },
        "pending_candidates": len(pending),
        "prefix_validation_performed": run_prefix_validation,
        "prefix_event_records_compared": int(prefix["event_records_compared"].sum())
        if not prefix.empty
        else 0,
        "prefix_event_fields_compared": int(prefix["event_fields_compared"].sum())
        if not prefix.empty
        else 0,
        "future_confirmed_structure_written_back": False,
        "all_case_assertions_passed": True,
        "visual_review_status": visual_review_status,
        "visual_review_stale_reason": visual_review_stale_reason,
        "visual_review_path": str(review_path.resolve())
        if review_path.is_file()
        else None,
        "contact_sheet_path": str(contact_sheet.resolve()) if render_charts else None,
        "case_metrics_path": str((root / "case_metrics.parquet").resolve()),
        "event_metrics_path": str((root / "event_metrics.parquet").resolve()),
        "variant_disagreement_path": str(
            (root / "variant_disagreement.parquet").resolve()
        ),
        "aggregate_variant_disagreement_path": str(
            (root / "aggregate_variant_disagreement.parquet").resolve()
        ),
        "pending_states_path": str((root / "pending_states.parquet").resolve()),
        "aggregate_pending_metrics_path": str(
            (root / "aggregate_pending_metrics.parquet").resolve()
        ),
        "prefix_metrics_path": str((root / "prefix_metrics.parquet").resolve()),
        "visual_review_ledger_path": str(
            (root / "visual_review_ledger.parquet").resolve()
        ),
        "future_path_test_performed": False,
        "return_test_performed": False,
        "profit_claim": False,
    }
    _write_json(root / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Run the frozen cross-path audit for the strict Chan parser."
    )
    argument_parser.add_argument("--audit", default=str(DEFAULT_AUDIT_PATH))
    argument_parser.add_argument(
        "--definition", default=str(parser.DEFAULT_DEFINITION_PATH)
    )
    argument_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    argument_parser.add_argument("--case-id", action="append", default=[])
    argument_parser.add_argument("--skip-prefix-validation", action="store_true")
    argument_parser.add_argument("--skip-charts", action="store_true")
    argument_parser.add_argument("--no-resume", action="store_true")
    return argument_parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_audit(
        audit_path=args.audit,
        definition_path=args.definition,
        output_root=args.output_root,
        case_ids=args.case_id,
        run_prefix_validation=not args.skip_prefix_validation,
        render_charts=not args.skip_charts,
        resume=not args.no_resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
