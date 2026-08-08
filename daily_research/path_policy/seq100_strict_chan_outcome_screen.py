"""Expanded outcome-blind screen for strict-Chan buy-point families."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_strict_chan_intraday as intraday
from daily_research.path_policy import seq100_strict_chan_outcome_probe as probe
from daily_research.path_policy import seq100_strict_chan_parser as parser
from daily_research.path_policy import seq100_strict_chan_stratified_audit as sample

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_strict_chan_outcome_screen_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_strict_chan_outcome_screen_v1"
)
STUDY_ID = "seq100_strict_chan_outcome_screen_v1"
CONFIRMATION_STUDY_ID = "seq100_strict_chan_type1_confirmation_v1"
SUPPORTED_STUDY_IDS = {STUDY_ID, CONFIRMATION_STUDY_ID}
SCHEMA_VERSION = "seq100_strict_chan_outcome_screen/1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study_path = Path(path).resolve()
    study = json.loads(study_path.read_text(encoding="utf-8"))
    if study.get("study_id") not in SUPPORTED_STUDY_IDS:
        raise ValueError("strict_chan_outcome_screen_study_id_mismatch")
    if int(study["source"]["forbidden_year"]) != 2026:
        raise ValueError("strict_chan_outcome_screen_forbidden_year_mismatch")
    development = set(map(int, study["evaluation"]["development_focal_years"]))
    validation = set(map(int, study["evaluation"]["validation_focal_years"]))
    if not development or not validation or development & validation:
        raise ValueError("strict_chan_outcome_screen_split_invalid")
    study["_study_path"] = str(study_path)
    return study


def _sample_study_path(study: Mapping[str, Any]) -> Path:
    return (WORKSPACE_ROOT / str(study["source"]["sample_study"])).resolve()


def _evaluation_split(focal_date: str, study: Mapping[str, Any]) -> str:
    year = int(str(focal_date)[:4])
    if year in set(map(int, study["evaluation"]["development_focal_years"])):
        return "development"
    if year in set(map(int, study["evaluation"]["validation_focal_years"])):
        return "validation"
    raise ValueError(f"strict_chan_outcome_screen_focal_year_unassigned:{year}")


def _period_id(case: Mapping[str, Any]) -> str:
    return "_".join(map(str, case["stratum_years"]))


def _candidate_frame(
    case: Mapping[str, Any],
    episodes: intraday.IntradayEpisodes,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    point_types = {
        int(value) for value in study["source"].get("point_types", (1, 2, 3))
    }
    variants = {str(value) for value in study["source"].get("variants", ())}
    rows: list[dict[str, Any]] = []
    for episode_id, frame in episodes.frame.groupby("episode_id", sort=True):
        parse_frame = frame.loc[:, parser.REQUIRED_INPUT_COLUMNS].reset_index(drop=True)
        result = parser.parse_strict_chan(parse_frame)
        for point in result.trade_points:
            if (
                point.side != "buy"
                or point.point_type not in point_types
                or (variants and point.variant not in variants)
            ):
                continue
            record = asdict(point)
            confirmed_date = str(point.confirmed_time)[:10]
            if not (
                str(case["display_start_date"])
                <= confirmed_date
                <= str(case["display_end_date"])
            ):
                continue
            record.update(
                {
                    "case_id": str(case["case_id"]),
                    "symbol": str(case["symbol"]),
                    "profile": "primary",
                    "episode_id": int(episode_id),
                    "confirmed_date": confirmed_date,
                }
            )
            rows.append(record)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).sort_values(["confirmed_date", "confirmed_time", "id"])
    return frame.drop_duplicates(
        ["case_id", "point_type", "confirmed_date"], keep="first"
    )


def _baseline_rows(
    *,
    case: Mapping[str, Any],
    daily: pd.DataFrame,
    horizons: Sequence[int],
    costs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dates = daily["trade_date"].astype(str).tolist()
    for signal_index, signal_date in enumerate(dates):
        if not (
            str(case["display_start_date"])
            <= signal_date
            <= str(case["display_end_date"])
        ):
            continue
        entry_index = signal_index + 1
        if entry_index >= len(daily):
            continue
        adjusted_entry = float(daily.iloc[entry_index]["open"])
        raw_entry = float(daily.iloc[entry_index].get("raw_open", adjusted_entry))
        if not np.isfinite(adjusted_entry) or adjusted_entry <= 0:
            continue
        for horizon in horizons:
            exit_index = entry_index + int(horizon) - 1
            if exit_index >= len(daily):
                continue
            adjusted_exit = float(daily.iloc[exit_index]["close"])
            raw_exit = float(daily.iloc[exit_index].get("raw_close", adjusted_exit))
            if not np.isfinite(adjusted_exit) or adjusted_exit <= 0:
                continue
            raw_return = adjusted_exit / adjusted_entry - 1.0
            item: dict[str, Any] = {
                "schema": SCHEMA_VERSION,
                "case_id": str(case["case_id"]),
                "symbol": str(case["symbol"]),
                "stratum_id": str(case["stratum_id"]),
                "period_id": _period_id(case),
                "focal_date": str(case["focal_date"]),
                "signal_date": signal_date,
                "entry_date": str(daily.iloc[entry_index]["trade_date"]),
                "horizon_sessions": int(horizon),
                "raw_return": raw_return,
            }
            for stress in (False, True):
                buy, sell = probe._cost_multipliers(
                    raw_entry,
                    str(daily.iloc[exit_index]["trade_date"]),
                    stress=stress,
                    costs=costs,
                    exit_price=raw_exit,
                )
                item["net_return_stress" if stress else "net_return_base"] = (
                    sell * (1.0 + raw_return) / buy - 1.0
                )
            rows.append(item)
    return rows


def _partition_fingerprint(
    *,
    study: Mapping[str, Any],
    selection_fingerprint: str,
    cases: Sequence[Mapping[str, Any]],
) -> str:
    sources = (
        Path(__file__).resolve(),
        Path(parser.__file__).resolve(),
        Path(intraday.__file__).resolve(),
        Path(probe.__file__).resolve(),
        parser.DEFAULT_DEFINITION_PATH.resolve(),
        Path(str(study["_study_path"])),
    )
    payload = {
        "schema": SCHEMA_VERSION,
        "selection_fingerprint": selection_fingerprint,
        "cases": list(cases),
        "source_sha256": {path.name: _sha256_file(path) for path in sources},
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_partition(
    *,
    period_id: str,
    cases: Sequence[Mapping[str, Any]],
    study: Mapping[str, Any],
    selection_fingerprint: str,
    output_root: Path,
    resume: bool,
) -> dict[str, Any]:
    part_root = output_root / "partitions" / f"period={period_id}"
    done_path = part_root / "done.json"
    fingerprint = _partition_fingerprint(
        study=study,
        selection_fingerprint=selection_fingerprint,
        cases=cases,
    )
    required = (
        part_root / "candidates.parquet",
        part_root / "event_outcomes.parquet",
        part_root / "baseline.parquet",
    )
    if resume and done_path.is_file() and all(path.is_file() for path in required):
        completed = json.loads(done_path.read_text(encoding="utf-8"))
        if completed.get("fingerprint") == fingerprint:
            return completed

    windows = {
        str(case["symbol"]): (str(case["start_date"]), str(case["end_date"]))
        for case in cases
    }
    if len(windows) != len(cases):
        raise ValueError("strict_chan_outcome_screen_duplicate_symbol")
    loaded = intraday.load_symbol_windows_episodes(
        windows,
        temporary_root=part_root / "duckdb_tmp",
        input_provenance={"mode": "batched_outcome_screen", "period_id": period_id},
    )
    horizons = [int(value) for value in study["execution"]["horizons_sessions"]]
    costs = dict(study["execution"]["costs"])
    candidate_frames: list[pd.DataFrame] = []
    outcome_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    case_stats: list[dict[str, Any]] = []
    for case in cases:
        episodes = loaded[str(case["symbol"])]
        candidates = _candidate_frame(case, episodes, study)
        if not candidates.empty:
            candidate_frames.append(candidates)
        daily = probe._daily_frame(episodes.frame)
        rows, stats = probe._outcome_rows(
            case_id=str(case["case_id"]),
            events=candidates,
            daily=daily,
            horizons=horizons,
            costs=costs,
        )
        split = _evaluation_split(str(case["focal_date"]), study)
        metadata = {
            "stratum_id": str(case["stratum_id"]),
            "period_id": period_id,
            "focal_date": str(case["focal_date"]),
            "evaluation_split": split,
        }
        for row in rows:
            row.update(metadata)
        outcome_rows.extend(rows)
        baseline = _baseline_rows(
            case=case,
            daily=daily,
            horizons=horizons,
            costs=costs,
        )
        for row in baseline:
            row["evaluation_split"] = split
        baseline_rows.extend(baseline)
        case_stats.append(
            {
                "case_id": str(case["case_id"]),
                "symbol": str(case["symbol"]),
                "candidate_count": int(stats["candidate_count"]),
                "eligible_candidate_count": int(stats["eligible_count"]),
                "usable_days": int(episodes.audit["usable_adjusted_days"]),
                "usable_rows": int(episodes.audit["usable_rows"]),
            }
        )
    candidates_all = (
        pd.concat(candidate_frames, ignore_index=True)
        if candidate_frames
        else pd.DataFrame(columns=("case_id", "symbol", "point_type"))
    )
    outcomes = pd.DataFrame(outcome_rows)
    baseline = pd.DataFrame(baseline_rows)
    _write_parquet(part_root / "candidates.parquet", candidates_all)
    _write_parquet(part_root / "event_outcomes.parquet", outcomes)
    _write_parquet(part_root / "baseline.parquet", baseline)
    result = {
        "schema": SCHEMA_VERSION,
        "status": "completed",
        "period_id": period_id,
        "fingerprint": fingerprint,
        "case_count": len(cases),
        "candidate_count": len(candidates_all),
        "outcome_rows": len(outcomes),
        "baseline_rows": len(baseline),
        "case_stats": case_stats,
    }
    _write_json(done_path, result)
    return result


def _bootstrap_interval(
    values: pd.Series, *, resamples: int, seed: int
) -> tuple[float, float]:
    array = values.dropna().to_numpy(dtype=np.float64)
    if not len(array):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(resamples, len(array)))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _stable_seed(base_seed: int, *values: Any) -> int:
    token = "|".join(map(str, (base_seed, *values)))
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)


def _aggregate(
    outcomes: pd.DataFrame, baseline: pd.DataFrame, study: Mapping[str, Any]
) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame()
    evaluation = study["evaluation"]
    resamples = int(evaluation["bootstrap_case_resamples"])
    base_seed = int(evaluation["bootstrap_seed"])
    rows: list[dict[str, Any]] = []
    split_frames = [
        (name, outcomes[outcomes["evaluation_split"].eq(name)])
        for name in ("development", "validation")
    ]
    split_frames.append(("all", outcomes))
    for split, split_frame in split_frames:
        split_baseline = (
            baseline
            if split == "all"
            else baseline[baseline["evaluation_split"].eq(split)]
        )
        for (point_type, horizon), group in split_frame.groupby(
            ["point_type", "horizon_sessions"], sort=True
        ):
            signal_case = group.groupby("case_id")["net_return_base"].mean()
            stress_case = group.groupby("case_id")["net_return_stress"].mean()
            comparable_baseline = split_baseline[
                split_baseline["horizon_sessions"].eq(horizon)
                & split_baseline["case_id"].isin(signal_case.index)
            ]
            baseline_case = comparable_baseline.groupby("case_id")[
                "net_return_base"
            ].mean()
            common = signal_case.index.intersection(baseline_case.index)
            excess_case = signal_case.loc[common] - baseline_case.loc[common]
            mean_low, mean_high = _bootstrap_interval(
                signal_case,
                resamples=resamples,
                seed=_stable_seed(base_seed, split, point_type, horizon, "mean"),
            )
            excess_low, excess_high = _bootstrap_interval(
                excess_case,
                resamples=resamples,
                seed=_stable_seed(base_seed, split, point_type, horizon, "excess"),
            )
            case_period = (
                group[["case_id", "period_id"]]
                .drop_duplicates("case_id")
                .set_index("case_id")["period_id"]
            )
            period_means = (
                signal_case.to_frame("return")
                .join(case_period)
                .groupby("period_id")["return"]
                .mean()
            )
            rows.append(
                {
                    "schema": SCHEMA_VERSION,
                    "evaluation_split": split,
                    "profile": "primary",
                    "point_type": int(point_type),
                    "horizon_sessions": int(horizon),
                    "event_count": len(group),
                    "case_count": len(signal_case),
                    "period_count": len(period_means),
                    "event_mean_net_base": float(group["net_return_base"].mean()),
                    "event_median_net_base": float(group["net_return_base"].median()),
                    "event_win_rate_base": float(group["net_return_base"].gt(0).mean()),
                    "case_mean_net_base": float(signal_case.mean()),
                    "case_mean_net_stress": float(stress_case.mean()),
                    "case_positive_fraction_base": float(signal_case.gt(0).mean()),
                    "baseline_case_mean_net_base": float(baseline_case.mean()),
                    "case_mean_excess_base": float(excess_case.mean()),
                    "bootstrap_case_mean_base_low": mean_low,
                    "bootstrap_case_mean_base_high": mean_high,
                    "bootstrap_case_excess_low": excess_low,
                    "bootstrap_case_excess_high": excess_high,
                    "positive_period_fraction_base": float(period_means.gt(0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _gate(aggregate: pd.DataFrame, study: Mapping[str, Any]) -> pd.DataFrame:
    evaluation = study["evaluation"]
    rows: list[dict[str, Any]] = []
    families = (
        aggregate[["point_type", "horizon_sessions"]]
        .drop_duplicates()
        .sort_values(["point_type", "horizon_sessions"])
    )
    for family in families.itertuples(index=False):
        reasons: list[str] = []
        for split, minimum_cases in (
            ("development", int(evaluation["minimum_cases_development"])),
            ("validation", int(evaluation["minimum_cases_validation"])),
        ):
            selected = aggregate[
                aggregate["evaluation_split"].eq(split)
                & aggregate["point_type"].eq(family.point_type)
                & aggregate["horizon_sessions"].eq(family.horizon_sessions)
            ]
            if selected.empty:
                reasons.append(f"{split}:missing")
                continue
            item = selected.iloc[0]
            checks = {
                "events": int(item["event_count"])
                >= int(evaluation["minimum_events_per_split"]),
                "cases": int(item["case_count"]) >= minimum_cases,
                "net_base": float(item["case_mean_net_base"]) > 0,
                "net_stress": float(item["case_mean_net_stress"]) > 0,
                "excess": float(item["case_mean_excess_base"]) > 0,
                "excess_ci": float(item["bootstrap_case_excess_low"]) > 0,
            }
            reasons.extend(
                f"{split}:{name}" for name, passed in checks.items() if not passed
            )
        rows.append(
            {
                "schema": SCHEMA_VERSION,
                "profile": "primary",
                "point_type": int(family.point_type),
                "horizon_sessions": int(family.horizon_sessions),
                "strict_gate_passed": not reasons,
                "failure_reasons": json.dumps(reasons, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def run_screen(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    maximum_cases: int | None = None,
    resume: bool = True,
    refresh_sample: bool = False,
) -> dict[str, Any]:
    study = _load_study(study_path)
    root = Path(output_root).resolve()
    frozen = sample.freeze_sample(
        spec_path=_sample_study_path(study),
        output_root=root / "sample",
        refresh_sample=refresh_sample,
    )
    cases = list(frozen["cases"])
    if maximum_cases is not None:
        cases = cases[: max(int(maximum_cases), 0)]
    if not cases:
        raise ValueError("strict_chan_outcome_screen_no_cases")
    by_period: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        by_period[_period_id(case)].append(case)
    partition_summaries = [
        _run_partition(
            period_id=period_id,
            cases=period_cases,
            study=study,
            selection_fingerprint=str(frozen["selection_fingerprint"]),
            output_root=root,
            resume=resume,
        )
        for period_id, period_cases in sorted(by_period.items())
    ]
    part_roots = [root / "partitions" / f"period={name}" for name in sorted(by_period)]
    candidates = pd.concat(
        [pd.read_parquet(path / "candidates.parquet") for path in part_roots],
        ignore_index=True,
    )
    outcomes = pd.concat(
        [pd.read_parquet(path / "event_outcomes.parquet") for path in part_roots],
        ignore_index=True,
    )
    baseline = pd.concat(
        [pd.read_parquet(path / "baseline.parquet") for path in part_roots],
        ignore_index=True,
    )
    aggregate = _aggregate(outcomes, baseline, study)
    gate = _gate(aggregate, study)
    _write_parquet(root / "candidates.parquet", candidates)
    _write_parquet(root / "event_outcomes.parquet", outcomes)
    _write_parquet(root / "baseline.parquet", baseline)
    _write_parquet(root / "aggregate.parquet", aggregate)
    _write_parquet(root / "strict_gate.parquet", gate)
    passed = gate[gate["strict_gate_passed"]]
    summary = {
        "schema": SCHEMA_VERSION,
        "status": "completed" if maximum_cases is None else "partial_smoke",
        "study_id": str(study["study_id"]),
        "sample_selection_fingerprint": frozen["selection_fingerprint"],
        "cases_configured": len(frozen["cases"]),
        "cases_completed": len(cases),
        "periods_completed": len(partition_summaries),
        "candidate_count": len(candidates),
        "outcome_rows": len(outcomes),
        "baseline_rows": len(baseline),
        "strict_gate_families": len(gate),
        "strict_gate_passed_count": len(passed),
        "strict_gate_passed": passed[["point_type", "horizon_sessions"]].to_dict(
            "records"
        ),
        "partition_summaries": partition_summaries,
        "profit_claim": False,
        "account_replay_allowed_by_result": bool(len(passed)) and maximum_cases is None,
        "paths": {
            "candidates": str((root / "candidates.parquet").resolve()),
            "event_outcomes": str((root / "event_outcomes.parquet").resolve()),
            "baseline": str((root / "baseline.parquet").resolve()),
            "aggregate": str((root / "aggregate.parquet").resolve()),
            "strict_gate": str((root / "strict_gate.parquet").resolve()),
        },
    }
    _write_json(root / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    argument_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    argument_parser.add_argument("--max-cases", type=int, default=None)
    argument_parser.add_argument("--no-resume", action="store_true")
    argument_parser.add_argument("--refresh-sample", action="store_true")
    return argument_parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_screen(
        study_path=args.study,
        output_root=args.output_root,
        maximum_cases=args.max_cases,
        resume=not args.no_resume,
        refresh_sample=args.refresh_sample,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
