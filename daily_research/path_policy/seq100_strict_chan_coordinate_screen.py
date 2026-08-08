"""Causal transparent-coordinate screen inside a frozen strict-Chan signal family."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_strict_chan_intraday as intraday

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/studies/seq100_strict_chan_coordinate_screen_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_strict_chan_coordinate_screen_v1"
)
STUDY_ID = "seq100_strict_chan_coordinate_screen_v1"
SCHEMA_VERSION = "seq100_strict_chan_coordinate_screen/1"


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
    if study.get("study_id") != STUDY_ID:
        raise ValueError("strict_chan_coordinate_screen_study_id_mismatch")
    if int(study["source"]["forbidden_year"]) != 2026:
        raise ValueError("strict_chan_coordinate_screen_forbidden_year_mismatch")
    directions = set(study["coordinates"].values())
    if not directions.issubset({"high", "low", "diagnostic"}):
        raise ValueError("strict_chan_coordinate_screen_direction_invalid")
    study["_study_path"] = str(study_path)
    return study


def _source_path(value: str) -> Path:
    return (WORKSPACE_ROOT / value).resolve()


def _load_signal_outcomes(
    root: Path, *, sample_id: str, study: Mapping[str, Any]
) -> pd.DataFrame:
    candidates = pd.read_parquet(root / "candidates.parquet")
    outcomes = pd.read_parquet(root / "event_outcomes.parquet")
    source = study["source"]
    selected_candidates = candidates[
        candidates["point_type"].eq(int(source["point_type"]))
        & candidates["variant"].eq(str(source["variant"]))
    ].copy()
    selected_candidates = selected_candidates.rename(columns={"id": "signal_id"})
    selected_outcomes = outcomes[
        outcomes["point_type"].eq(int(source["point_type"]))
        & outcomes["horizon_sessions"].eq(int(source["horizon_sessions"]))
    ].copy()
    metadata_columns = [
        "case_id",
        "signal_id",
        "confirmed_date",
        "variant",
        "level",
    ]
    joined = selected_outcomes.merge(
        selected_candidates.loc[:, metadata_columns],
        on=["case_id", "signal_id"],
        how="inner",
        validate="one_to_one",
    )
    joined["sample_id"] = sample_id
    joined["signal_date"] = joined["confirmed_date"].astype(str)
    joined["signal_key"] = (
        joined["sample_id"].astype(str)
        + "|"
        + joined["case_id"].astype(str)
        + "|"
        + joined["signal_id"].astype(str)
    )
    if joined["signal_key"].duplicated().any():
        raise ValueError("strict_chan_coordinate_screen_signal_duplicate")
    return joined


def _coordinate_sources(study: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = _source_path(str(study["source"]["coordinate_manifest"]))
    validation_path = _source_path(str(study["source"]["coordinate_validation"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or validation.get("status") != "passed":
        raise ValueError("strict_chan_coordinate_screen_source_not_validated")
    inherited = manifest["source_contract"]["inherited_source_contract"]
    row_index = Path(str(inherited["row_index"]["path"])).resolve()
    coordinate_paths = tuple(
        Path(str(item["path"])).resolve() for item in manifest["coordinates"]
    )
    if not row_index.is_file() or any(not path.is_file() for path in coordinate_paths):
        raise ValueError("strict_chan_coordinate_screen_source_file_missing")
    available = set(manifest["coordinate_audit"]["coordinate_names"])
    requested = set(study["coordinates"])
    if not requested.issubset(available):
        raise ValueError(
            f"strict_chan_coordinate_screen_coordinate_missing:{sorted(requested - available)}"
        )
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "validation_path": validation_path,
        "validation_sha256": _sha256_file(validation_path),
        "row_index": row_index,
        "coordinate_paths": coordinate_paths,
    }


def _join_coordinates(
    signals: pd.DataFrame,
    *,
    study: Mapping[str, Any],
    sources: Mapping[str, Any],
    temporary_root: Path,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    connection, runtime = intraday._connect(temporary_root)
    relation = "strict_chan_coordinate_signals"
    connection.register(relation, signals)
    try:
        row_scan = intraday._scan((Path(sources["row_index"]),))
        coordinate_scan = intraday._scan(tuple(sources["coordinate_paths"]))
        columns = ",\n".join(f'c."{name}" AS "{name}"' for name in study["coordinates"])
        joined = connection.execute(
            f"""
            SELECT s.*, r.date_idx AS coordinate_date_idx,
                   r.symbol_idx AS coordinate_symbol_idx,
                   {columns}
            FROM {relation} AS s
            LEFT JOIN {row_scan} AS r
              ON r.symbol = s.symbol AND r.trade_date = s.signal_date
            LEFT JOIN {coordinate_scan} AS c
              ON c.date_idx = r.date_idx AND c.symbol_idx = r.symbol_idx
            ORDER BY s.signal_key
            """
        ).fetchdf()
    finally:
        connection.unregister(relation)
        connection.close()
    if len(joined) != len(signals) or joined["signal_key"].duplicated().any():
        raise ValueError("strict_chan_coordinate_screen_join_cardinality")
    joined["coordinate_available"] = joined["coordinate_date_idx"].notna()
    return joined, runtime


def _favorable_score(values: pd.Series, direction: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if direction in {"high", "diagnostic"}:
        return numeric
    if direction == "low":
        return 1.0 - numeric
    raise ValueError(f"strict_chan_coordinate_screen_direction_unknown:{direction}")


def _add_scores(
    frame: pd.DataFrame, study: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, bool]]:
    output = frame.copy()
    gate_eligible: dict[str, bool] = {}
    for name, direction in study["coordinates"].items():
        score_name = f"score__{name}"
        output[score_name] = _favorable_score(output[name], str(direction))
        gate_eligible[name] = direction != "diagnostic"
    for composite, components in study["composites"].items():
        values: list[pd.Series] = []
        for token in components:
            name, direction = str(token).rsplit(":", 1)
            if name in study["coordinates"]:
                source_values = output[name]
            elif f"score__{name}" in output:
                source_values = output[f"score__{name}"]
            else:
                raise ValueError(
                    f"strict_chan_coordinate_screen_composite_source_missing:{name}"
                )
            values.append(_favorable_score(source_values, direction))
        output[f"score__{composite}"] = pd.concat(values, axis=1).mean(
            axis=1, skipna=False
        )
        gate_eligible[composite] = True
    return output, gate_eligible


def _evaluation_split(signal_date: str, study: Mapping[str, Any]) -> str:
    year = int(str(signal_date)[:4])
    evaluation = study["evaluation"]
    if year in set(map(int, evaluation["development_signal_years"])):
        return "development"
    if year in set(map(int, evaluation["validation_signal_years"])):
        return "validation"
    raise ValueError(f"strict_chan_coordinate_screen_year_unassigned:{year}")


def _bootstrap_lift(
    case_groups: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    values = case_groups.loc[:, ["favorable", "unfavorable"]].to_numpy(dtype=np.float64)
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    sampled = values[indices]
    with np.errstate(invalid="ignore"):
        favorable = np.nanmean(sampled[:, :, 0], axis=1)
        unfavorable = np.nanmean(sampled[:, :, 1], axis=1)
    lift = favorable - unfavorable
    lift = lift[np.isfinite(lift)]
    if not len(lift):
        return np.nan, np.nan
    return float(np.quantile(lift, 0.025)), float(np.quantile(lift, 0.975))


def _stable_seed(base: int, *values: Any) -> int:
    token = "|".join(map(str, (base, *values)))
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)


def _contrast_rows(
    frame: pd.DataFrame,
    *,
    study: Mapping[str, Any],
    score_names: Sequence[str],
) -> pd.DataFrame:
    evaluation = study["evaluation"]
    threshold = float(evaluation["favorable_threshold"])
    resamples = int(evaluation["bootstrap_case_resamples"])
    base_seed = int(evaluation["bootstrap_seed"])
    rows: list[dict[str, Any]] = []
    for split in ("development", "validation", "all"):
        selected = (
            frame if split == "all" else frame[frame["evaluation_split"].eq(split)]
        )
        for name in score_names:
            score_column = f"score__{name}"
            usable = selected[selected[score_column].notna()].copy()
            usable["group"] = np.where(
                usable[score_column].ge(threshold), "favorable", "unfavorable"
            )
            favorable = usable[usable["group"].eq("favorable")]
            unfavorable = usable[usable["group"].eq("unfavorable")]
            case_groups = (
                usable.groupby(["case_id", "group"])["net_return_base"]
                .mean()
                .unstack()
                .reindex(columns=["favorable", "unfavorable"])
            )
            low, high = _bootstrap_lift(
                case_groups,
                resamples=resamples,
                seed=_stable_seed(base_seed, split, name),
            )
            favorable_case = favorable.groupby("case_id")["net_return_base"].mean()
            favorable_stress_case = favorable.groupby("case_id")[
                "net_return_stress"
            ].mean()
            unfavorable_case = unfavorable.groupby("case_id")["net_return_base"].mean()
            rows.append(
                {
                    "schema": SCHEMA_VERSION,
                    "evaluation_split": split,
                    "coordinate": name,
                    "favorable_event_count": len(favorable),
                    "unfavorable_event_count": len(unfavorable),
                    "favorable_case_count": len(favorable_case),
                    "unfavorable_case_count": len(unfavorable_case),
                    "favorable_event_median_net_base": float(
                        favorable["net_return_base"].median()
                    ),
                    "favorable_case_mean_net_base": float(favorable_case.mean()),
                    "favorable_case_mean_net_stress": float(
                        favorable_stress_case.mean()
                    ),
                    "unfavorable_case_mean_net_base": float(unfavorable_case.mean()),
                    "favorable_minus_unfavorable_base": float(
                        favorable_case.mean() - unfavorable_case.mean()
                    ),
                    "bootstrap_lift_low": low,
                    "bootstrap_lift_high": high,
                }
            )
    return pd.DataFrame(rows)


def _gate(
    contrasts: pd.DataFrame,
    *,
    study: Mapping[str, Any],
    gate_eligible: Mapping[str, bool],
) -> pd.DataFrame:
    evaluation = study["evaluation"]
    rows: list[dict[str, Any]] = []
    for coordinate in sorted(gate_eligible):
        reasons: list[str] = []
        if not gate_eligible[coordinate]:
            reasons.append("diagnostic_only")
        for split, minimum_events, minimum_cases in (
            (
                "development",
                int(evaluation["minimum_favorable_events_development"]),
                int(evaluation["minimum_favorable_cases_development"]),
            ),
            (
                "validation",
                int(evaluation["minimum_favorable_events_validation"]),
                int(evaluation["minimum_favorable_cases_validation"]),
            ),
        ):
            selected = contrasts[
                contrasts["evaluation_split"].eq(split)
                & contrasts["coordinate"].eq(coordinate)
            ]
            if selected.empty:
                reasons.append(f"{split}:missing")
                continue
            item = selected.iloc[0]
            checks = {
                "events": int(item["favorable_event_count"]) >= minimum_events,
                "cases": int(item["favorable_case_count"]) >= minimum_cases,
                "net_base": float(item["favorable_case_mean_net_base"]) > 0,
                "net_stress": float(item["favorable_case_mean_net_stress"]) > 0,
                "lift": float(item["favorable_minus_unfavorable_base"]) > 0,
                "lift_ci": float(item["bootstrap_lift_low"]) > 0,
            }
            reasons.extend(
                f"{split}:{name}" for name, passed in checks.items() if not passed
            )
        rows.append(
            {
                "schema": SCHEMA_VERSION,
                "coordinate": coordinate,
                "strict_gate_passed": not reasons,
                "failure_reasons": json.dumps(reasons, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def run_screen(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = _load_study(study_path)
    root = Path(output_root).resolve()
    exploratory = _load_signal_outcomes(
        _source_path(str(study["source"]["exploratory_outcome_root"])),
        sample_id="exploratory",
        study=study,
    )
    confirmation = _load_signal_outcomes(
        _source_path(str(study["source"]["confirmation_outcome_root"])),
        sample_id="confirmation",
        study=study,
    )
    signals = pd.concat([exploratory, confirmation], ignore_index=True)
    if signals["signal_date"].ge("2026-01-01").any():
        raise ValueError("strict_chan_coordinate_screen_forbidden_signal")
    sources = _coordinate_sources(study)
    joined, runtime = _join_coordinates(
        signals,
        study=study,
        sources=sources,
        temporary_root=root / "duckdb_tmp",
    )
    scored, gate_eligible = _add_scores(joined, study)
    scored["evaluation_split"] = scored["signal_date"].map(
        lambda value: _evaluation_split(str(value), study)
    )
    score_names = [*study["coordinates"], *study["composites"]]
    contrasts = _contrast_rows(scored, study=study, score_names=score_names)
    gate = _gate(contrasts, study=study, gate_eligible=gate_eligible)
    _write_parquet(root / "signal_coordinates.parquet", scored)
    _write_parquet(root / "coordinate_contrasts.parquet", contrasts)
    _write_parquet(root / "strict_gate.parquet", gate)
    passed = gate[gate["strict_gate_passed"]]
    matched = scored[scored["coordinate_available"]]
    unmatched = scored[~scored["coordinate_available"]]
    summary = {
        "schema": SCHEMA_VERSION,
        "status": "completed",
        "study_id": STUDY_ID,
        "signals": len(scored),
        "exploratory_signals": int(scored["sample_id"].eq("exploratory").sum()),
        "confirmation_signals": int(scored["sample_id"].eq("confirmation").sum()),
        "coordinate_matched_signals": len(matched),
        "coordinate_unmatched_signals": len(unmatched),
        "coordinate_match_fraction": float(scored["coordinate_available"].mean()),
        "matched_mean_net_base": float(matched["net_return_base"].mean()),
        "unmatched_mean_net_base": float(unmatched["net_return_base"].mean()),
        "coordinates": len(study["coordinates"]),
        "composites": len(study["composites"]),
        "strict_gate_passed_count": len(passed),
        "strict_gate_passed": passed["coordinate"].astype(str).tolist(),
        "account_replay_allowed_by_result": False,
        "profit_claim": False,
        "runtime": runtime,
        "source": {
            "study_sha256": _sha256_file(str(study["_study_path"])),
            "coordinate_manifest": str(sources["manifest_path"]),
            "coordinate_manifest_sha256": sources["manifest_sha256"],
            "coordinate_validation": str(sources["validation_path"]),
            "coordinate_validation_sha256": sources["validation_sha256"],
        },
        "paths": {
            "signal_coordinates": str((root / "signal_coordinates.parquet").resolve()),
            "coordinate_contrasts": str(
                (root / "coordinate_contrasts.parquet").resolve()
            ),
            "strict_gate": str((root / "strict_gate.parquet").resolve()),
        },
    }
    _write_json(root / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    argument_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return argument_parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_screen(study_path=args.study, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
