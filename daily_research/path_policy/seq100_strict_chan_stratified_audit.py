"""Freeze and run an outcome-blind stratified audit of the strict Chan parser."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from daily_research.path_policy import seq100_strict_chan_audit as audit
from daily_research.path_policy import seq100_strict_chan_parser as parser

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRATIFIED_AUDIT_PATH = (
    WORKSPACE_ROOT
    / "daily_research"
    / "studies"
    / "seq100_strict_chan_stratified_audit_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research"
    / "output"
    / "path_policy"
    / "studies"
    / "seq100_strict_chan_stratified_audit_v1"
)
STUDY_ID = "seq100_strict_chan_stratified_audit_v1"
SCHEMA_VERSION = "seq100_strict_chan_stratified_audit/1"
SELECTION_COLUMNS = ("symbol", "trade_date")
_REQUIRED_STRATUM_FIELDS = ("stratum_id", "years", "symbol_suffix", "samples")


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


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _stable_rank(
    *,
    study_id: str,
    seed: int,
    stratum_id: str,
    symbol: str,
    trade_date: str,
) -> str:
    value = "|".join(
        (
            study_id,
            str(seed),
            stratum_id,
            symbol,
            trade_date,
        )
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_stratified_spec(
    path: str | Path = DEFAULT_STRATIFIED_AUDIT_PATH,
) -> dict[str, Any]:
    """Load and validate the frozen, outcome-blind selection contract."""

    spec_path = Path(path).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("study_id") != STUDY_ID:
        raise ValueError("strict_chan_stratified_audit_study_id_mismatch")
    if spec.get("parent_study_id") != parser.STUDY_ID:
        raise ValueError("strict_chan_stratified_audit_parent_study_mismatch")
    boundaries = dict(spec.get("boundaries", {}))
    for key in (
        "future_path_test_performed",
        "return_test_performed",
        "profit_claim_allowed",
        "account_replay_allowed",
        "model_training_allowed",
    ):
        if bool(boundaries.get(key)):
            raise ValueError(f"strict_chan_stratified_audit_boundary_violation:{key}")

    selection = dict(spec.get("selection", {}))
    if selection.get("algorithm") != "sha256_lexicographic_row_rank":
        raise ValueError("strict_chan_stratified_audit_selection_algorithm_invalid")
    if tuple(selection.get("selection_columns", ())) != SELECTION_COLUMNS:
        raise ValueError("strict_chan_stratified_audit_selection_columns_invalid")
    suffix_contract = tuple(selection.get("population_symbol_suffix_contract", ()))
    if not suffix_contract or any(
        not str(value).startswith(".") for value in suffix_contract
    ):
        raise ValueError("strict_chan_stratified_audit_suffix_contract_invalid")
    if not bool(selection.get("globally_unique_symbols")):
        raise ValueError("strict_chan_stratified_audit_unique_symbol_contract_missing")
    bounds = dict(selection.get("focal_date_bounds", {}))
    earliest = str(bounds.get("earliest", ""))
    latest = str(bounds.get("latest", ""))
    if not earliest or not latest or earliest > latest:
        raise ValueError("strict_chan_stratified_audit_focal_bounds_invalid")
    if earliest < "2012-01-01" or latest > "2025-12-31":
        raise ValueError("strict_chan_stratified_audit_focal_bounds_outside_contract")
    window = dict(selection.get("window", {}))
    for name in (
        "parse_lookback_calendar_days",
        "parse_lookahead_calendar_days",
        "display_before_calendar_days",
        "display_after_calendar_days",
    ):
        if int(window.get(name, 0)) < 0:
            raise ValueError(f"strict_chan_stratified_audit_window_invalid:{name}")

    strata = list(selection.get("strata", []))
    stratum_ids = [str(item.get("stratum_id", "")) for item in strata]
    if not strata or len(stratum_ids) != len(set(stratum_ids)):
        raise ValueError("strict_chan_stratified_audit_strata_invalid")
    for stratum in strata:
        if any(field not in stratum for field in _REQUIRED_STRATUM_FIELDS):
            raise ValueError(
                f"strict_chan_stratified_audit_stratum_missing_field:{stratum.get('stratum_id', '')}"
            )
        years = [int(year) for year in stratum["years"]]
        if (
            not years
            or years != sorted(set(years))
            or any(year < 2012 or year > 2025 for year in years)
        ):
            raise ValueError(
                f"strict_chan_stratified_audit_stratum_years_invalid:{stratum['stratum_id']}"
            )
        suffix = str(stratum["symbol_suffix"])
        if not suffix.startswith(".") or len(suffix) < 3:
            raise ValueError(
                f"strict_chan_stratified_audit_stratum_suffix_invalid:{stratum['stratum_id']}"
            )
        if int(stratum["samples"]) <= 0:
            raise ValueError(
                f"strict_chan_stratified_audit_stratum_samples_invalid:{stratum['stratum_id']}"
            )

    pool_manifest = WORKSPACE_ROOT / str(spec["quality_pool_manifest"])
    if not pool_manifest.is_file():
        raise ValueError("strict_chan_stratified_audit_quality_pool_manifest_missing")
    spec["_spec_path"] = str(spec_path)
    spec["_quality_pool_manifest_path"] = str(pool_manifest.resolve())
    return spec


def _case_date(focal: pd.Timestamp, days: int) -> str:
    return (focal + pd.Timedelta(days=days)).strftime("%Y-%m-%d")


def _make_case(
    *,
    spec: Mapping[str, Any],
    stratum: Mapping[str, Any],
    sample_index: int,
    symbol: str,
    trade_date: str,
    rank: str,
) -> dict[str, Any]:
    selection = spec["selection"]
    window = selection["window"]
    focal = pd.Timestamp(trade_date)
    case_id = f"stratified_{stratum['stratum_id']}_{sample_index:02d}"
    return {
        "case_id": case_id,
        "category": "stratified_quality_pool",
        "stratum_id": str(stratum["stratum_id"]),
        "stratum_years": [int(year) for year in stratum["years"]],
        "symbol_suffix": str(stratum["symbol_suffix"]),
        "sample_index": sample_index,
        "selection_rank": rank,
        "symbol": symbol,
        "start_date": _case_date(focal, -int(window["parse_lookback_calendar_days"])),
        "end_date": _case_date(focal, int(window["parse_lookahead_calendar_days"])),
        "display_start_date": _case_date(
            focal, -int(window["display_before_calendar_days"])
        ),
        "display_end_date": _case_date(
            focal, int(window["display_after_calendar_days"])
        ),
        "focal_date": trade_date,
        "quality_pool_expected": True,
        "selection_basis": (
            "Outcome-blind SHA-256 row rank over quality_liquidity_pit symbol and "
            "trade_date only; no future return, label, parser event, or trade result "
            f"was read for stratum {stratum['stratum_id']}."
        ),
        "assertions": [],
    }


def select_cases_from_frames(
    spec: Mapping[str, Any],
    frames: Mapping[int, pd.DataFrame],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select frozen cases from already column-restricted pool frames.

    This pure selection step deliberately accepts only ``symbol`` and
    ``trade_date`` semantics. Callers should read no outcome or parser columns.
    """

    selection = spec["selection"]
    seed = int(selection["random_seed"])
    bounds = selection["focal_date_bounds"]
    earliest = pd.Timestamp(str(bounds["earliest"]))
    latest = pd.Timestamp(str(bounds["latest"]))
    unique_symbols = bool(selection["globally_unique_symbols"])
    selected_symbols: set[str] = set()
    cases: list[dict[str, Any]] = []
    statistics: list[dict[str, Any]] = []

    for stratum in selection["strata"]:
        stratum_id = str(stratum["stratum_id"])
        years = [int(year) for year in stratum["years"]]
        missing_years = [year for year in years if year not in frames]
        if missing_years:
            raise ValueError(
                f"strict_chan_stratified_audit_missing_year_frames:{stratum_id}"
            )
        frame = pd.concat(
            [frames[year].loc[:, list(SELECTION_COLUMNS)] for year in years],
            ignore_index=True,
        )
        if tuple(frame.columns) != SELECTION_COLUMNS:
            raise ValueError("strict_chan_stratified_audit_frame_columns_invalid")
        frame["symbol"] = frame["symbol"].astype(str)
        frame["trade_date"] = frame["trade_date"].astype(str)
        dates = pd.to_datetime(frame["trade_date"], format="%Y-%m-%d", errors="raise")
        eligible = frame.loc[
            dates.between(earliest, latest)
            & frame["symbol"].str.endswith(str(stratum["symbol_suffix"]))
        ].drop_duplicates(list(SELECTION_COLUMNS))
        ranked = sorted(
            (
                _stable_rank(
                    study_id=STUDY_ID,
                    seed=seed,
                    stratum_id=stratum_id,
                    symbol=str(row.symbol),
                    trade_date=str(row.trade_date),
                ),
                str(row.symbol),
                str(row.trade_date),
            )
            for row in eligible.itertuples(index=False)
        )
        requested = int(stratum["samples"])
        chosen: list[tuple[str, str, str]] = []
        skipped_duplicate_symbols = 0
        for rank, symbol, trade_date in ranked:
            if unique_symbols and symbol in selected_symbols:
                skipped_duplicate_symbols += 1
                continue
            chosen.append((rank, symbol, trade_date))
            selected_symbols.add(symbol)
            if len(chosen) == requested:
                break
        if len(chosen) != requested:
            raise ValueError(
                "strict_chan_stratified_audit_insufficient_candidates:" + stratum_id
            )
        for sample_index, (rank, symbol, trade_date) in enumerate(chosen, start=1):
            cases.append(
                _make_case(
                    spec=spec,
                    stratum=stratum,
                    sample_index=sample_index,
                    symbol=symbol,
                    trade_date=trade_date,
                    rank=rank,
                )
            )
        statistics.append(
            {
                "stratum_id": stratum_id,
                "years": years,
                "symbol_suffix": str(stratum["symbol_suffix"]),
                "candidate_rows": len(frame),
                "eligible_rows": len(eligible),
                "requested_samples": requested,
                "selected_samples": len(chosen),
                "skipped_duplicate_symbols": skipped_duplicate_symbols,
            }
        )
    return cases, statistics


def _load_pool_frames(
    spec: Mapping[str, Any],
) -> tuple[dict[int, pd.DataFrame], list[dict[str, Any]], str]:
    manifest_path = Path(str(spec["_quality_pool_manifest_path"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requested_years = sorted(
        {
            int(year)
            for stratum in spec["selection"]["strata"]
            for year in stratum["years"]
        }
    )
    partitions = {int(item["year"]): item for item in manifest["partitions"]}
    frames: dict[int, pd.DataFrame] = {}
    evidence: list[dict[str, Any]] = []
    for year in requested_years:
        if year not in partitions:
            raise ValueError(f"strict_chan_stratified_audit_pool_year_missing:{year}")
        item = partitions[year]
        path = Path(str(item["path"]))
        declared_sha256 = str(item["sha256"])
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != declared_sha256:
            raise ValueError(
                f"strict_chan_stratified_audit_partition_hash_mismatch:{year}"
            )
        frame = pd.read_parquet(path, columns=list(SELECTION_COLUMNS))
        if tuple(frame.columns) != SELECTION_COLUMNS:
            raise ValueError("strict_chan_stratified_audit_pool_columns_invalid")
        frames[year] = frame
        evidence.append(
            {
                "year": year,
                "path": str(path.resolve()),
                "sha256": declared_sha256,
                "actual_sha256": actual_sha256,
                "row_count": int(item["row_count"]),
                "rows_read": len(frame),
            }
        )
    return frames, evidence, _sha256_file(manifest_path)


def _symbol_suffix_statistics(
    frames: Mapping[int, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    combined = pd.concat(
        [frame.loc[:, ["symbol"]] for frame in frames.values()], ignore_index=True
    )
    suffixes = combined["symbol"].astype(str).str.rsplit(".", n=1).str[-1].radd(".")
    for suffix, values in suffixes.groupby(suffixes, sort=True):
        rows.append(
            {
                "symbol_suffix": str(suffix),
                "rows": len(values),
                "unique_symbols": int(combined.loc[values.index, "symbol"].nunique()),
            }
        )
    return rows


def _selection_fingerprint(
    spec: Mapping[str, Any],
    *,
    source_partitions: Sequence[Mapping[str, Any]],
    manifest_sha256: str,
    cases: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "schema": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "spec_sha256": _sha256_file(str(spec["_spec_path"])),
        "runner_sha256": _sha256_file(Path(__file__)),
        "definition_sha256": _sha256_file(parser.DEFAULT_DEFINITION_PATH),
        "quality_pool_manifest_sha256": manifest_sha256,
        "source_partitions": list(source_partitions),
        "cases": list(cases),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_sample_manifest(
    spec_path: str | Path = DEFAULT_STRATIFIED_AUDIT_PATH,
) -> dict[str, Any]:
    spec = load_stratified_spec(spec_path)
    frames, source_partitions, manifest_sha256 = _load_pool_frames(spec)
    suffix_statistics = _symbol_suffix_statistics(frames)
    observed_suffixes = {item["symbol_suffix"] for item in suffix_statistics}
    configured_suffixes = {
        str(value) for value in spec["selection"]["population_symbol_suffix_contract"]
    }
    if not observed_suffixes.issubset(configured_suffixes):
        raise ValueError(
            "strict_chan_stratified_audit_unconfigured_symbol_suffix:"
            + ",".join(sorted(observed_suffixes - configured_suffixes))
        )
    cases, statistics = select_cases_from_frames(spec, frames)
    fingerprint = _selection_fingerprint(
        spec,
        source_partitions=source_partitions,
        manifest_sha256=manifest_sha256,
        cases=cases,
    )
    for case in cases:
        case["selection_fingerprint"] = fingerprint
    return {
        "schema": SCHEMA_VERSION,
        "status": "frozen_sample",
        "study_id": STUDY_ID,
        "parent_study_id": spec["parent_study_id"],
        "spec_path": str(Path(str(spec["_spec_path"])).resolve()),
        "spec_sha256": _sha256_file(str(spec["_spec_path"])),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": _sha256_file(Path(__file__)),
        "quality_pool_manifest_path": str(
            Path(str(spec["_quality_pool_manifest_path"])).resolve()
        ),
        "quality_pool_manifest_sha256": manifest_sha256,
        "selection_columns": list(SELECTION_COLUMNS),
        "outcome_blind_selection": True,
        "parser_outcomes_used_for_selection": False,
        "returns_used_for_selection": False,
        "source_partitions": list(source_partitions),
        "population_symbol_suffix_contract": sorted(configured_suffixes),
        "population_symbol_suffix_statistics": suffix_statistics,
        "strata_statistics": statistics,
        "cases_configured": len(cases),
        "selection_fingerprint": fingerprint,
        "cases": cases,
    }


def freeze_sample(
    *,
    spec_path: str | Path = DEFAULT_STRATIFIED_AUDIT_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    refresh_sample: bool = False,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "sample_manifest.json"
    candidate = build_sample_manifest(spec_path)
    if manifest_path.is_file() and not refresh_sample:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("selection_fingerprint") != candidate["selection_fingerprint"]:
            raise ValueError("strict_chan_stratified_audit_frozen_sample_mismatch")
        selected_path = root / "selected_cases.parquet"
        if not selected_path.is_file():
            _write_parquet(
                selected_path,
                pd.DataFrame(
                    [
                        {
                            "case_id": case["case_id"],
                            "stratum_id": case["stratum_id"],
                            "sample_index": case["sample_index"],
                            "symbol": case["symbol"],
                            "focal_date": case["focal_date"],
                            "selection_rank": case["selection_rank"],
                            "selection_fingerprint": case["selection_fingerprint"],
                        }
                        for case in existing["cases"]
                    ]
                ),
            )
        return existing
    _write_json(manifest_path, candidate)
    selected_frame = pd.DataFrame(
        [
            {
                "case_id": case["case_id"],
                "stratum_id": case["stratum_id"],
                "sample_index": case["sample_index"],
                "symbol": case["symbol"],
                "focal_date": case["focal_date"],
                "selection_rank": case["selection_rank"],
                "selection_fingerprint": case["selection_fingerprint"],
            }
            for case in candidate["cases"]
        ]
    )
    _write_parquet(root / "selected_cases.parquet", selected_frame)
    return candidate


def _materialized_audit_spec(
    spec: Mapping[str, Any],
    sample: Mapping[str, Any],
    *,
    sample_manifest_path: Path,
) -> dict[str, Any]:
    return {
        "study_id": STUDY_ID,
        "status": "frozen_materialized_sample",
        "parent_study_id": spec["parent_study_id"],
        "quality_pool_manifest": spec["quality_pool_manifest"],
        "selection_manifest": str(sample_manifest_path.resolve()),
        "selection_fingerprint": sample["selection_fingerprint"],
        "boundaries": dict(spec["boundaries"]),
        "profiles": dict(spec["profiles"]),
        "prefix_validation": dict(spec["prefix_validation"]),
        "visualization": dict(spec["visualization"]),
        "cases": list(sample["cases"]),
    }


def run_stratified_audit(
    *,
    stratified_audit_path: str | Path = DEFAULT_STRATIFIED_AUDIT_PATH,
    definition_path: str | Path = parser.DEFAULT_DEFINITION_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    case_ids: Sequence[str] | None = None,
    run_prefix_validation: bool = True,
    render_charts: bool = True,
    resume: bool = True,
    refresh_sample: bool = False,
) -> dict[str, Any]:
    spec = load_stratified_spec(stratified_audit_path)
    root = Path(output_root).resolve()
    sample = freeze_sample(
        spec_path=stratified_audit_path,
        output_root=root,
        refresh_sample=refresh_sample,
    )
    sample_manifest_path = root / "sample_manifest.json"
    frozen_audit_path = root / "frozen_audit_spec.json"
    frozen_spec = _materialized_audit_spec(
        spec,
        sample,
        sample_manifest_path=sample_manifest_path,
    )
    _write_json(frozen_audit_path, frozen_spec)
    summary = audit.run_audit(
        audit_path=frozen_audit_path,
        definition_path=definition_path,
        output_root=root,
        case_ids=case_ids,
        run_prefix_validation=run_prefix_validation,
        render_charts=render_charts,
        resume=resume,
    )
    summary.update(
        {
            "selection_manifest_path": str(sample_manifest_path.resolve()),
            "selected_cases_path": str((root / "selected_cases.parquet").resolve()),
            "frozen_audit_spec_path": str(frozen_audit_path.resolve()),
            "selection_fingerprint": sample["selection_fingerprint"],
            "selection_columns": list(SELECTION_COLUMNS),
            "outcome_blind_selection": True,
            "parser_outcomes_used_for_selection": False,
            "returns_used_for_selection": False,
            "strata_configured": len(spec["selection"]["strata"]),
            "strata_statistics": sample["strata_statistics"],
        }
    )
    _write_json(root / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Run the frozen outcome-blind stratified strict Chan audit."
    )
    argument_parser.add_argument(
        "--stratified-audit", default=str(DEFAULT_STRATIFIED_AUDIT_PATH)
    )
    argument_parser.add_argument(
        "--definition", default=str(parser.DEFAULT_DEFINITION_PATH)
    )
    argument_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    argument_parser.add_argument("--case-id", action="append", default=[])
    argument_parser.add_argument("--skip-prefix-validation", action="store_true")
    argument_parser.add_argument("--skip-charts", action="store_true")
    argument_parser.add_argument("--no-resume", action="store_true")
    argument_parser.add_argument("--refresh-sample", action="store_true")
    return argument_parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_stratified_audit(
        stratified_audit_path=args.stratified_audit,
        definition_path=args.definition,
        output_root=args.output_root,
        case_ids=args.case_id,
        run_prefix_validation=not args.skip_prefix_validation,
        render_charts=not args.skip_charts,
        resume=not args.no_resume,
        refresh_sample=args.refresh_sample,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
