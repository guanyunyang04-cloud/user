"""Validate persisted outputs of the strict-Chan structure increment study."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from daily_research.path_policy import seq100_strict_chan_structure_increment as study

DEFAULT_OUTPUT_ROOT = study.DEFAULT_OUTPUT_ROOT


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    root = study._resolve(output_root)
    summary = _read_json(root / "summary.json")
    config = study.load_study(summary["study"]["path"])
    frames: dict[str, pd.DataFrame] = {}
    output_evidence: dict[str, Any] = {}
    for name, record in summary["outputs"].items():
        path = Path(str(record["path"]))
        _require(path.is_file(), f"strict_chan_structure_validate_missing:{name}")
        _require(
            study._sha256(path) == str(record["sha256"]),
            f"strict_chan_structure_validate_hash:{name}",
        )
        frame = pd.read_parquet(path)
        _require(
            len(frame) == int(record["rows"]),
            f"strict_chan_structure_validate_rows:{name}",
        )
        frames[name] = frame
        output_evidence[name] = {
            "path": str(path.resolve()),
            "sha256": str(record["sha256"]),
            "rows": len(frame),
        }
    selected = pd.read_parquet(root / "sample" / "selected_cases.parquet")
    _require(len(selected) == 700, "strict_chan_structure_validate_sample_count")
    _require(
        selected["symbol"].nunique() == len(selected),
        "strict_chan_structure_validate_sample_unique",
    )
    prior_symbols: set[str] = set()
    sample_spec, _ = study._sample_spec(config)
    for value in sample_spec["selection"]["exclude_selected_cases"]:
        prior = pd.read_parquet(study._resolve(value), columns=["symbol"])
        prior_symbols.update(prior["symbol"].astype(str))
    overlap = set(selected["symbol"].astype(str)) & prior_symbols
    _require(not overlap, "strict_chan_structure_validate_prior_overlap")

    signals = frames["signals"]
    daily = frames["daily_paths"]
    pairs = frames["matched_pairs"]
    predictions = frames["neighbor_predictions"]
    _require(
        not signals["signal_key"].duplicated().any(),
        "strict_chan_structure_validate_signal_duplicate",
    )
    _require(
        set(signals["point_type"].astype(int)) == {1, 2, 3},
        "strict_chan_structure_validate_point_types",
    )
    _require(
        signals["signal_date"].astype(str).between("2012-01-01", "2025-12-31").all(),
        "strict_chan_structure_validate_signal_dates",
    )
    _require(
        signals["exit_date"].astype(str).le("2025-12-31").all(),
        "strict_chan_structure_validate_signal_exit",
    )
    _require(
        not daily["daily_key"].duplicated().any(),
        "strict_chan_structure_validate_daily_duplicate",
    )
    _require(
        daily["exit_date"].astype(str).le("2025-12-31").all(),
        "strict_chan_structure_validate_daily_exit",
    )
    _require(
        not pairs["signal_key"].duplicated().any(),
        "strict_chan_structure_validate_pair_signal_duplicate",
    )
    _require(
        not pairs.duplicated(["signal_date", "control_key"]).any(),
        "strict_chan_structure_validate_control_reuse",
    )
    _require(
        pairs["symbol"].astype(str).ne(pairs["control_symbol"].astype(str)).all(),
        "strict_chan_structure_validate_same_symbol_control",
    )
    _require(
        pairs["match_distance"].le(float(config["matching"]["maximum_distance"])).all(),
        "strict_chan_structure_validate_match_distance",
    )
    pattern_keys = set(
        signals["case_id"].astype(str)
        + "|"
        + signals["episode_id"].astype(str)
        + "|"
        + signals["signal_date"].astype(str)
    )
    _require(
        not (set(pairs["control_key"].astype(str)) & pattern_keys),
        "strict_chan_structure_validate_pattern_control",
    )
    _require(
        predictions["latest_historical_exit_date"]
        .astype(str)
        .lt(predictions["signal_date"].astype(str))
        .all(),
        "strict_chan_structure_validate_unresolved_history",
    )
    _require(
        not predictions.duplicated(["signal_key", "representation", "target"]).any(),
        "strict_chan_structure_validate_prediction_duplicate",
    )
    _require(
        set(predictions["representation"].astype(str))
        == {"coordinates", "structure", "combined"},
        "strict_chan_structure_validate_representations",
    )
    _require(
        len(signals) == int(summary["signal_count"])
        and len(daily) == int(summary["daily_path_count"])
        and len(pairs) == int(summary["matched_pair_count"])
        and len(predictions) == int(summary["prediction_rows"]),
        "strict_chan_structure_validate_summary_counts",
    )
    _require(
        not bool(summary["account_replay_authorized"]),
        "strict_chan_structure_validate_account_boundary",
    )
    result: dict[str, Any] = {
        "schema": "seq100_strict_chan_structure_increment_validate/1",
        "status": "passed",
        "study_id": summary["study_id"],
        "sample_cases": len(selected),
        "sample_unique_symbols": int(selected["symbol"].nunique()),
        "prior_sample_overlap": len(overlap),
        "signal_count": len(signals),
        "point_type_counts": (
            signals.groupby("point_type", sort=True).size().to_dict()
        ),
        "daily_path_count": len(daily),
        "matched_pair_count": len(pairs),
        "prediction_rows": len(predictions),
        "unresolved_history_violations": 0,
        "future_date_violations": 0,
        "account_replay_authorized": False,
        "outputs": output_evidence,
    }
    study._write_json(root / "validation.json", result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = validate(output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
