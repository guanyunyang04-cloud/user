"""Independent prefix-invariance checks for the strict Chan parser."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_strict_chan_parser as parser

VALIDATION_SCHEMA_VERSION = "seq100_strict_chan_validation/1"


def _canonical_record(record: Mapping[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, ensure_ascii=False, default=str)


def _records_before(
    result: parser.ChanParseResult, cutoff: int | None = None
) -> list[str]:
    records = result.event_records()
    if cutoff is not None:
        records = [
            record for record in records if int(record["confirmed_index"]) < cutoff
        ]
    return [_canonical_record(record) for record in records]


def verify_prefix_invariance(
    frame: pd.DataFrame,
    *,
    cutoffs: Sequence[int] | None = None,
    definition_path: str | Path = parser.DEFAULT_DEFINITION_PATH,
    variant_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    length = len(frame)
    if length == 0:
        raise ValueError("strict_chan_prefix_empty_frame")
    if cutoffs is None:
        cutoffs = np.unique(
            np.linspace(1, length, num=min(12, length), dtype=np.int64)
        ).tolist()
    normalized_cutoffs = sorted({int(value) for value in cutoffs})
    if (
        not normalized_cutoffs
        or normalized_cutoffs[0] < 1
        or normalized_cutoffs[-1] > length
    ):
        raise ValueError("strict_chan_prefix_cutoff_out_of_range")
    full = parser.parse_strict_chan(
        frame,
        definition_path=definition_path,
        variant_overrides=variant_overrides,
    )
    compared_records = 0
    compared_fields = 0
    cutoff_summaries: list[dict[str, Any]] = []
    for cutoff in normalized_cutoffs:
        prefix = parser.parse_strict_chan(
            frame.iloc[:cutoff].reset_index(drop=True),
            definition_path=definition_path,
            variant_overrides=variant_overrides,
        )
        actual = _records_before(prefix)
        expected = _records_before(full, cutoff)
        if actual != expected:
            actual_set = set(actual)
            expected_set = set(expected)
            missing = [
                json.loads(item) for item in sorted(expected_set - actual_set)[:5]
            ]
            extra = [json.loads(item) for item in sorted(actual_set - expected_set)[:5]]
            raise ValueError(
                "strict_chan_prefix_mismatch:"
                + json.dumps(
                    {
                        "cutoff": cutoff,
                        "missing": missing,
                        "extra": extra,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            )
        compared_records += len(actual)
        compared_fields += sum(len(json.loads(item)) for item in actual)
        cutoff_summaries.append(
            {
                "cutoff": cutoff,
                "events": len(actual),
                "summary": prefix.summary(),
            }
        )
    return {
        "schema": VALIDATION_SCHEMA_VERSION,
        "status": "passed",
        "bars": length,
        "cutoffs": len(normalized_cutoffs),
        "event_records_compared": compared_records,
        "event_fields_compared": compared_fields,
        "future_confirmed_structure_written_back": False,
        "variant_overrides": dict(variant_overrides or {}),
        "full_summary": full.summary(),
        "cutoff_summaries": cutoff_summaries,
    }


def validate_symbol_episodes(
    symbol: str,
    *,
    start_date: str,
    end_date: str,
    definition_path: str | Path = parser.DEFAULT_DEFINITION_PATH,
    output_path: str | Path | None = None,
    variant_overrides: Mapping[str, str] | None = None,
    maximum_episodes: int = 3,
    cutoffs_per_episode: int = 8,
) -> dict[str, Any]:
    from daily_research.path_policy import seq100_strict_chan_intraday as intraday

    loaded = intraday.load_symbol_episodes(
        symbol,
        start_date=start_date,
        end_date=end_date,
        definition_path=definition_path,
    )
    episode_sizes = (
        loaded.frame.groupby("episode_id", sort=True)
        .size()
        .sort_values(ascending=False)
    )
    selected_ids = [int(item) for item in episode_sizes.index[:maximum_episodes]]
    validations: list[dict[str, Any]] = []
    for episode_id in selected_ids:
        frame = loaded.frame[loaded.frame["episode_id"].eq(episode_id)]
        parse_frame = frame.loc[:, parser.REQUIRED_INPUT_COLUMNS].reset_index(drop=True)
        cutoffs = np.unique(
            np.linspace(
                1,
                len(parse_frame),
                num=min(cutoffs_per_episode, len(parse_frame)),
                dtype=np.int64,
            )
        ).tolist()
        validation = verify_prefix_invariance(
            parse_frame,
            cutoffs=cutoffs,
            definition_path=definition_path,
            variant_overrides=variant_overrides,
        )
        validation["episode_id"] = episode_id
        validation["start_time"] = pd.Timestamp(
            parse_frame["timestamp"].min()
        ).isoformat()
        validation["end_time"] = pd.Timestamp(
            parse_frame["timestamp"].max()
        ).isoformat()
        validations.append(validation)
    result = {
        "schema": VALIDATION_SCHEMA_VERSION,
        "status": "passed",
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "episodes_available": int(loaded.frame["episode_id"].nunique()),
        "episodes_validated": len(validations),
        "event_records_compared": sum(
            item["event_records_compared"] for item in validations
        ),
        "event_fields_compared": sum(
            item["event_fields_compared"] for item in validations
        ),
        "future_confirmed_structure_written_back": False,
        "input_audit": dict(loaded.audit),
        "episode_validations": validations,
    }
    if output_path is not None:
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="Validate strict Chan event-prefix invariance on real QDP bars."
    )
    argument_parser.add_argument("--symbol", required=True)
    argument_parser.add_argument("--start-date", default="2024-01-01")
    argument_parser.add_argument("--end-date", default="2025-12-31")
    argument_parser.add_argument(
        "--definition", default=str(parser.DEFAULT_DEFINITION_PATH)
    )
    argument_parser.add_argument("--output", default="")
    argument_parser.add_argument("--maximum-episodes", type=int, default=3)
    argument_parser.add_argument("--cutoffs-per-episode", type=int, default=8)
    argument_parser.add_argument(
        "--stroke-rule",
        choices=(
            "disjoint_fractals_and_price_ranges",
            "disjoint_fractals_only",
        ),
        default="",
    )
    return argument_parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    variants = {"stroke_rule": args.stroke_rule} if args.stroke_rule else {}
    result = validate_symbol_episodes(
        args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        definition_path=args.definition,
        output_path=args.output or None,
        variant_overrides=variants,
        maximum_episodes=args.maximum_episodes,
        cutoffs_per_episode=args.cutoffs_per_episode,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
