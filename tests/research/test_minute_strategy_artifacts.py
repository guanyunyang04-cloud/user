from __future__ import annotations

import pandas as pd
import pytest

from quantlab.research.minute_strategy_artifacts import (
    MinuteStrategyArtifactError,
    normalized_partition_complete,
    normalized_partition_paths,
    split_outcome_frame,
    write_normalized_partition,
)
from quantlab.research.minute_strategy_portfolio import (
    _read_signal_file,
    _signal_paths,
)


def _outcomes() -> pd.DataFrame:
    common = {
        "symbol": "000001.SZ",
        "signal_date": "2022-01-03",
        "signal_time": "093100000",
        "signal_executable": True,
        "entry_date": "2022-01-03",
        "entry_time": "093200000",
        "entry_price": 10.0,
        "entry_adjusted_price": 10.0,
        "entry_observed": True,
        "entry_executable": True,
        "gross_return_5m": 0.01,
        "net_return_5m": 0.009,
        "reference_symbol": None,
        "liquidity_match_ratio": None,
    }
    first = {
        **common,
        "signal_id": "sig-main",
        "strategy_id": "s1_touch_reclaim",
        "reference_signal_id": None,
    }
    second = {
        **common,
        "signal_id": "sig-control",
        "strategy_id": "s0_liquidity_matched",
        "reference_signal_id": "sig-main",
    }
    return pd.DataFrame([first, second])


def test_split_outcome_frame_reuses_one_path_and_keeps_reference_mapping() -> None:
    events, paths, references = split_outcome_frame(_outcomes())

    assert len(events) == 2
    assert len(paths) == 1
    assert len(references) == 2
    assert paths["path_id"].is_unique
    assert set(references["reference_signal_id"].dropna()) == {"sig-main"}
    assert events["path_id"].nunique() == 1


def test_write_normalized_partition_creates_three_readable_tables(tmp_path) -> None:
    stats = write_normalized_partition(_outcomes(), tmp_path, "2022-01-03")

    assert stats == {
        "trade_date": "2022-01-03",
        "event_rows": 2,
        "path_rows": 1,
        "reference_rows": 2,
    }
    assert len(normalized_partition_paths(tmp_path, "events")) == 1
    assert len(normalized_partition_paths(tmp_path, "paths")) == 1
    assert len(normalized_partition_paths(tmp_path, "references")) == 1
    assert normalized_partition_complete(tmp_path, "2022-01-03")
    assert pd.read_parquet(normalized_partition_paths(tmp_path, "paths")[0])["path_id"].is_unique


def test_write_normalized_partition_rejects_mismatched_trade_date(tmp_path) -> None:
    with pytest.raises(MinuteStrategyArtifactError, match="trade_date_mismatch"):
        write_normalized_partition(_outcomes(), tmp_path, "2022-01-04")


def test_path_deduplication_uses_normalized_time_key() -> None:
    outcomes = pd.concat(
        [
            _outcomes().iloc[[0]].copy(),
            _outcomes().iloc[[0]].assign(
                signal_id="sig-equivalent-time",
                signal_time="09:31:00",
            ),
        ],
        ignore_index=True,
    )

    _, paths, _ = split_outcome_frame(outcomes)

    assert len(paths) == 1


def test_split_outcome_frame_rejects_duplicate_signal_ids() -> None:
    duplicate = pd.concat([_outcomes(), _outcomes().iloc[[0]]], ignore_index=True)

    with pytest.raises(MinuteStrategyArtifactError, match="signal_id_duplicate"):
        split_outcome_frame(duplicate)


def test_portfolio_reads_normalized_event_and_path_tables(tmp_path) -> None:
    source = tmp_path / "run"
    source.mkdir()
    (source / "outcomes" / "date=2022-01-03").mkdir(parents=True)
    _outcomes().to_parquet(
        source / "outcomes" / "date=2022-01-03" / "outcomes.parquet",
        index=False,
    )
    write_normalized_partition(
        _outcomes(), source / "normalized", "2022-01-03"
    )

    paths = _signal_paths([source])
    frame = _read_signal_file(paths["2022-01-03"], ("s1_touch_reclaim",))

    assert len(frame) == 1
    assert frame.iloc[0]["entry_time"] == "093200000"
    assert frame.iloc[0]["entry_price"] == pytest.approx(10.0)


def test_signal_paths_falls_back_for_partial_normalized_migration(tmp_path) -> None:
    source = tmp_path / "run"
    for trade_date in ("2022-01-03", "2022-01-04"):
        outcome = _outcomes().assign(signal_date=trade_date, entry_date=trade_date)
        destination = source / "outcomes" / f"date={trade_date}"
        destination.mkdir(parents=True)
        outcome.to_parquet(destination / "outcomes.parquet", index=False)
    write_normalized_partition(
        _outcomes(), source / "normalized", "2022-01-03"
    )

    paths = _signal_paths([source])

    assert set(paths) == {"2022-01-03", "2022-01-04"}
    assert all(path.name == "outcomes.parquet" for path in paths.values())
