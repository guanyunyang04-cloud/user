from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from quantlab.research.minute_v2 import builder as minute_v2_builder
from quantlab.research.minute_v2 import cli as minute_v2_cli
from quantlab.research.minute_v2.builder import (
    BUILD_IMPLEMENTATION_REVISION,
    _artifact,
    _checkpoint_specs_compatible,
    _manifest_is_complete,
    _materialize_cached_view,
    _protect_existing_manifest,
    _remove_stale_base_artifact,
    verify_month,
)
from quantlab.research.minute_v2.contracts import (
    DAILY_WINDOWS,
    EXPECTED_DECISION_BARS,
    MODEL_FEATURE_COLUMNS,
    MinuteV2Config,
    MinuteV2Error,
    is_decision_bar,
)
from quantlab.research.minute_v2.features import build_feature_frame
from quantlab.research.minute_v2.labels import (
    _fixture_factor_quality,
    build_label_frame,
)
from quantlab.research.minute_v2.mining import mine_formula_features
from quantlab.research.minute_v2.models import (
    evaluate_scores,
    fit_ridge,
    fit_ridge_chunks,
    rule_score,
)
from quantlab.research.minute_v2.replay import EventReplayConfig, replay_events
from quantlab.research.minute_v2.sampling import (
    EVENT_CONTEXT_COLUMNS,
    EVENT_FLAG_COLUMNS,
    audit_candidate_recall,
    audit_candidate_recall_files,
    build_event_frame,
)
from quantlab.research.minute_v2.source import stock_day_query
from quantlab.research.minute_v2.training import (
    _collect_cross_section_sample,
    _evaluate_score_file,
    _load_top_scored,
    _period_parts,
    _write_multiple_scored_periods,
    _write_scored_period,
)


def _times() -> list[str]:
    morning = pd.date_range("2000-01-01 09:31", "2000-01-01 11:30", freq="min")
    afternoon = pd.date_range("2000-01-01 13:01", "2000-01-01 15:00", freq="min")
    return [*morning.strftime("%H%M00000"), *afternoon.strftime("%H%M00000")]


def test_checkpoint_reuses_parts_when_only_resource_controls_change() -> None:
    current = {
        "schema": "quantlab.minute_v2_day_parts/3",
        "source_dataset_ids": {"minute": "stable"},
        "year": 2022,
        "month": 6,
        "keep_base": True,
        "extended_end": "2022-07-08",
        "config": {
            "processing_days_per_chunk": 1,
            "duckdb_threads": 2,
            "memory_floor_gib": 4.0,
            "duckdb_memory_limit_gib": 1.0,
            "candidate_background_percent": 1,
        },
    }
    requested = {
        **current,
        "config": {
            **current["config"],
            "duckdb_threads": 4,
            "memory_floor_gib": 1.0,
            "duckdb_memory_limit_gib": 2.0,
        },
    }
    assert _checkpoint_specs_compatible(current, requested)
    requested["config"]["candidate_background_percent"] = 2
    assert not _checkpoint_specs_compatible(current, requested)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("minimum_daily_liquidity_rank", float("nan"), "liquidity_rank_invalid"),
        ("candidate_stock_rank_floor", float("inf"), "candidate_stock_rank_invalid"),
        ("maximum_participation_rate", float("nan"), "participation_rate_invalid"),
        ("memory_floor_gib", float("inf"), "memory_floor_invalid"),
        ("commission_bps", float("nan"), "commission_bps_invalid"),
        ("processing_days_per_chunk", 1.5, "processing_chunk_invalid"),
        ("duckdb_threads", 0, "duckdb_threads_invalid"),
        ("morning_decision_start", "093200000", "morning_decision_start_contract_invalid"),
    ],
)
def test_minute_v2_config_rejects_nonfinite_or_ignored_contract_values(
    field: str,
    value: object,
    message: str,
) -> None:
    config = MinuteV2Config(**{field: value})
    with pytest.raises(MinuteV2Error, match=message):
        config.validate()


def test_builder_refuses_old_month_manifest_without_force(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema":"quantlab.minute_v2_month/1","status":"ok"}',
        encoding="utf-8",
    )
    with pytest.raises(MinuteV2Error, match="requires_force"):
        _protect_existing_manifest(manifest, force=False)
    _protect_existing_manifest(manifest, force=True)


def test_builder_rejects_non_object_existing_manifest(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(MinuteV2Error, match="unreadable"):
        _protect_existing_manifest(manifest, force=False)


def test_drop_base_removes_only_the_exact_month_artifact(tmp_path) -> None:
    base = tmp_path / "base.parquet"
    base.write_bytes(b"stale")
    assert _remove_stale_base_artifact(tmp_path)
    assert not base.exists()
    assert not _remove_stale_base_artifact(tmp_path)


def test_verify_month_rejects_unexpected_base_for_drop_base_manifest(tmp_path) -> None:
    for name in ("events", "labels"):
        pd.DataFrame(
            {"symbol": ["A"], "trade_date": ["2022-06-01"], "bar_time": ["093100000"]}
        ).to_parquet(tmp_path / f"{name}.parquet", index=False)
    (tmp_path / "base.parquet").write_bytes(b"stale")
    manifest = tmp_path / "manifest.json"
    artifacts = {
        name: _artifact(tmp_path / f"{name}.parquet") for name in ("events", "labels")
    }
    manifest.write_text(
        json.dumps(
            {
                "schema": "quantlab.minute_v2_month/2",
                "status": "ok",
                "build_spec": {
                    "schema": "quantlab.minute_v2_build_spec/1",
                    "keep_base": False,
                    "implementation_revision": BUILD_IMPLEMENTATION_REVISION,
                    "content_signature": "test",
                },
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MinuteV2Error, match="unexpected_base_artifact"):
        verify_month(manifest)


def test_manifest_reuse_requires_build_identity_and_full_artifact_fingerprint(tmp_path) -> None:
    artifact = tmp_path / "events.parquet"
    labels_artifact = tmp_path / "labels.parquet"
    pd.DataFrame({"symbol": ["A"], "trade_date": ["2022-06-01"], "bar_time": ["093100000"]}).to_parquet(
        artifact, index=False
    )
    pd.DataFrame({"symbol": ["A"], "trade_date": ["2022-06-01"], "bar_time": ["093100000"]}).to_parquet(
        labels_artifact, index=False
    )
    record = _artifact(artifact)
    labels_record = _artifact(labels_artifact)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "quantlab.minute_v2_month/2",
                "status": "ok",
                "build_spec": {"token": 1},
                "artifacts": {"events": record, "labels": {**labels_record, "path": str(tmp_path / "missing.parquet")}},
            }
        ),
        encoding="utf-8",
    )
    # The manifest is incomplete because labels.parquet is absent.
    assert not _manifest_is_complete(
        manifest, keep_base=False, expected_build_spec={"token": 1}
    )

    # A valid single-artifact contract is accepted, but a different build
    # identity or a changed file is rejected.
    manifest.write_text(
        json.dumps(
            {
                "schema": "quantlab.minute_v2_month/2",
                "status": "ok",
                "build_spec": {"token": 1},
                "artifacts": {"events": record, "labels": labels_record},
            }
        ),
        encoding="utf-8",
    )
    assert _manifest_is_complete(
        manifest, keep_base=False, expected_build_spec={"token": 1}
    )
    assert not _manifest_is_complete(
        manifest, keep_base=False, expected_build_spec={"token": 2}
    )
    artifact.write_bytes(artifact.read_bytes() + b"x")
    assert not _manifest_is_complete(
        manifest, keep_base=False, expected_build_spec={"token": 1}
    )


def test_cached_views_require_their_own_artifact_fingerprint(tmp_path) -> None:
    source = pd.DataFrame({"value": [1, 2, 3]})
    path = tmp_path / "cache.parquet"
    metadata = path.with_suffix(".json")
    con = duckdb.connect(":memory:")
    try:
        con.register("source", source)
        _, first_reused = _materialize_cached_view(
            con,
            view_name="cached_values",
            path=path,
            query="SELECT * FROM source",
            metadata_path=metadata,
            cache_spec={"test": True},
        )
        _, second_reused = _materialize_cached_view(
            con,
            view_name="cached_values",
            path=path,
            query="SELECT * FROM source",
            metadata_path=metadata,
            cache_spec={"test": True},
        )
        assert not first_reused
        assert second_reused
        pd.DataFrame({"value": [4, 5, 6]}).to_parquet(path, index=False)
        _, after_mutation_reused = _materialize_cached_view(
            con,
            view_name="cached_values",
            path=path,
            query="SELECT * FROM source",
            metadata_path=metadata,
            cache_spec={"test": True},
        )
        assert not after_mutation_reused
    finally:
        con.close()


def test_verify_month_rejects_external_artifact_paths(tmp_path) -> None:
    for name in ("events", "labels"):
        pd.DataFrame(
            {"symbol": ["A"], "trade_date": ["2022-06-01"], "bar_time": ["093100000"]}
        ).to_parquet(tmp_path / f"{name}.parquet", index=False)
    outside = tmp_path / "outside.parquet"
    pd.DataFrame(
        {"symbol": ["A"], "trade_date": ["2022-06-01"], "bar_time": ["093100000"]}
    ).to_parquet(outside, index=False)
    manifest = tmp_path / "manifest.json"
    artifacts = {
        name: _artifact(tmp_path / f"{name}.parquet") for name in ("events", "labels")
    }
    artifacts["labels"]["path"] = str(outside)
    manifest.write_text(
        json.dumps(
            {
                "schema": "quantlab.minute_v2_month/2",
                "status": "ok",
                "build_spec": {
                    "schema": "quantlab.minute_v2_build_spec/1",
                    "keep_base": False,
                    "implementation_revision": BUILD_IMPLEMENTATION_REVISION,
                    "content_signature": "test",
                },
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MinuteV2Error, match="artifact_path_invalid"):
        verify_month(manifest)


def test_training_period_parts_rejects_malformed_and_external_manifests(tmp_path) -> None:
    directory = tmp_path / "months" / "year=2022" / "month=01"
    directory.mkdir(parents=True)
    manifest = directory / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(MinuteV2Error, match="manifest_unreadable"):
        _period_parts(tmp_path, 2022, 2022)

    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"placeholder")
    manifest.write_text(
        json.dumps(
            {
                "schema": "quantlab.minute_v2_month/2",
                "status": "ok",
                "year": 2022,
                "month": 1,
                "build_spec": {
                    "schema": "quantlab.minute_v2_build_spec/1",
                    "implementation_revision": BUILD_IMPLEMENTATION_REVISION,
                    "keep_base": True,
                    "content_signature": "test",
                },
                "artifacts": {
                    "events": {"path": str(outside)},
                    "labels": {"path": str(directory / "labels.parquet")},
                    "base": {"path": str(directory / "base.parquet")},
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MinuteV2Error, match="artifact_path_invalid"):
        _period_parts(tmp_path, 2022, 2022)


def test_verify_dataset_rejects_malformed_month_list_and_external_paths(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(MinuteV2Error, match="unreadable"):
        minute_v2_builder.verify_dataset(manifest)

    common = {
        "schema": "quantlab.minute_v2_dataset/1",
        "status": "ok",
        "start_year": 2022,
        "end_year": 2022,
        "month_count": 12,
        "expected_decision_rows": 0,
        "event_rows": 0,
        "label_rows": 0,
        "observed_label_rows": 0,
    }
    malformed = {
        **common,
        "months": [None] * 12,
    }
    manifest.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(MinuteV2Error, match="month_record_invalid"):
        minute_v2_builder.verify_dataset(manifest)

    records = [
        {
            "year": 2022,
            "month": month,
            "manifest": str(tmp_path / "outside" / f"{month:02d}.json"),
        }
        for month in range(1, 13)
    ]
    manifest.write_text(
        json.dumps({**common, "months": records}),
        encoding="utf-8",
    )
    with pytest.raises(MinuteV2Error, match="month_manifest_path_invalid"):
        minute_v2_builder.verify_dataset(manifest)


def test_verify_dataset_uses_verified_contract_totals(tmp_path, monkeypatch) -> None:
    root = tmp_path / "months"
    records = []
    for month in range(1, 13):
        directory = root / "year=2022" / f"month={month:02d}"
        directory.mkdir(parents=True)
        month_manifest = directory / "manifest.json"
        month_manifest.write_text(
            json.dumps(
                {
                    "year": 2022,
                    "month": month,
                    "date_selection": {"selected_trading_days": month},
                    # Deliberately inconsistent values: verify_dataset must
                    # use the verifier's computed contract instead.
                    "verification": {
                        "expected_decision_rows": 999,
                        "event_rows": 999,
                        "label_rows": 999,
                        "observed_label_rows": 999,
                    },
                }
            ),
            encoding="utf-8",
        )
        records.append(
            {
                "year": 2022,
                "month": month,
                "manifest": str(month_manifest),
            }
        )

    def fake_verify(path):
        return {
            "artifacts": {
                "base": {"size": 10},
                "events": {"size": 5},
                "labels": {"size": 5},
            },
            "contract": {
                "expected_decision_rows": 10,
                "event_rows": 5,
                "label_rows": 5,
                "observed_label_rows": 4,
            },
        }

    monkeypatch.setattr(minute_v2_builder, "verify_month", fake_verify)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "quantlab.minute_v2_dataset/1",
                "status": "ok",
                "start_year": 2022,
                "end_year": 2022,
                "month_count": 12,
                "months": records,
                "expected_decision_rows": 120,
                "event_rows": 60,
                "label_rows": 60,
                "observed_label_rows": 48,
            }
        ),
        encoding="utf-8",
    )
    result = minute_v2_builder.verify_dataset(manifest)
    assert result["month_count"] == 12
    assert result["selected_trading_days"] == sum(range(1, 13))
    assert result["expected_decision_rows"] == 120
    assert result["artifact_bytes"] == 12 * 20


def test_candidate_recall_cli_writes_complete_json_output(tmp_path, monkeypatch, capsys) -> None:
    expected = {
        "schema": "quantlab.minute_v2_candidate_recall/1",
        "target": "label_return_5m",
        "top_k_recall": {"top_1": 1.0},
    }
    calls: list[tuple[str, str, str]] = []

    def fake_audit(base: str, candidates: str, outcomes: str, **kwargs):
        calls.append((base, candidates, outcomes))
        assert kwargs == {"target": "label_return_5m", "top_k": [1]}
        return expected

    monkeypatch.setattr(minute_v2_cli, "audit_candidate_recall_files", fake_audit)
    output = tmp_path / "recall.json"
    assert (
        minute_v2_cli.main(
            [
                "audit-candidate-recall",
                "--base",
                "base.parquet",
                "--candidates",
                "candidates.parquet",
                "--outcomes",
                "outcomes.parquet",
                "--top-k",
                "1",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert calls == [("base.parquet", "candidates.parquet", "outcomes.parquet")]
    assert json.loads(output.read_text(encoding="utf-8")) == expected
    assert json.loads(capsys.readouterr().out) == expected


def test_stage_one_cli_forwards_manifest_output_and_top_k(monkeypatch, capsys) -> None:
    expected = {
        "schema": "quantlab.minute_v2_stage_one_audit/1",
        "status": "ok",
    }
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_stage_one(manifest: str, **kwargs):
        calls.append((manifest, kwargs))
        return expected

    monkeypatch.setattr(minute_v2_cli, "run_stage_one_audit", fake_stage_one)
    assert (
        minute_v2_cli.main(
            [
                "stage-one-audit",
                "--manifest",
                "month.json",
                "--output-directory",
                "audit-output",
                "--top-k",
                "1",
                "5",
                "--force",
            ]
        )
        == 0
    )
    assert calls == [
        (
            "month.json",
            {
                "output_directory": "audit-output",
                "top_k": (1, 5),
                "force": True,
            },
        )
    ]
    assert json.loads(capsys.readouterr().out) == expected


def _bars(
    *,
    dates: tuple[str, ...] = ("2022-06-01",),
    symbols: tuple[str, ...] = ("600000.SH", "000001.SZ"),
) -> pd.DataFrame:
    rows = []
    for date_index, trade_date in enumerate(dates):
        for symbol_index, symbol in enumerate(symbols):
            for minute_index, bar_time in enumerate(_times()):
                price = 10.0 + symbol_index + date_index * 0.1 + minute_index * 0.001
                volume = 1000.0 + minute_index + symbol_index * 10
                rows.append(
                    {
                        "symbol": symbol,
                        "trade_date": trade_date,
                        "bar_time": bar_time,
                        "open": price,
                        "high": price + 0.01,
                        "low": price - 0.01,
                        "close": price + 0.005,
                        "volume": volume,
                        "amount": volume * (price + 0.002),
                    }
                )
    return pd.DataFrame(rows)


def _stock_days(
    *,
    trade_date: str = "2022-06-01",
    symbols: tuple[str, ...] = ("600000.SH", "000001.SZ"),
) -> pd.DataFrame:
    rows = []
    for symbol_index, symbol in enumerate(symbols):
        row = {
                "symbol": symbol,
                "trade_date": trade_date,
                "industry_name": f"industry_{symbol_index}",
                "adjust_factor": 1.0,
                "previous_adjust_factor": 1.0,
                "previous_close": 9.9 + symbol_index,
                # The auction price is intentionally different from the first
                # continuous bar so the 60-minute opening seed is observable.
                "auction_price": 9.5 + symbol_index,
                "auction_amount": 1_000_000.0,
                "previous_return_1d": 0.01,
                "previous_amount_20d": 200_000_000.0,
                "history_120d_available": True,
                "history_240d_available": True,
                "previous_total_share": 2_000_000_000.0,
                "previous_float_share": 1_000_000_000.0,
                "previous_total_mv": 20_000_000.0,
                "previous_circ_mv": 10_000_000.0,
                "previous_turnover_rate": 1.2,
                "previous_pe": 12.0,
                "previous_pb": 1.1,
                "corporate_action_today": False,
                "cash_dividend_per_10": 0.0,
                "bonus_share_per_10": 0.0,
                "transfer_share_per_10": 0.0,
                "daily_liquidity_rank": 0.8,
                "exclude_open": False,
                "exclude_high": symbol_index == 1,
                "exclude_low": False,
                "exclude_close": False,
            }
        for window in DAILY_WINDOWS:
            row[f"previous_return_{window}d"] = window / 10_000.0
            row[f"previous_close_to_sma_{window}d"] = window / 20_000.0
            row[f"previous_volatility_{window}d"] = 0.01 + window / 100_000.0
            row[f"previous_amount_ratio_{window}d"] = window / 50_000.0
        rows.append(row)
    return pd.DataFrame(rows)


def _stock_day_source_frames() -> tuple[dict[str, pd.DataFrame], list[str]]:
    dates = pd.bdate_range("2022-01-03", periods=62).strftime("%Y-%m-%d").tolist()
    symbol = "600000.SH"
    daily = pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": dates,
            "close": np.linspace(10.0, 10.61, len(dates)),
            "amount": 100_000_000.0 + np.arange(len(dates)) * 100_000.0,
        }
    )
    factors = pd.DataFrame(
        {"symbol": symbol, "trade_date": dates, "adjust_factor": 1.0}
    )
    capital = pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": dates,
            "total_share": 2_000_000_000.0,
            "float_share": 1_000_000_000.0,
        }
    )
    valuation = pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": dates,
            "total_mv": 20_000_000.0 + np.arange(len(dates)),
            "circ_mv": 10_000_000.0 + np.arange(len(dates)),
            "turnover_rate": 1.0,
            "pe": 12.0,
            "pb": 1.2,
        }
    )
    targets = dates[59:61]
    frames = {
        "trading_calendar": pd.DataFrame(
            {"trade_date": dates, "is_open": True, "exchange": "SSE"}
        ),
        "adjust_factor": factors,
        "daily_raw": daily,
        "share_capital": capital,
        "valuation": valuation,
        "universe_snapshot": pd.DataFrame(
            {"symbol": symbol, "trade_date": targets, "board": "main"}
        ),
        "security_status": pd.DataFrame(
            {
                "symbol": symbol,
                "trade_date": targets,
                "is_st": False,
                "is_suspended": False,
                "is_delisted": False,
            }
        ),
        "industry_concept": pd.DataFrame(
            {
                "symbol": symbol,
                "trade_date": targets,
                "industry_name": "bank",
                "industry": "bank",
            }
        ),
        "opening_auction": pd.DataFrame(
            {
                "symbol": symbol,
                "trade_date": targets,
                "volume": 1000.0,
                "amount": 10_000.0,
            }
        ),
        "corporate_actions": pd.DataFrame(
            {
                "symbol": [symbol],
                "trade_date": [targets[1]],
                "announcement_date": [targets[0]],
                "ex_date": [targets[1]],
                "cash_dividend_per_10": [1.0],
                "bonus_share_per_10": [0.0],
                "transfer_share_per_10": [0.0],
            }
        ),
        "minute_feature_exclusions": pd.DataFrame(
            {
                "symbol": pd.Series(dtype="string"),
                "trade_date": pd.Series(dtype="string"),
                "exclude_open": pd.Series(dtype=bool),
                "exclude_high": pd.Series(dtype=bool),
                "exclude_low": pd.Series(dtype=bool),
                "exclude_close": pd.Series(dtype=bool),
            }
        ),
        "session_feature_exclusions": pd.DataFrame(
            {
                "symbol": pd.Series(dtype="string"),
                "trade_date": pd.Series(dtype="string"),
                "exclusion_reason": pd.Series(dtype="string"),
            }
        ),
    }
    return frames, dates


def _run_stock_day_fixture(frames: dict[str, pd.DataFrame], dates: list[str]) -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    try:
        for name, frame in frames.items():
            con.register(name, frame)
        return con.execute(
            stock_day_query(
                start_date=dates[59],
                end_date=dates[60],
                config=MinuteV2Config(),
            )
        ).fetchdf()
    finally:
        con.close()


def test_stock_pool_requires_sixty_prior_days_and_lags_daily_state() -> None:
    frames, dates = _stock_day_source_frames()
    original = _run_stock_day_fixture(frames, dates)
    assert original["trade_date"].astype(str).tolist() == [dates[60]]
    assert not bool(original.iloc[0]["history_120d_available"])
    assert bool(original.iloc[0]["corporate_action_today"])
    mutated = {name: frame.copy() for name, frame in frames.items()}
    current = mutated["daily_raw"]["trade_date"].eq(dates[60])
    mutated["daily_raw"].loc[current, ["close", "amount"]] *= 100.0
    mutated["valuation"].loc[
        mutated["valuation"]["trade_date"].eq(dates[60]),
        ["total_mv", "circ_mv", "turnover_rate", "pe", "pb"],
    ] *= 100.0
    after = _run_stock_day_fixture(mutated, dates)
    assert_frame_equal(original, after, check_exact=True)


@pytest.mark.parametrize("domain,field", [("daily_raw", "close"), ("daily_raw", "amount"), ("adjust_factor", "adjust_factor")])
def test_stock_pool_does_not_count_nonfinite_daily_history(domain: str, field: str) -> None:
    frames, dates = _stock_day_source_frames()
    mutated = {name: frame.copy() for name, frame in frames.items()}
    mutated[domain].loc[mutated[domain]["trade_date"].eq(dates[58]), field] = np.inf

    result = _run_stock_day_fixture(mutated, dates)

    assert result.empty


def test_decision_grid_excludes_lunch_and_closing_auction() -> None:
    selected = [value for value in _times() if is_decision_bar(value)]
    assert len(selected) == EXPECTED_DECISION_BARS
    assert "112900000" in selected
    assert "113000000" not in selected
    assert "145500000" in selected
    assert "145600000" not in selected


def test_features_are_causal_and_field_masks_are_specific() -> None:
    dates = ("2022-05-30", "2022-05-31", "2022-06-01")
    bars = _bars(dates=dates)
    stock_days = _stock_days()
    original = build_feature_frame(bars, stock_days)
    mutated_bars = bars.copy()
    future = mutated_bars["trade_date"].eq("2022-06-01") & mutated_bars["bar_time"].gt(
        "100000000"
    )
    mutated_bars.loc[future, ["open", "high", "low", "close"]] *= 8.0
    mutated_bars.loc[future, ["volume", "amount"]] *= 5.0
    mutated = build_feature_frame(mutated_bars, stock_days)
    columns = ["symbol", "trade_date", "bar_time", *MODEL_FEATURE_COLUMNS]
    before = original.loc[original["bar_time"] <= "100000000", columns].reset_index(drop=True)
    after = mutated.loc[mutated["bar_time"] <= "100000000", columns].reset_index(drop=True)
    assert_frame_equal(before, after, check_exact=True)
    masked = original.loc[original["symbol"] == "000001.SZ"]
    assert masked["cumulative_range"].isna().all()
    assert masked["breakout_20m"].isna().all()
    assert masked["return_5m"].notna().sum() > 0
    assert masked["rolling_range_20m"].isna().all()
    first = original.loc[original["symbol"] == "600000.SH"].set_index("bar_time")
    assert bool(first.loc["093100000", "crossed_overnight_from_previous_bar"])
    assert np.isfinite(first.loc["093100000", "return_240m"])
    assert first.loc["093100000", "partial_60m_return"] == pytest.approx(
        first.loc["093100000", "close"] / 9.5 - 1.0
    )
    assert first.loc["093100000", "partial_60m_range"] > 0.0
    assert first.loc["093100000", "m60_close_to_sma_5bar"] == first.loc[
        "102900000", "m60_close_to_sma_5bar"
    ]
    assert first.loc["102900000", "m60_close_to_sma_5bar"] != first.loc[
        "103000000", "m60_close_to_sma_5bar"
    ]
    assert np.isfinite(first.loc["093100000", "return_120m"])
    assert np.isfinite(first.loc["130100000", "return_120m"])
    assert np.isnan(masked.set_index("bar_time").loc["103000000", "m60_range_5bar"])

    prior_mutated = bars.copy()
    prior_tail = prior_mutated["trade_date"].eq("2022-05-31") & prior_mutated[
        "bar_time"
    ].ge("145000000")
    prior_mutated.loc[prior_tail, ["open", "high", "low", "close"]] *= 1.02
    changed = build_feature_frame(prior_mutated, stock_days)
    original_open = original.loc[
        (original["symbol"] == "600000.SH") & (original["bar_time"] == "093100000"),
        "moving_average_deviation_240m",
    ].iloc[0]
    changed_open = changed.loc[
        (changed["symbol"] == "600000.SH") & (changed["bar_time"] == "093100000"),
        "moving_average_deviation_240m",
    ].iloc[0]
    assert original_open != changed_open


@pytest.mark.parametrize("field", ["open", "high", "low", "close", "volume", "amount"])
def test_features_normalise_nonfinite_bar_values_without_emitting_infinite_features(field: str) -> None:
    bars = _bars()
    bars.loc[bars.index[10], field] = np.inf

    result = build_feature_frame(bars, _stock_days())

    assert len(result) == EXPECTED_DECISION_BARS * 2
    values = result.loc[:, list(MODEL_FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    assert not np.isinf(values.to_numpy(dtype=float)).any()


def test_features_mask_both_extrema_when_bar_high_is_below_low() -> None:
    bars = _bars(symbols=("600000.SH",))
    malformed = bars.copy()
    row = (malformed["bar_time"] == "094000000")
    malformed.loc[row, "high"] = malformed.loc[row, "low"] - 0.1
    result = build_feature_frame(malformed, _stock_days(symbols=("600000.SH",)))
    current = result.loc[result["bar_time"] == "094000000"].iloc[0]
    assert np.isnan(current["bar_range"])
    assert np.isfinite(current["return_5m"])


def test_features_reject_a_present_nonfinite_fixture_factor() -> None:
    bars = _bars()
    stock_days = _stock_days()
    stock_days.loc[0, "adjust_factor"] = np.inf

    with pytest.raises(MinuteV2Error, match="bar_day_context_adjust_factor_invalid"):
        build_feature_frame(bars, stock_days)


@pytest.mark.parametrize("value", [np.nan, np.inf, 0.0, -1.0])
def test_labels_reject_a_present_invalid_fixture_factor(value: float) -> None:
    events = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "trade_date": ["2022-06-01"],
            "adjust_factor": [1.0],
            "valid_high": [True],
            "valid_low": [True],
            "valid_close": [True],
        }
    )
    extended_bars = pd.DataFrame(
        {
            "symbol": ["600000.SH", "600000.SH"],
            "trade_date": ["2022-06-01", "2022-06-02"],
        }
    )
    label_stock_days = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "trade_date": ["2022-06-02"],
            "adjust_factor": [value],
        }
    )
    with pytest.raises(MinuteV2Error, match="bar_day_context_adjust_factor_invalid"):
        _fixture_factor_quality(events, extended_bars, label_stock_days)


def test_labels_use_explicit_fixture_factor_only_for_missing_context() -> None:
    events = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "trade_date": ["2022-06-01"],
            "adjust_factor": [2.0],
            "valid_high": [True],
            "valid_low": [True],
            "valid_close": [True],
        }
    )
    extended_bars = pd.DataFrame(
        {
            "symbol": ["600000.SH", "600000.SH"],
            "trade_date": ["2022-06-01", "2022-06-02"],
        }
    )
    label_stock_days = pd.DataFrame(
        columns=["symbol", "trade_date", "adjust_factor"]
    )
    factors, _ = _fixture_factor_quality(events, extended_bars, label_stock_days)
    values = factors.set_index(["symbol", "trade_date"])["adjust_factor"]
    assert values.loc[("600000.SH", "2022-06-01")] == 2.0
    assert values.loc[("600000.SH", "2022-06-02")] == 1.0


def test_candidate_gate_is_deterministic_and_keeps_every_decision_minute() -> None:
    features = build_feature_frame(_bars(), _stock_days())
    first = build_event_frame(features)
    second = build_event_frame(features)
    assert_frame_equal(first, second)
    assert 0 < len(first) <= len(features)
    assert first.groupby(["trade_date", "bar_time"]).ngroups == EXPECTED_DECISION_BARS
    assert first.groupby(["trade_date", "bar_time"]).size().ge(1).all()
    assert first["candidate_selected"].all()
    assert first["event_mask"].ge(0).all()
    assert not any(column.startswith(("label_", "entry_", "actual_exit")) for column in first.columns)
    assert set(EVENT_CONTEXT_COLUMNS).issubset(first.columns)
    assert set(EVENT_FLAG_COLUMNS).issubset(first.columns)
    assert not set(MODEL_FEATURE_COLUMNS).difference(EVENT_CONTEXT_COLUMNS).intersection(first.columns)


def test_candidate_recall_audit_uses_keys_and_reports_top_k_hits() -> None:
    base = pd.DataFrame(
        [
            {"symbol": "A", "trade_date": "2022-06-01", "bar_time": "093100000"},
            {"symbol": "B", "trade_date": "2022-06-01", "bar_time": "093100000"},
            {"symbol": "C", "trade_date": "2022-06-01", "bar_time": "093100000"},
            {"symbol": "A", "trade_date": "2022-06-01", "bar_time": "093200000"},
            {"symbol": "B", "trade_date": "2022-06-01", "bar_time": "093200000"},
            {"symbol": "C", "trade_date": "2022-06-01", "bar_time": "093200000"},
        ]
    )
    candidates = base.loc[
        ((base["symbol"] == "B") & (base["bar_time"] == "093100000"))
        | ((base["symbol"] == "B") & (base["bar_time"] == "093200000"))
    ].copy()
    outcomes = base.copy()
    outcomes["label_return_5m"] = [0.01, 0.03, 0.02, 0.03, 0.01, 0.02]
    result = audit_candidate_recall(base, candidates, outcomes)
    assert result["schema"] == "quantlab.minute_v2_candidate_recall/2"
    assert result["group_count"] == 2
    assert result["groups_without_candidates"] == 0
    assert result["top_k_hits"]["top_1"] == 1
    assert result["top_k_recall"]["top_1"] == 0.5
    assert result["top_k_recall"]["top_3"] == 1.0
    assert result["top_k_group_hit_rate"]["top_1"] == 0.5
    assert result["top_k_row_hits"]["top_1"] == 1
    assert result["top_k_rows"]["top_1"] == 2
    assert result["top_k_row_recall"]["top_1"] == 0.5
    assert result["top_k_random_group_hit_rate"]["top_1"] == pytest.approx(1 / 3)
    assert result["top_k_random_row_recall"]["top_1"] == pytest.approx(1 / 3)
    assert result["top_k_row_recall_lift_vs_random"]["top_1"] == pytest.approx(1.5)
    assert result["positive_outcome_row_recall"] == pytest.approx(1 / 3)
    assert result["random_positive_outcome_row_recall"] == pytest.approx(1 / 3)
    assert result["positive_utility_capture"] == pytest.approx(1 / 3)
    assert result["random_positive_utility_capture"] == pytest.approx(1 / 3)


def test_candidate_recall_reports_symmetric_upside_and_downside_capture() -> None:
    base = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "trade_date": ["2022-06-01"] * 4,
            "bar_time": ["093100000"] * 4,
        }
    )
    candidates = base.iloc[:2].copy()
    outcomes = base.assign(label_return_5m=[0.4, -0.3, 0.1, -0.2])
    result = audit_candidate_recall(base, candidates, outcomes, top_k=(1,))
    assert result["positive_outcome_row_recall"] == 0.5
    assert result["random_positive_outcome_row_recall"] == 0.5
    assert result["positive_utility_capture"] == pytest.approx(0.8)
    assert result["random_positive_utility_capture"] == 0.5
    assert result["negative_outcome_row_recall"] == 0.5
    assert result["random_negative_outcome_row_recall"] == 0.5
    assert result["negative_utility_capture"] == pytest.approx(0.6)
    assert result["random_negative_utility_capture"] == 0.5


def test_file_candidate_recall_matches_memory_and_checks_coverage(tmp_path) -> None:
    base = pd.DataFrame(
        [
            {"symbol": "A", "trade_date": "2022-06-01", "bar_time": "093100000"},
            {"symbol": "B", "trade_date": "2022-06-01", "bar_time": "093100000"},
            {"symbol": "C", "trade_date": "2022-06-01", "bar_time": "093100000"},
            {"symbol": "A", "trade_date": "2022-06-01", "bar_time": "093200000"},
            {"symbol": "B", "trade_date": "2022-06-01", "bar_time": "093200000"},
            {"symbol": "C", "trade_date": "2022-06-01", "bar_time": "093200000"},
        ]
    )
    candidates = base.loc[base["symbol"] == "B"].copy()
    outcomes = base.copy()
    outcomes["label_return_5m"] = [0.01, 0.03, 0.02, 0.03, 0.01, 0.02]
    base_path = tmp_path / "base.parquet"
    candidate_path = tmp_path / "candidates.parquet"
    outcome_path = tmp_path / "outcomes.parquet"
    base.to_parquet(base_path, index=False)
    candidates.to_parquet(candidate_path, index=False)
    outcomes.to_parquet(outcome_path, index=False)
    expected = audit_candidate_recall(base, candidates, outcomes)
    actual = audit_candidate_recall_files(base_path, candidate_path, outcome_path)
    for key in (
        "base_rows",
        "outcome_rows",
        "finite_outcome_rows",
        "candidate_rows",
        "group_count",
        "groups_without_candidates",
        "top_k_hits",
        "top_k_eligible_groups",
        "top_k_recall",
        "top_k_group_hit_rate",
        "top_k_row_hits",
        "top_k_rows",
        "top_k_row_recall",
    ):
        assert actual[key] == expected[key]
    for key in (
        "positive_outcome_row_recall",
        "random_positive_outcome_row_recall",
        "positive_utility_capture",
        "random_positive_utility_capture",
    ):
        assert actual[key] == pytest.approx(expected[key])
    with pytest.raises(MinuteV2Error, match="columns_missing"):
        audit_candidate_recall_files(
            base_path,
            candidate_path,
            outcome_path,
            target="missing_target",
        )
    incomplete = outcomes.iloc[:-1]
    incomplete.to_parquet(outcome_path, index=False)
    with pytest.raises(MinuteV2Error, match="coverage_invalid"):
        audit_candidate_recall_files(base_path, candidate_path, outcome_path)


def test_file_candidate_recall_rejects_duplicate_primary_keys(tmp_path) -> None:
    base = pd.DataFrame(
        [{"symbol": "A", "trade_date": "2022-06-01", "bar_time": "093100000"}]
    )
    candidates = pd.concat([base, base], ignore_index=True)
    outcomes = base.assign(label_return_5m=[0.01])
    base_path = tmp_path / "base.parquet"
    candidate_path = tmp_path / "candidates.parquet"
    outcome_path = tmp_path / "outcomes.parquet"
    base.to_parquet(base_path, index=False)
    candidates.to_parquet(candidate_path, index=False)
    outcomes.to_parquet(outcome_path, index=False)
    with pytest.raises(MinuteV2Error, match="candidates_duplicate_keys"):
        audit_candidate_recall_files(base_path, candidate_path, outcome_path)


def test_file_candidate_recall_rejects_implicit_key_type_casts(tmp_path) -> None:
    base = pd.DataFrame(
        [
            {"symbol": "A", "trade_date": "2022-06-01", "bar_time": "093100000"},
            {"symbol": "B", "trade_date": "2022-06-01", "bar_time": "093100000"},
        ]
    )
    outcomes = base.assign(label_return_5m=[0.1, 0.2])
    base_path = tmp_path / "base.parquet"
    candidate_path = tmp_path / "candidates.parquet"
    outcome_path = tmp_path / "outcomes.parquet"
    base.to_parquet(base_path, index=False)
    base.assign(bar_time=[93100000, 93100000]).to_parquet(candidate_path, index=False)
    outcomes.to_parquet(outcome_path, index=False)
    with pytest.raises(MinuteV2Error, match="key_types_inconsistent"):
        audit_candidate_recall_files(base_path, candidate_path, outcome_path)

    base.to_parquet(candidate_path, index=False)
    outcomes.assign(trade_date=pd.to_datetime(outcomes["trade_date"])).to_parquet(
        outcome_path, index=False
    )
    with pytest.raises(MinuteV2Error, match="key_types_inconsistent"):
        audit_candidate_recall_files(base_path, candidate_path, outcome_path)


def test_memory_candidate_recall_rejects_implicit_key_type_casts() -> None:
    base = pd.DataFrame(
        [{"symbol": "A", "trade_date": "2022-06-01", "bar_time": "093100000"}]
    )
    candidates = base.copy()
    outcomes = base.assign(
        trade_date=pd.to_datetime(base["trade_date"]),
        label_return_5m=[0.1],
    )
    with pytest.raises(MinuteV2Error, match="key_types_inconsistent"):
        audit_candidate_recall(base, candidates, outcomes)


def test_file_candidate_recall_reports_empty_finite_outcome_set(tmp_path) -> None:
    empty_keys = {
        "symbol": pd.Series(dtype="string"),
        "trade_date": pd.Series(dtype="string"),
        "bar_time": pd.Series(dtype="string"),
    }
    base = pd.DataFrame(empty_keys)
    candidates = base.copy()
    outcomes = base.assign(label_return_5m=pd.Series(dtype="float64"))
    paths = [tmp_path / name for name in ("base.parquet", "candidates.parquet", "outcomes.parquet")]
    for path, frame in zip(paths, (base, candidates, outcomes), strict=True):
        frame.to_parquet(path, index=False)
    with pytest.raises(MinuteV2Error, match="no_finite_outcomes"):
        audit_candidate_recall_files(*paths)
    with pytest.raises(MinuteV2Error, match="target_invalid"):
        audit_candidate_recall_files(*paths, target="symbol")


def _minimal_event_frame(*, symbol: str = "A") -> pd.DataFrame:
    values: dict[str, list[object]] = {
        "symbol": [symbol],
        "trade_date": ["2022-06-01"],
        "bar_time": ["093100000"],
    }
    for name in EVENT_CONTEXT_COLUMNS:
        values[name] = [None]
    for name in EVENT_FLAG_COLUMNS:
        values[name] = [False]
    values.update(
        {
            "candidate_attention": [True],
            "candidate_background_control": [False],
            "candidate_selected": [True],
            "candidate_reason_mask": [1],
            "event_mask": [1],
        }
    )
    return pd.DataFrame(values)


def _minimal_label_frame(*, symbol: str = "A") -> pd.DataFrame:
    values: dict[str, list[object]] = {
        "symbol": [symbol],
        "trade_date": ["2022-06-01"],
        "bar_time": ["093100000"],
    }
    from quantlab.research.minute_v2.labels import LABEL_COLUMNS

    for name in LABEL_COLUMNS:
        values.setdefault(name, [None])
    return pd.DataFrame(values)


def test_verify_month_checks_label_schema_and_key_equality(tmp_path) -> None:
    events = _minimal_event_frame()
    labels = _minimal_label_frame(symbol="B")
    event_path = tmp_path / "events.parquet"
    label_path = tmp_path / "labels.parquet"
    events.to_parquet(event_path, index=False)
    labels.to_parquet(label_path, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "quantlab.minute_v2_month/2",
                "status": "ok",
                "build_spec": {
                    "schema": "quantlab.minute_v2_build_spec/1",
                    "keep_base": False,
                    "implementation_revision": BUILD_IMPLEMENTATION_REVISION,
                    "content_signature": "test",
                },
                "verification": {
                    "expected_decision_rows": 1,
                    "decision_group_count": 1,
                },
                "artifacts": {
                    "events": _artifact(event_path),
                    "labels": _artifact(label_path),
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MinuteV2Error, match="event_label_key_contract_invalid"):
        verify_month(manifest)


def test_labels_use_next_bar_and_next_market_day() -> None:
    bars = _bars(dates=("2022-06-01", "2022-06-02"), symbols=("600000.SH",))
    features = build_feature_frame(
        bars.loc[bars["trade_date"] == "2022-06-01"],
        _stock_days(symbols=("600000.SH",)),
    )
    events = features.loc[features["bar_time"] == "093500000"].copy()
    label_stock_days = pd.DataFrame(
        [
            {
                "symbol": "600000.SH",
                "trade_date": "2022-06-02",
                "high": 10.4,
                "low": 10.0,
                "close": 10.3,
                "adjust_factor": 1.0,
                "previous_close": 10.2,
                "previous_adjust_factor": 1.0,
                "is_suspended": False,
                "is_delisted": False,
                "exclude_high": False,
                "exclude_low": False,
                "exclude_close": False,
                "corporate_action_count": 0,
            }
        ]
    )
    calendar = pd.DataFrame(
        [
            {"trade_date": "2022-06-01", "calendar_index": 1, "next_trade_date": "2022-06-02"},
            {"trade_date": "2022-06-02", "calendar_index": 2, "next_trade_date": None},
        ]
    )
    labels = build_label_frame(
        events,
        bars.loc[bars["trade_date"] == "2022-06-01"],
        bars,
        label_stock_days,
        calendar,
    )
    row = labels.iloc[0]
    assert row["entry_bar_time"] == "093600000"
    assert row["planned_exit_date"] == "2022-06-02"
    assert row["actual_exit_date"] == "2022-06-02"
    assert bool(row["entry_executable"])
    assert bool(row["label_observed"])
    assert np.isfinite(row["label_net_return"])
    assert row["label_net_return"] < row["label_gross_return"]
    assert bool(row["label_5m_observed"])
    assert bool(row["label_1d_observed"])


def test_minute_labels_are_near_close_safe_and_masks_are_field_specific() -> None:
    bars = _bars(dates=("2022-06-01", "2022-06-02"), symbols=("600000.SH",))
    stock_days = _stock_days(symbols=("600000.SH",))
    stock_days.loc[:, "exclude_high"] = True
    features = build_feature_frame(bars, stock_days)
    events = features.loc[
        features["bar_time"].isin(["093500000", "112900000", "145500000"])
    ].copy()
    empty_daily = pd.DataFrame(
        columns=[
            "symbol",
            "trade_date",
            "high",
            "low",
            "close",
            "adjust_factor",
            "previous_close",
            "previous_adjust_factor",
            "is_suspended",
            "is_delisted",
            "exclude_high",
            "exclude_low",
            "exclude_close",
            "corporate_action_count",
        ]
    )
    calendar = pd.DataFrame(
        [{"trade_date": "2022-06-01", "calendar_index": 1, "next_trade_date": None}]
    )
    calendar = pd.DataFrame(
        [
            {"trade_date": "2022-06-01", "calendar_index": 1, "next_trade_date": "2022-06-02"},
            {"trade_date": "2022-06-02", "calendar_index": 2, "next_trade_date": None},
        ]
    )
    labels = build_label_frame(
        events,
        bars.loc[bars["trade_date"] == "2022-06-01"],
        bars,
        empty_daily,
        calendar,
    )
    early = labels.loc[labels["bar_time"] == "093500000"].iloc[0]
    lunch = labels.loc[labels["bar_time"] == "112900000"].iloc[0]
    late = labels.loc[labels["bar_time"] == "145500000"].iloc[0]
    assert bool(early["entry_executable"])
    assert np.isfinite(early["label_return_5m"])
    assert np.isnan(early["label_mfe_5m"])
    assert np.isfinite(early["label_mae_5m"])
    assert lunch["label_end_bar_time_5m"] == "130500000"
    assert bool(lunch["label_5m_crossed_lunch"])
    assert lunch["label_5m_invalid_reason"] == ""
    assert bool(late["label_5m_observed"])
    assert bool(late["label_5m_crossed_overnight"])
    assert late["label_end_date_5m"] == "2022-06-02"
    assert late["label_end_bar_time_5m"] == "093500000"
    assert bool(late["label_session_5m_observed"])
    assert not bool(late["label_session_15m_observed"])
    assert late["label_session_15m_invalid_reason"] == "same_session_window_incomplete"


def test_session_labels_reject_a_missing_raw_bar() -> None:
    bars = _bars(symbols=("600000.SH",))
    features = build_feature_frame(bars, _stock_days(symbols=("600000.SH",)))
    events = features.loc[features["bar_time"] == "093500000"].copy()
    gapped = bars.loc[bars["bar_time"] != "093700000"].copy()
    empty_daily = pd.DataFrame(
        columns=[
            "symbol",
            "trade_date",
            "high",
            "low",
            "close",
            "adjust_factor",
            "previous_close",
            "previous_adjust_factor",
            "is_suspended",
            "is_delisted",
            "exclude_high",
            "exclude_low",
            "exclude_close",
            "corporate_action_count",
        ]
    )
    labels = build_label_frame(
        events,
        bars,
        gapped,
        empty_daily,
        pd.DataFrame(
            [{"trade_date": "2022-06-01", "calendar_index": 1, "next_trade_date": None}]
        ),
    )
    row = labels.iloc[0]
    assert not bool(row["label_session_5m_observed"])
    assert row["label_session_5m_invalid_reason"] == "same_session_window_incomplete"


def test_labels_normalise_nonfinite_entry_prices_to_explicit_unfilled() -> None:
    bars = _bars(dates=("2022-06-01", "2022-06-02"), symbols=("600000.SH",))
    features = build_feature_frame(
        bars.loc[bars["trade_date"] == "2022-06-01"],
        _stock_days(symbols=("600000.SH",)),
    )
    events = features.loc[features["bar_time"] == "093500000"].copy()
    malformed = bars.copy()
    malformed.loc[
        (malformed["trade_date"] == "2022-06-01")
        & (malformed["bar_time"] == "093600000"),
        "high",
    ] = np.nan
    empty_daily = pd.DataFrame(
        columns=[
            "symbol",
            "trade_date",
            "high",
            "low",
            "close",
            "adjust_factor",
            "previous_close",
            "previous_adjust_factor",
            "is_suspended",
            "is_delisted",
            "exclude_high",
            "exclude_low",
            "exclude_close",
            "corporate_action_count",
        ]
    )
    calendar = pd.DataFrame(
        [
            {
                "trade_date": "2022-06-01",
                "calendar_index": 1,
                "next_trade_date": "2022-06-02",
            },
            {"trade_date": "2022-06-02", "calendar_index": 2, "next_trade_date": None},
        ]
    )
    labels = build_label_frame(
        events,
        bars.loc[bars["trade_date"] == "2022-06-01"],
        malformed,
        empty_daily,
        calendar,
    )
    row = labels.iloc[0]
    assert pd.notna(row["entry_executable"])
    assert not bool(row["entry_executable"])
    assert row["entry_unfilled_reason"] == "entry_high_low_invalid"
    assert not bool(row["label_observed"])
    assert row["label_5m_invalid_reason"] == "continuous_entry_high_low_invalid"
    assert row["label_session_5m_invalid_reason"] == "entry_high_low_invalid"
    assert row["label_1d_invalid_reason"] == "entry_high_low_invalid"


@pytest.mark.parametrize("field", ["high", "low"])
def test_labels_keep_close_observable_when_one_future_extremum_is_nonfinite(field: str) -> None:
    bars = _bars(dates=("2022-06-01", "2022-06-02"), symbols=("600000.SH",))
    features = build_feature_frame(
        bars.loc[bars["trade_date"] == "2022-06-01"],
        _stock_days(symbols=("600000.SH",)),
    )
    events = features.loc[features["bar_time"] == "093500000"].copy()
    malformed = bars.copy()
    malformed.loc[
        (malformed["trade_date"] == "2022-06-01")
        & (malformed["bar_time"] == "094000000"),
        field,
    ] = np.nan
    empty_daily = pd.DataFrame(
        columns=[
            "symbol",
            "trade_date",
            "high",
            "low",
            "close",
            "adjust_factor",
            "previous_close",
            "previous_adjust_factor",
            "is_suspended",
            "is_delisted",
            "exclude_high",
            "exclude_low",
            "exclude_close",
            "corporate_action_count",
        ]
    )
    calendar = pd.DataFrame(
        [
            {
                "trade_date": "2022-06-01",
                "calendar_index": 1,
                "next_trade_date": "2022-06-02",
            },
            {"trade_date": "2022-06-02", "calendar_index": 2, "next_trade_date": None},
        ]
    )
    labels = build_label_frame(
        events,
        bars.loc[bars["trade_date"] == "2022-06-01"],
        malformed,
        empty_daily,
        calendar,
    )
    row = labels.iloc[0]
    assert bool(row["label_5m_observed"])
    assert np.isfinite(row["label_return_5m"])
    if field == "high":
        assert np.isnan(row["label_mfe_5m"])
        assert row["label_5m_mfe_invalid_reason"] == "future_high_nonfinite"
        assert np.isfinite(row["label_mae_5m"])
        assert row["label_5m_mae_invalid_reason"] == ""
        assert np.isnan(row["label_session_mfe_5m"])
        assert np.isfinite(row["label_session_mae_5m"])
    else:
        assert np.isfinite(row["label_mfe_5m"])
        assert row["label_5m_mfe_invalid_reason"] == ""
        assert np.isnan(row["label_mae_5m"])
        assert row["label_5m_mae_invalid_reason"] == "future_low_nonfinite"
        assert np.isfinite(row["label_session_mfe_5m"])
        assert np.isnan(row["label_session_mae_5m"])


def test_labels_mask_both_future_extrema_when_high_is_below_low() -> None:
    bars = _bars(dates=("2022-06-01", "2022-06-02"), symbols=("600000.SH",))
    features = build_feature_frame(
        bars.loc[bars["trade_date"] == "2022-06-01"],
        _stock_days(symbols=("600000.SH",)),
    )
    events = features.loc[features["bar_time"] == "093500000"].copy()
    malformed = bars.copy()
    target = (malformed["trade_date"] == "2022-06-01") & malformed["bar_time"].eq("094000000")
    malformed.loc[target, "high"] = malformed.loc[target, "low"] - 0.1
    empty_daily = pd.DataFrame(
        columns=[
            "symbol",
            "trade_date",
            "high",
            "low",
            "close",
            "adjust_factor",
            "previous_close",
            "previous_adjust_factor",
            "is_suspended",
            "is_delisted",
            "exclude_high",
            "exclude_low",
            "exclude_close",
            "corporate_action_count",
        ]
    )
    calendar = pd.DataFrame(
        [
            {"trade_date": "2022-06-01", "calendar_index": 1, "next_trade_date": "2022-06-02"},
            {"trade_date": "2022-06-02", "calendar_index": 2, "next_trade_date": None},
        ]
    )
    labels = build_label_frame(
        events,
        bars.loc[bars["trade_date"] == "2022-06-01"],
        malformed,
        empty_daily,
        calendar,
    )
    row = labels.iloc[0]
    assert bool(row["label_5m_observed"])
    assert np.isfinite(row["label_return_5m"])
    assert np.isnan(row["label_mfe_5m"])
    assert np.isnan(row["label_mae_5m"])
    assert row["label_5m_mfe_invalid_reason"] == "future_high_nonfinite"
    assert row["label_5m_mae_invalid_reason"] == "future_low_nonfinite"
    assert np.isnan(row["label_session_mfe_5m"])
    assert np.isnan(row["label_session_mae_5m"])


def test_labels_do_not_execute_a_nonfinite_exit_window() -> None:
    bars = _bars(dates=("2022-06-01", "2022-06-02"), symbols=("600000.SH",))
    features = build_feature_frame(
        bars.loc[bars["trade_date"] == "2022-06-01"],
        _stock_days(symbols=("600000.SH",)),
    )
    events = features.loc[features["bar_time"] == "093500000"].copy()
    malformed = bars.copy()
    malformed.loc[
        (malformed["trade_date"] == "2022-06-02")
        & (malformed["bar_time"] == "093500000"),
        ["high", "volume", "amount"],
    ] = np.nan
    empty_daily = pd.DataFrame(
        columns=[
            "symbol",
            "trade_date",
            "high",
            "low",
            "close",
            "adjust_factor",
            "previous_close",
            "previous_adjust_factor",
            "is_suspended",
            "is_delisted",
            "exclude_high",
            "exclude_low",
            "exclude_close",
            "corporate_action_count",
        ]
    )
    calendar = pd.DataFrame(
        [
            {
                "trade_date": "2022-06-01",
                "calendar_index": 1,
                "next_trade_date": "2022-06-02",
            },
            {"trade_date": "2022-06-02", "calendar_index": 2, "next_trade_date": None},
        ]
    )
    labels = build_label_frame(
        events,
        bars.loc[bars["trade_date"] == "2022-06-01"],
        malformed,
        empty_daily,
        calendar,
    )
    row = labels.iloc[0]
    assert not bool(row["label_observed"])
    assert pd.isna(row["label_net_return"])
    assert bool(row["label_5m_observed"])


def test_daily_label_keeps_close_and_low_when_high_is_nonfinite() -> None:
    bars = _bars(dates=("2022-06-01", "2022-06-02"), symbols=("600000.SH",))
    features = build_feature_frame(
        bars.loc[bars["trade_date"] == "2022-06-01"],
        _stock_days(symbols=("600000.SH",)),
    )
    events = features.loc[features["bar_time"] == "093500000"].copy()
    daily = pd.DataFrame(
        [
            {
                "symbol": "600000.SH",
                "trade_date": "2022-06-02",
                "high": np.nan,
                "low": 10.0,
                "close": 10.3,
                "adjust_factor": 1.0,
                "previous_close": 10.0,
                "previous_adjust_factor": 1.0,
                "is_suspended": False,
                "is_delisted": False,
                "exclude_high": False,
                "exclude_low": False,
                "exclude_close": False,
                "corporate_action_count": 0,
            }
        ]
    )
    calendar = pd.DataFrame(
        [
            {
                "trade_date": "2022-06-01",
                "calendar_index": 1,
                "next_trade_date": "2022-06-02",
            },
            {"trade_date": "2022-06-02", "calendar_index": 2, "next_trade_date": None},
        ]
    )
    labels = build_label_frame(
        events,
        bars.loc[bars["trade_date"] == "2022-06-01"],
        bars,
        daily,
        calendar,
    )
    row = labels.iloc[0]
    assert bool(row["label_1d_observed"])
    assert np.isfinite(row["label_return_1d"])
    assert np.isnan(row["label_mfe_1d"])
    assert np.isfinite(row["label_mae_1d"])


def test_daily_labels_cross_holidays_and_adjust_for_corporate_actions() -> None:
    calendar_days = [
        "2022-06-01",
        "2022-06-02",
        "2022-06-06",
        "2022-06-07",
        "2022-06-08",
        "2022-06-09",
        "2022-06-10",
        "2022-06-13",
        "2022-06-14",
        "2022-06-15",
        "2022-06-16",
    ]
    target = _bars(symbols=("600000.SH",))
    extended = _bars(dates=("2022-06-01", "2022-06-02"), symbols=("600000.SH",))
    features = build_feature_frame(target, _stock_days(symbols=("600000.SH",)))
    events = features.loc[features["bar_time"] == "093500000"].copy()
    entry_price = float(
        target.loc[target["bar_time"] == "093600000", "amount"].iloc[0]
        / target.loc[target["bar_time"] == "093600000", "volume"].iloc[0]
    )
    daily_rows = []
    for index, trade_date in enumerate(calendar_days[1:], start=1):
        factor = 2.0
        adjusted_close = entry_price if index == 1 else entry_price * (1.0 + index / 100.0)
        raw_close = adjusted_close / factor
        daily_rows.append(
            {
                "symbol": "600000.SH",
                "trade_date": trade_date,
                "high": raw_close * 1.01,
                "low": raw_close * 0.99,
                "close": raw_close,
                "adjust_factor": factor,
                "previous_close": 10.0 if index == 1 else raw_close,
                "previous_adjust_factor": 1.0 if index == 1 else factor,
                "is_suspended": False,
                "is_delisted": False,
                "exclude_high": False,
                "exclude_low": False,
                "exclude_close": False,
                "corporate_action_count": 1 if index == 1 else 0,
            }
        )
    calendar = pd.DataFrame(
        [
            {
                "trade_date": value,
                "calendar_index": index + 1,
                "next_trade_date": calendar_days[index + 1]
                if index + 1 < len(calendar_days)
                else None,
            }
            for index, value in enumerate(calendar_days)
        ]
    )
    labels = build_label_frame(
        events,
        target,
        extended,
        pd.DataFrame(daily_rows),
        calendar,
    )
    row = labels.iloc[0]
    assert row["label_end_date_3d"] == "2022-06-07"
    assert abs(float(row["label_return_1d"])) < 1.0e-12
    assert int(row["label_action_count_1d"]) == 1


def test_replay_does_not_replace_an_unfilled_top_rank() -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": "600000.SH",
                "trade_date": "2022-06-01",
                "bar_time": "093500000",
                "score": 2.0,
                "planned_exit_date": "2022-06-02",
                "entry_bar_time": "093600000",
                "entry_price": 10.0,
                "entry_amount": 10_000_000.0,
                "entry_executable": False,
                "actual_exit_date": None,
                "exit_amount": np.nan,
                "label_net_return": np.nan,
                "label_observed": False,
            },
            {
                "symbol": "000001.SZ",
                "trade_date": "2022-06-01",
                "bar_time": "093500000",
                "score": 1.0,
                "planned_exit_date": "2022-06-02",
                "entry_bar_time": "093600000",
                "entry_price": 11.0,
                "entry_amount": 10_000_000.0,
                "entry_executable": True,
                "actual_exit_date": "2022-06-02",
                "exit_amount": 10_000_000.0,
                "label_net_return": 0.02,
                "label_observed": True,
            },
        ]
    )
    result, _, trades = replay_events(
        frame,
        replay_config=EventReplayConfig(top_k_per_minute=1),
    )
    assert result["selected_event_count"] == 1
    assert result["unfilled_selection_count"] == 1
    assert trades.empty


def test_replay_skips_nonfinite_scores_capacity_and_returns() -> None:
    rows = []
    template = {
        "symbol": "600000.SH",
        "trade_date": "2022-06-01",
        "bar_time": "093500000",
        "score": 1.0,
        "planned_exit_date": "2022-06-02",
        "entry_bar_time": "093600000",
        "entry_price": 10.0,
        "entry_amount": 100_000.0,
        "entry_executable": True,
        "actual_exit_date": "2022-06-02",
        "exit_amount": 100_000.0,
        "label_net_return": 0.02,
        "label_observed": True,
    }
    for symbol, field, value in (
        ("A", "entry_amount", np.nan),
        ("B", "exit_amount", np.inf),
        ("C", "label_net_return", np.inf),
    ):
        row = {**template, "symbol": symbol, field: value}
        rows.append(row)
    result, _, trades = replay_events(
        pd.DataFrame(rows),
        replay_config=EventReplayConfig(score_threshold=-1.0e30),
    )
    assert result["selected_event_count"] == 3
    assert result["unfilled_selection_count"] == 3
    assert trades.empty


def test_rule_ridge_and_formula_mining_have_small_deterministic_contracts() -> None:
    rng = np.random.default_rng(7)
    rows = []
    for day in range(20):
        period = "train" if day < 14 else "validation"
        for minute in ("093500000", "100000000"):
            for symbol_index in range(10):
                signal = rng.normal()
                row = {
                    "symbol": f"{600000 + symbol_index}.SH",
                    "trade_date": f"2022-06-{day + 1:02d}",
                    "bar_time": minute,
                    "period": period,
                    "return_5m": signal,
                    "return_1m": signal * 0.2,
                    "label_net_return": signal * 0.01 + rng.normal(scale=0.001),
                }
                for feature in MODEL_FEATURE_COLUMNS:
                    row.setdefault(feature, rng.normal())
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame["score"] = rule_score(frame)
    assert frame["score"].notna().all()
    ridge = fit_ridge(
        frame.loc[frame["period"] == "train"],
        feature_names=("return_1m", "return_5m"),
        alpha=1.0,
    )
    prediction = ridge.predict(frame.loc[frame["period"] == "validation"])
    assert np.corrcoef(prediction, frame.loc[frame["period"] == "validation", "label_net_return"])[0, 1] > 0.8
    mining = mine_formula_features(
        frame.loc[frame["period"] == "train"],
        frame.loc[frame["period"] == "validation"],
        seed_features=("return_1m", "return_5m"),
        maximum_candidates=12,
        maximum_selected=3,
        minimum_coverage=0.9,
    )
    assert mining["candidate_count"] == 9
    assert mining["selected_count"] > 0


def test_streamed_ridge_matches_batch_fit_when_medians_are_fully_observed() -> None:
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "label_net_return": rng.normal(size=300),
            "f1": rng.normal(size=300),
            "f2": rng.normal(size=300),
        }
    )
    batch = fit_ridge(
        frame,
        target="label_net_return",
        feature_names=("f1", "f2"),
        alpha=1.0,
    )
    streamed, metadata = fit_ridge_chunks(
        lambda: [frame.iloc[:100], frame.iloc[100:200], frame.iloc[200:]],
        target="label_net_return",
        feature_names=("f1", "f2"),
        alpha=1.0,
        median_sample_size=1000,
    )
    assert metadata["row_count"] == len(frame)
    assert metadata["second_pass_rows"] == len(frame)
    assert np.allclose(batch.coefficients, streamed.coefficients)
    assert np.allclose(batch.means, streamed.means)
    assert np.allclose(batch.scales, streamed.scales)
    assert streamed.intercept == pytest.approx(batch.intercept)


def test_cross_section_sample_cap_is_global_across_input_chunks() -> None:
    chunks = [
        pd.DataFrame(
            {
                "symbol": [f"{part}{index}" for index in range(10)],
                "trade_date": ["2022-06-01"] * 10,
                "bar_time": ["093100000"] * 10,
                "label_net_return": np.arange(10, dtype=float),
            }
        )
        for part in ("A", "B", "C", "D")
    ]
    sample, metadata = _collect_cross_section_sample(
        lambda: chunks,
        rows_per_group=2,
        maximum_rows=100,
    )
    assert metadata["sample_rows"] == 2
    assert sample.groupby(["trade_date", "bar_time"]).size().tolist() == [2]


def test_streamed_score_file_matches_in_memory_metrics(tmp_path) -> None:
    rows = []
    for minute_index, bar_time in enumerate(("093100000", "093200000")):
        for symbol_index in range(6):
            rows.append(
                {
                    "symbol": f"{symbol_index:06d}.SH",
                    "trade_date": "2022-06-01",
                    "bar_time": bar_time,
                    "score": float(symbol_index + minute_index * 0.1),
                    "label_net_return": float(0.01 * (5 - symbol_index) + minute_index * 0.001),
                    "planned_exit_date": "2022-06-02",
                    "entry_bar_time": "093600000",
                    "entry_price": 10.0,
                    "entry_amount": 100_000.0,
                    "entry_executable": True,
                    "actual_exit_date": "2022-06-02",
                    "exit_amount": 100_000.0,
                    "label_observed": True,
                }
            )
    frame = pd.DataFrame(rows)
    path = tmp_path / "scores.parquet"
    metadata = _write_scored_period(
        lambda: [frame.iloc[:5], frame.iloc[5:]],
        lambda value: value["score"],
        path,
    )
    assert metadata["row_count"] == len(frame)
    streamed = _evaluate_score_file(path, top_k=3)
    expected = evaluate_scores(frame, score="score", target="label_net_return", top_k=3)
    for key in (
        "row_count",
        "group_count",
        "rank_ic_mean",
        "rank_ic_positive_fraction",
        "universe_row_mean_net_return",
        "universe_group_mean_net_return",
        "top_k_mean_net_return",
        "top_k_mean_excess_over_group_mean",
        "top_k_positive_excess_group_fraction",
        "daily_top_k_mean_net_return",
        "positive_day_fraction",
    ):
        assert streamed[key] == pytest.approx(expected[key])
    top = _load_top_scored(path, top_k=3)
    assert len(top) == 6


def test_score_file_coerces_invalid_values_and_matches_tie_metrics(tmp_path) -> None:
    rows = []
    for index, (score, target) in enumerate(
        [(1, 0.1), (1, 0.2), (2, 0.2), (3, 0.3), (3, 0.1), (4, 0.4)]
    ):
        rows.append(
            {
                "symbol": chr(65 + index),
                "trade_date": "2022-06-01",
                "bar_time": "093100000",
                "score": str(score),
                "label_net_return": str(target),
                "planned_exit_date": "2022-06-02",
                "entry_bar_time": "093600000",
                "entry_price": "10",
                "entry_amount": "100",
                "entry_executable": True,
                "actual_exit_date": "2022-06-02",
                "exit_amount": "100",
                "label_observed": True,
            }
        )
    frame = pd.DataFrame(rows)
    path = tmp_path / "scores.parquet"
    frame.to_parquet(path, index=False)
    expected_frame = frame.copy()
    expected_frame["score"] = pd.to_numeric(expected_frame["score"])
    expected_frame["label_net_return"] = pd.to_numeric(expected_frame["label_net_return"])
    expected = evaluate_scores(expected_frame)
    actual = _evaluate_score_file(path)
    assert actual["rank_ic_mean"] == pytest.approx(expected["rank_ic_mean"])
    assert actual["top_k_positive_excess_group_fraction"] == pytest.approx(
        expected["top_k_positive_excess_group_fraction"]
    )

    invalid = frame.copy()
    invalid.loc[0, "score"] = "not-a-number"
    invalid.loc[1, "label_net_return"] = "not-a-number"
    invalid.to_parquet(path, index=False)
    assert _evaluate_score_file(path)["row_count"] == len(invalid) - 2
    loaded = _load_top_scored(path, top_k=3)
    assert len(loaded) == 3
    assert set(loaded["symbol"]) == {"D", "E", "F"}


def test_multiple_score_writer_cleans_partials_on_empty_input(tmp_path) -> None:
    with pytest.raises(MinuteV2Error, match="stream_score_empty"):
        _write_multiple_scored_periods(
            lambda: [],
            {"rule": lambda frame: np.zeros(len(frame))},
            tmp_path,
        )
    assert not list(tmp_path.glob("*.partial"))


def test_multiple_score_writer_rolls_back_already_committed_files(tmp_path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "trade_date": ["2022-06-01", "2022-06-01"],
            "bar_time": ["093100000", "093100000"],
            "label_net_return": [0.01, 0.02],
        }
    )
    existing = {}
    for name in ("rule", "ridge"):
        target = tmp_path / f"{name}_test_scores.parquet"
        target.write_bytes(f"old-{name}".encode("ascii"))
        existing[name] = target.read_bytes()
    import quantlab.research.minute_v2.training as minute_v2_training

    original_training_replace = minute_v2_training.os.replace
    failing_target = (tmp_path / "ridge_test_scores.parquet").resolve()

    def flaky_replace(source, target):
        if (
            Path(target).resolve() == failing_target
            and str(source).endswith(".parquet.partial")
        ):
            raise OSError("injected commit failure")
        return original_training_replace(source, target)

    monkeypatch.setattr(minute_v2_training.os, "replace", flaky_replace)
    try:
        with pytest.raises(OSError, match="injected commit failure"):
            _write_multiple_scored_periods(
                lambda: [frame],
                {
                    "rule": lambda value: np.ones(len(value)),
                    "ridge": lambda value: np.zeros(len(value)),
                },
                tmp_path,
            )
    finally:
        monkeypatch.setattr(minute_v2_training.os, "replace", original_training_replace)
    for name, content in existing.items():
        assert (tmp_path / f"{name}_test_scores.parquet").read_bytes() == content
    assert not list(tmp_path.glob("*.partial"))
    assert not list(tmp_path.glob("*.rollback-*"))


def test_score_evaluation_excludes_infinite_scores_and_targets() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E", "F"],
            "trade_date": ["2022-06-01"] * 6,
            "bar_time": ["093100000"] * 6,
            "score": [6.0, 5.0, 4.0, 3.0, np.inf, 1.0],
            "label_net_return": [0.06, 0.05, 0.04, 0.03, 0.02, np.nan],
        }
    )
    result = evaluate_scores(frame, top_k=3)
    assert result["row_count"] == 4
    assert result["group_count"] == 1
    assert np.isfinite(result["top_k_mean_net_return"])


@pytest.mark.parametrize(
    "top_k",
    [0, -1, None, float("nan"), float("inf"), "invalid", 1.5, 3.0, True],
)
def test_score_evaluation_rejects_non_numeric_top_k(top_k) -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A"] * 5,
            "trade_date": ["2022-06-01"] * 5,
            "bar_time": ["093100000"] * 5,
            "score": [1.0] * 5,
            "label_net_return": [0.01] * 5,
        }
    )
    with pytest.raises(MinuteV2Error, match="top_k_invalid"):
        evaluate_scores(frame, top_k=top_k)


@pytest.mark.parametrize("top_k", [[1.5], [3.0], [True], "3", None])
def test_candidate_recall_rejects_non_integer_top_k(top_k) -> None:
    base = pd.DataFrame(
        [{"symbol": "A", "trade_date": "2022-06-01", "bar_time": "093100000"}]
    )
    outcomes = base.assign(label_return_5m=[0.01])
    with pytest.raises(MinuteV2Error, match="top_k_invalid"):
        audit_candidate_recall(base, base, outcomes, top_k=top_k)
