from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    _manifest_schema_from_arrow,
    qdp_v2_root,
    write_active_manifest,
    write_dataset_manifest,
)
from quantlab.data.qdp_v2.minute_repair.candidate import (
    EXPECTED_SESSION_TIMES,
    PROVIDER_FIELDS,
    TargetBatch,
    _download_one_batch,
    evaluate_target_day,
    load_explicit_targets,
    load_priority_targets,
    load_provider_target_minutes,
    load_reusable_prior_batches,
    normalize_provider_minutes,
    plan_target_batches,
)
from quantlab.data.qdp_v2.minute_repair.install import (
    finish_existing_reaudits,
    stage_active_shard_replacements,
)
from quantlab.data.qdp_v2.minute_repair.local_source import (
    LOCAL_SOURCE_TAG,
    capture_local_candidate_batches,
    local_source_inventory,
    prefilter_local_candidates,
)


def _session(symbol: str = "600000.SH", trade_date: str = "2010-05-17") -> pd.DataFrame:
    times = sorted(EXPECTED_SESSION_TIMES)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": trade_date,
            "bar_time": times,
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 100.0,
            "amount": 1000.0,
            "source": "local_minute_zip",
            "adjusted_flag": "raw_unadjusted",
            "domain": [
                "market_opening_auction" if value == "093000000" else "market_intraday_1m"
                for value in times
            ],
            "active_shard_path": "fixture.parquet",
        }
    )


def _provider(local: pd.DataFrame) -> pd.DataFrame:
    return local.loc[:, ["symbol", "trade_date", "bar_time", "open", "high", "low", "close"]].copy()


def _target(**updates: object) -> dict[str, object]:
    target: dict[str, object] = {
        "symbol": "600000.SH",
        "trade_date": "2010-05-17",
        "d_open": 10.0,
        "d_high": 10.0,
        "d_low": 10.0,
        "d_close": 10.0,
        "agg_open": 10.0,
        "agg_high": 10.0,
        "agg_low": 10.0,
        "agg_close": 10.0,
        "price_reference_scale": 10.0,
        "exclude_open": False,
        "exclude_high": True,
        "exclude_low": False,
        "exclude_close": False,
    }
    target.update(updates)
    return target


def test_evaluate_target_day_repairs_only_the_corrupt_extreme_row() -> None:
    local = _session()
    row = local.index[local["bar_time"].eq("150000000")].item()
    local.loc[row, ["open", "high"]] = 100.0
    provider = _provider(local)
    provider.loc[row, ["open", "high", "low", "close"]] = 10.0

    decision, changes = evaluate_target_day(_target(), local, provider)

    assert decision["status"] == "accepted"
    assert decision["repaired_fields"] == "high"
    assert changes["bar_time"].tolist() == ["150000000"]
    assert changes[["open", "high", "low", "close"]].iloc[0].tolist() == [10.0, 10.0, 10.0, 10.0]
    assert changes[["volume", "amount"]].iloc[0].tolist() == [100.0, 1000.0]
    assert changes[["old_open", "old_high"]].iloc[0].tolist() == [100.0, 100.0]


def test_evaluate_target_day_rejects_a_shared_upstream_error() -> None:
    local = _session()
    row = local.index[local["bar_time"].eq("150000000")].item()
    local.loc[row, ["open", "high"]] = 100.0
    provider = _provider(local)

    decision, changes = evaluate_target_day(_target(), local, provider)

    assert decision["status"] == "rejected"
    assert decision["reason"] == "provider_does_not_clear_any_flagged_field"
    assert changes.empty


def test_evaluate_target_day_can_repair_one_field_and_keep_another_masked() -> None:
    local = _session()
    high_row = local.index[local["bar_time"].eq("140000000")].item()
    low_row = local.index[local["bar_time"].eq("141000000")].item()
    local.loc[high_row, ["open", "high", "close"]] = [10.0, 11.0, 10.0]
    local.loc[low_row, ["open", "low", "close"]] = [10.0, 9.0, 10.0]
    provider = _provider(local)
    provider.loc[low_row, ["open", "high", "low", "close"]] = 10.0
    target = _target(exclude_low=True)

    decision, changes = evaluate_target_day(target, local, provider)

    assert decision["status"] == "accepted"
    assert decision["repaired_fields"] == "low"
    assert decision["post_high"] == 11.0
    assert decision["post_low"] == 10.0
    assert changes["bar_time"].tolist() == ["141000000"]


def test_evaluate_target_day_requires_a_complete_exact_session() -> None:
    local = _session()
    provider = _provider(local).iloc[:-1].copy()

    decision, changes = evaluate_target_day(_target(), local, provider)

    assert decision["status"] == "rejected"
    assert decision["reason"] == "provider_session_row_count:240"
    assert changes.empty


def test_evaluate_target_day_rejects_an_empty_provider_session() -> None:
    decision, changes = evaluate_target_day(_target(), _session(), pd.DataFrame())

    assert decision["status"] == "rejected"
    assert decision["reason"] == "provider_session_row_count:0"
    assert changes.empty


def test_plan_target_batches_respects_calendar_day_limit() -> None:
    targets = pd.DataFrame(
        {
            "symbol": ["600000.SH"] * 4 + ["000001.SZ"],
            "trade_date": ["2010-01-01", "2010-01-15", "2010-01-31", "2010-02-01", "2010-03-01"],
        }
    )

    batches = plan_target_batches(targets, max_calendar_days=30)

    assert [(item.symbol, item.start_date, item.end_date) for item in batches] == [
        ("000001.SZ", "2010-03-01", "2010-03-01"),
        ("600000.SH", "2010-01-01", "2010-01-31"),
        ("600000.SH", "2010-02-01", "2010-02-01"),
    ]


def test_plan_target_batches_can_use_trading_day_limit() -> None:
    targets = pd.DataFrame(
        {
            "symbol": ["600000.SH"] * 4,
            "trade_date": ["2010-01-04", "2010-01-08", "2010-02-20", "2010-03-01"],
        }
    )
    trading_dates = ["2010-01-04", "2010-01-08", "2010-02-20", "2010-03-01"]

    batches = plan_target_batches(
        targets,
        trading_dates=trading_dates,
        max_trading_days=3,
    )

    assert [item.target_dates for item in batches] == [
        ("2010-01-04", "2010-01-08", "2010-02-20"),
        ("2010-03-01",),
    ]


def test_normalize_provider_minutes_sorts_and_maps_keys() -> None:
    frame = pd.DataFrame(
        {
            "ts_code": ["600000.SH", "600000.SH"],
            "trade_time": ["2010-01-04 09:31:00", "2010-01-04 09:30:00"],
            "open": [10, 9],
            "high": [10, 9],
            "low": [10, 9],
            "close": [10, 9],
            "vol": [100, 200],
            "amount": [1000, 1800],
        }
    )

    result = normalize_provider_minutes(frame, symbol="600000.SH")

    assert result["bar_time"].tolist() == ["093000000", "093100000"]
    assert result["provider_volume"].tolist() == [200.0, 100.0]


def test_download_batch_falls_back_to_exact_target_dates(tmp_path: Path) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def fetch(
            self,
            api_name: str,
            *,
            params: dict[str, str],
            fields: tuple[str, ...],
            retries: int = 4,
        ) -> pd.DataFrame:
            assert api_name == "stk_mins"
            assert fields == PROVIDER_FIELDS
            self.calls.append(params)
            if params["start_date"][:10] != params["end_date"][:10]:
                raise TimeoutError("fixture_range_timeout")
            trade_date = params["start_date"][:10]
            return pd.DataFrame(
                [["600000.SH", f"{trade_date} 09:30:00", 10, 10, 10, 10, 100, 1000]],
                columns=PROVIDER_FIELDS,
            )

    client = FakeClient()
    batch = TargetBatch(
        batch_id="fixture",
        symbol="600000.SH",
        start_date="2010-01-04",
        end_date="2010-01-06",
        target_dates=("2010-01-04", "2010-01-06"),
    )

    captured = _download_one_batch(
        client,  # type: ignore[arg-type]
        batch,
        raw_dir=tmp_path,
        api_url="https://example.invalid/api",
    )

    metadata = json.loads(Path(captured.metadata_path).read_text(encoding="utf-8"))
    assert len(client.calls) == 3
    assert metadata["request_strategy"] == "target_dates_fallback"
    assert len(metadata["actual_requests"]) == 2
    assert len(pd.read_parquet(captured.raw_path)) == 2


def test_prior_range_capture_can_be_reused_for_a_new_target_date(tmp_path: Path) -> None:
    class FakeClient:
        @staticmethod
        def fetch(
            api_name: str,
            *,
            params: dict[str, str],
            fields: tuple[str, ...],
            retries: int = 4,
        ) -> pd.DataFrame:
            assert api_name == "stk_mins"
            return pd.DataFrame(
                [
                    ["600000.SH", "2010-01-04 09:30:00", 10, 10, 10, 10, 100, 1000],
                    ["600000.SH", "2010-01-06 09:30:00", 11, 11, 11, 11, 200, 2200],
                ],
                columns=fields,
            )

    prior_run = (
        tmp_path
        / "data"
        / "qdp"
        / "source_archives"
        / "tushare_compatible"
        / "minute_repair"
        / "prior"
    )
    parent = TargetBatch(
        batch_id="parent",
        symbol="600000.SH",
        start_date="2010-01-04",
        end_date="2010-01-08",
        target_dates=("2010-01-04",),
    )
    captured = _download_one_batch(
        FakeClient(),  # type: ignore[arg-type]
        parent,
        raw_dir=prior_run / "raw",
        api_url="https://example.invalid/api",
    )
    (prior_run / "batches.json").write_text(
        json.dumps({"schema": "fixture", "batches": [captured.to_dict()]}),
        encoding="utf-8",
    )
    targets = pd.DataFrame({"symbol": ["600000.SH"], "trade_date": ["2010-01-06"]})

    reused = load_reusable_prior_batches(
        targets,
        run_dir=prior_run.parent / "current",
        workspace_root=tmp_path,
    )
    provider, sources = load_provider_target_minutes(reused)

    assert len(reused) == 1
    assert reused[0].target_dates == ("2010-01-06",)
    assert provider[["trade_date", "open"]].to_dict("records") == [
        {"trade_date": "2010-01-06", "open": 11.0}
    ]
    assert sources["provider_raw_sha256"].str.len().tolist() == [64]


def _quality_row(symbol: str, trade_date: str, *, high: float, daily_high: float, relative: float) -> dict[str, object]:
    row: dict[str, object] = {
        "symbol": symbol,
        "trade_date": trade_date,
        "agg_open": 10.0,
        "agg_high": high,
        "agg_low": 10.0,
        "agg_close": 10.0,
        "agg_volume": 100.0,
        "agg_amount": 1000.0,
        "d_open": 10.0,
        "d_high": daily_high,
        "d_low": 10.0,
        "d_close": 10.0,
        "d_volume": 100.0,
        "d_amount": 1000.0,
        "open_abs_error": 0.0,
        "high_abs_error": abs(high - daily_high),
        "low_abs_error": 0.0,
        "close_abs_error": 0.0,
        "price_reference_scale": max(high, daily_high),
        "volume_relative_error": 0.0,
        "amount_relative_error": 0.0,
        "price_max_abs_error": abs(high - daily_high),
        "price_max_relative_error": relative,
        "price_quality_class": "severe" if relative > 0.01 else "unreliable",
        "exclude_open": False,
        "exclude_high": True,
        "exclude_low": False,
        "exclude_close": False,
    }
    return row


def test_load_priority_targets_selects_envelope_and_large_errors(tmp_path: Path) -> None:
    quality = tmp_path / "data" / "qdp" / "source_archives" / "minute" / "quality" / "year=2010"
    quality.mkdir(parents=True)
    rows = pd.DataFrame(
        [
            _quality_row("600001.SH", "2010-01-04", high=10.2, daily_high=10.0, relative=0.02),
            _quality_row("600002.SH", "2010-01-05", high=9.0, daily_high=10.0, relative=0.10),
            _quality_row("600003.SH", "2010-01-06", high=9.9, daily_high=10.0, relative=0.01),
        ]
    )
    rows.to_parquet(quality / "daily_parity_material_mismatches.parquet", index=False)

    result = load_priority_targets(tmp_path, minimum_relative_error=0.05, include_known_probes=False)

    assert result["symbol"].tolist() == ["600001.SH", "600002.SH"]
    assert result.set_index("symbol")["selection_reason"].to_dict() == {
        "600001.SH": "outside_daily_price_envelope",
        "600002.SH": "relative_error_above_priority_threshold",
    }


def test_load_priority_targets_can_select_only_open_field_errors(tmp_path: Path) -> None:
    quality = tmp_path / "data" / "qdp" / "source_archives" / "minute" / "quality" / "year=2010"
    quality.mkdir(parents=True)
    open_error = _quality_row("600001.SH", "2010-01-04", high=10.0, daily_high=10.0, relative=0.08)
    open_error.update(
        {
            "agg_open": 9.0,
            "d_open": 10.0,
            "open_abs_error": 1.0,
            "price_max_abs_error": 1.0,
            "price_reference_scale": 10.0,
            "exclude_open": True,
            "exclude_high": False,
        }
    )
    high_only = _quality_row("600002.SH", "2010-01-05", high=9.0, daily_high=10.0, relative=0.10)
    pd.DataFrame([open_error, high_only]).to_parquet(
        quality / "daily_parity_material_mismatches.parquet", index=False
    )

    result = load_priority_targets(
        tmp_path,
        minimum_relative_error=0.05,
        include_known_probes=False,
        selection_mode="open",
    )

    assert result[["symbol", "selection_reason"]].to_dict("records") == [
        {"symbol": "600001.SH", "selection_reason": "open_relative_error_above_threshold"}
    ]


def test_load_priority_targets_can_bound_open_relative_error(tmp_path: Path) -> None:
    quality = tmp_path / "data" / "qdp" / "source_archives" / "minute" / "quality" / "year=2010"
    quality.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for index, open_relative_error in enumerate((0.01, 0.015, 0.02, 0.021), start=1):
        row = _quality_row(
            f"60000{index}.SH",
            f"2010-01-{index + 3:02d}",
            high=10.0,
            daily_high=10.0,
            relative=open_relative_error,
        )
        row.update(
            {
                "agg_open": 10.0 * (1.0 - open_relative_error),
                "d_open": 10.0,
                "open_abs_error": 10.0 * open_relative_error,
                "price_max_abs_error": 10.0 * open_relative_error,
                "price_reference_scale": 10.0,
                "exclude_open": True,
                "exclude_high": False,
            }
        )
        rows.append(row)
    pd.DataFrame(rows).to_parquet(
        quality / "daily_parity_material_mismatches.parquet", index=False
    )

    result = load_priority_targets(
        tmp_path,
        minimum_relative_error=0.01,
        maximum_relative_error=0.02,
        include_known_probes=False,
        selection_mode="open",
    )

    assert result["symbol"].tolist() == ["600002.SH", "600003.SH"]


def test_load_explicit_targets_filters_queue_and_uses_current_audit(tmp_path: Path) -> None:
    quality = tmp_path / "data" / "qdp" / "source_archives" / "minute" / "quality" / "year=2010"
    quality.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                **_quality_row(
                    "600001.SH", "2010-01-04", high=11.0, daily_high=10.0, relative=0.10
                ),
                "exclude_open": True,
            },
            _quality_row("600002.SH", "2010-01-05", high=11.0, daily_high=10.0, relative=0.10),
        ]
    ).to_parquet(quality / "daily_parity_material_mismatches.parquet", index=False)
    queue = tmp_path / "fallback.parquet"
    pd.DataFrame(
        [
            {
                "symbol": "600001.SH",
                "trade_date": "2010-01-04",
                "fallback_reasons": "local_source_target_day_missing",
                "severity": "severe",
            },
            {
                "symbol": "600002.SH",
                "trade_date": "2010-01-05",
                "fallback_reasons": "local_source_prices_nonpositive_or_invalid_ohlc",
                "severity": "unreliable",
            },
            {
                "symbol": "600003.SH",
                "trade_date": "2010-01-06",
                "fallback_reasons": "local_source_target_day_missing",
                "severity": "severe",
            },
        ]
    ).to_parquet(queue, index=False)

    result = load_explicit_targets(
        queue,
        tmp_path,
        include_reasons=("local_source_target_day_missing",),
        include_severities=("severe",),
        include_fields=("open",),
    )

    assert result[["symbol", "selection_reason"]].to_dict("records") == [
        {
            "symbol": "600001.SH",
            "selection_reason": "explicit_fallback:local_source_target_day_missing",
        }
    ]


def test_load_explicit_targets_ignores_flow_only_current_mismatches(tmp_path: Path) -> None:
    quality = tmp_path / "data" / "qdp" / "source_archives" / "minute" / "quality" / "year=2010"
    quality.mkdir(parents=True)
    price_row = _quality_row(
        "600001.SH", "2010-01-04", high=11.0, daily_high=10.0, relative=0.10
    )
    flow_only_row = _quality_row(
        "600002.SH", "2010-01-05", high=10.0, daily_high=10.0, relative=0.0
    )
    flow_only_row.update(
        {
            "volume_relative_error": 0.25,
            "price_quality_class": "reliable",
            "exclude_high": False,
        }
    )
    pd.DataFrame([price_row, flow_only_row]).to_parquet(
        quality / "daily_parity_material_mismatches.parquet", index=False
    )
    queue = tmp_path / "fallback.parquet"
    pd.DataFrame(
        [
            {"symbol": "600001.SH", "trade_date": "2010-01-04"},
            {"symbol": "600002.SH", "trade_date": "2010-01-05"},
        ]
    ).to_parquet(queue, index=False)

    result = load_explicit_targets(queue, tmp_path)

    assert result["symbol"].tolist() == ["600001.SH"]


def test_load_priority_targets_can_select_high_low_masks(tmp_path: Path) -> None:
    quality = tmp_path / "data" / "qdp" / "source_archives" / "minute" / "quality" / "year=2010"
    quality.mkdir(parents=True)
    high = _quality_row("600001.SH", "2010-01-04", high=11.0, daily_high=10.0, relative=0.10)
    low = _quality_row("600002.SH", "2010-01-05", high=10.0, daily_high=10.0, relative=0.10)
    low.update(
        {
            "agg_low": 8.0,
            "d_low": 9.0,
            "low_abs_error": 1.0,
            "price_max_abs_error": 1.0,
            "exclude_high": False,
            "exclude_low": True,
        }
    )
    open_only = _quality_row("600003.SH", "2010-01-06", high=10.0, daily_high=10.0, relative=0.10)
    open_only.update(
        {
            "agg_open": 9.0,
            "d_open": 10.0,
            "open_abs_error": 1.0,
            "price_max_abs_error": 1.0,
            "exclude_open": True,
            "exclude_high": False,
        }
    )
    pd.DataFrame([high, low, open_only]).to_parquet(
        quality / "daily_parity_material_mismatches.parquet", index=False
    )

    result = load_priority_targets(
        tmp_path,
        include_known_probes=False,
        selection_mode="high-low",
    )

    assert result[["symbol", "selection_reason"]].to_dict("records") == [
        {"symbol": "600001.SH", "selection_reason": "high_quality_mask"},
        {"symbol": "600002.SH", "selection_reason": "low_quality_mask"},
    ]


def test_local_source_inventory_ignores_parenthesized_duplicates(tmp_path: Path) -> None:
    source = tmp_path / "stock_1min"
    source.mkdir()
    pd.DataFrame({"value": [1]}).to_parquet(source / "600000.SH.parquet", index=False)
    pd.DataFrame({"value": [1]}).to_parquet(source / "600000.SH(1).parquet", index=False)

    files, inventory = local_source_inventory(source)

    assert files == {"600000.SH": source / "600000.SH.parquet"}
    assert inventory["base_file_count"] == 1
    assert inventory["ignored_duplicate_files"] == ["600000.SH(1).parquet"]


def test_prefilter_local_candidates_rejects_missing_and_accepts_improving_open() -> None:
    targets = pd.DataFrame(
        [
            {
                **_target(
                    agg_open=9.0,
                    d_open=10.0,
                    exclude_open=True,
                    exclude_high=False,
                ),
                "selection_reason": "fixture",
            },
            {
                **_target(
                    symbol="600001.SH",
                    trade_date="2010-05-18",
                    agg_open=9.0,
                    d_open=10.0,
                    exclude_open=True,
                    exclude_high=False,
                ),
                "selection_reason": "fixture",
            },
        ]
    )
    aggregates = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "trade_date": ["2010-05-17"],
            "row_count": [241],
            "unique_time_count": [241],
            "bad_time_count": [0],
            "bad_price_row_count": [0],
            "ext_open": [10.0],
            "ext_high": [10.0],
            "ext_low": [10.0],
            "ext_close": [10.0],
            "ext_volume": [100.0],
            "ext_amount": [1000.0],
        }
    )

    candidates, rejected, evidence = prefilter_local_candidates(targets, aggregates)

    assert candidates[["symbol", "trade_date"]].to_dict("records") == [
        {"symbol": "600000.SH", "trade_date": "2010-05-17"}
    ]
    assert rejected[["symbol", "reason"]].to_dict("records") == [
        {"symbol": "600001.SH", "reason": "local_source_target_day_missing"}
    ]
    assert evidence["status"].tolist() == ["candidate", "rejected"]


def test_capture_local_candidate_batch_and_custom_source_tag(tmp_path: Path) -> None:
    source = tmp_path / "stock_1min"
    source.mkdir()
    local = _session()
    raw = pd.DataFrame(
        {
            "ts_code": local["symbol"],
            "open": local["open"],
            "high": local["high"],
            "low": local["low"],
            "close": local["close"],
            "vol": local["volume"],
            "amount": local["amount"],
            "trade_date": pd.to_datetime(local["trade_date"]),
            "trade_time": pd.to_datetime(
                local["trade_date"] + " " + local["bar_time"].str[:6],
                format="%Y-%m-%d %H%M%S",
            ),
        }
    )
    raw.to_parquet(source / "600000.SH.parquet", index=False)
    targets = pd.DataFrame(
        {"symbol": ["600000.SH"], "trade_date": ["2010-05-17"]}
    )

    batches, capture = capture_local_candidate_batches(
        targets,
        source_root=source,
        raw_dir=tmp_path / "capture",
    )
    provider, _ = load_provider_target_minutes(batches)
    broken = local.copy()
    row = broken.index[broken["bar_time"].eq("150000000")].item()
    broken.loc[row, ["open", "high"]] = 100.0
    decision, changes = evaluate_target_day(
        _target(),
        broken,
        provider,
        source_tag=LOCAL_SOURCE_TAG,
    )

    assert capture["new_batches"] == 1
    assert capture["raw_rows"] == 241
    assert decision["status"] == "accepted"
    assert changes["source"].unique().tolist() == [LOCAL_SOURCE_TAG]


def test_stage_active_shard_replacement_preserves_keys_flow_and_schema(tmp_path: Path) -> None:
    domain = "market_intraday_1m"
    dataset_id = "market_intraday_1m__fixture"
    root = qdp_v2_root(tmp_path)
    path = root / "datasets" / domain / dataset_id / "shards" / "part.parquet"
    path.parent.mkdir(parents=True)
    frame = _session().loc[lambda x: x["bar_time"].isin(["093100000", "150000000"])]
    frame = frame.loc[
        :, ["symbol", "trade_date", "bar_time", "open", "high", "low", "close", "volume", "amount", "source", "adjusted_flag"]
    ].reset_index(drop=True)
    frame.to_parquet(path, index=False)
    write_dataset_manifest(
        root,
        DatasetManifest(
            dataset_id=dataset_id,
            domain=domain,
            layer="raw",
            frequency="1m",
            contract_version="fixture_v1",
            primary_key=["symbol", "trade_date", "bar_time"],
            start_date="2010-05-17",
            end_date="2010-05-17",
            row_count=len(frame),
            shards=[
                ShardManifestEntry(
                    path=path.relative_to(root).as_posix(),
                    row_count=len(frame),
                    start_date="2010-05-17",
                    end_date="2010-05-17",
                    file_size=path.stat().st_size,
                )
            ],
            source={"provider": "fixture"},
            quality={"primary_key_unique": True},
            schema=_manifest_schema_from_arrow(pq.read_schema(path)),
        ),
    )
    write_active_manifest(root, {"version": 2, "datasets": {domain: dataset_id}})
    selected = frame.loc[frame["bar_time"].eq("150000000")].copy()
    for field in ("open", "high", "low", "close"):
        selected[f"old_{field}"] = selected[field]
    selected[["open", "high"]] = 9.5
    selected["source"] = "local_minute_zip+tushare_compatible_price_repair"
    selected["domain"] = domain
    selected["active_shard_path"] = str(path)

    records = stage_active_shard_replacements(selected, run_dir=tmp_path / "run", workspace_root=tmp_path)

    assert len(records) == 1
    prepared = pd.read_parquet(records[0]["prepared_path"]).sort_values("bar_time").reset_index(drop=True)
    original = frame.sort_values("bar_time").reset_index(drop=True)
    assert pq.read_schema(records[0]["prepared_path"]).equals(pq.read_schema(path), check_metadata=False)
    assert prepared[["symbol", "trade_date", "bar_time", "volume", "amount"]].equals(
        original[["symbol", "trade_date", "bar_time", "volume", "amount"]]
    )
    repaired = prepared.loc[prepared["bar_time"].eq("150000000")].iloc[0]
    assert repaired["open"] == 9.5
    assert repaired["high"] == 9.5


def test_finish_existing_reaudit_is_idempotent_and_run_scoped(tmp_path: Path) -> None:
    year = 2010
    run = tmp_path / "run_abc"
    live = tmp_path / "data" / "qdp" / "source_archives" / "minute" / "quality" / f"year={year}"
    backup = run / "backup_pre_repair" / "quality" / f"year={year}"
    temporary = run / "reaudit" / f"year={year}"
    for directory in (live, temporary):
        directory.mkdir(parents=True)
    audit = {
        "quality": {
            "common_stock_days": 2,
            "open_exact_rate": 0.5,
            "session_feature_exclusion_rows": 1,
            "price_over_five_cent_rows": 2,
            "price_warning_rows": 0,
            "price_unreliable_rows": 2,
            "price_severe_rows": 1,
            "minute_feature_exclusion_rows": 2,
            "total_minute_feature_exclusion_rows": 3,
        }
    }
    (live / "audit.json").write_text(__import__("json").dumps(audit), encoding="utf-8")
    material = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "trade_date": ["2010-01-04"],
            "price_max_abs_error": [0.2],
            "price_max_relative_error": [0.02],
        }
    )
    exclusions = pd.DataFrame(
        {"symbol": ["600000.SH"], "trade_date": ["2010-01-04"], "severity": ["severe"]}
    )
    missing = pd.DataFrame({"symbol": ["600001.SH"], "trade_date": ["2010-01-04"]})
    artifacts = {
        "daily_parity_material_mismatches.parquet": material,
        "minute_feature_exclusions.parquet": exclusions,
        "daily_reference_missing_minute.parquet": missing,
    }
    for name, frame in artifacts.items():
        frame.to_parquet(live / name, index=False)
        frame.to_parquet(temporary / name, index=False)
    changes = pd.DataFrame({"trade_date": ["2010-01-04"]})
    decisions = pd.DataFrame(
        {
            "trade_date": ["2010-01-04"],
            "status": ["accepted"],
            "baseline_open": [9.0],
            "post_open": [10.0],
            "daily_open": [10.0],
            "baseline_high": [10.0],
            "post_high": [10.0],
            "daily_high": [10.0],
            "baseline_low": [9.0],
            "post_low": [9.0],
            "daily_low": [9.0],
            "baseline_close": [9.5],
            "post_close": [9.5],
            "daily_close": [9.5],
        }
    )

    first = finish_existing_reaudits(changes, decisions, run_dir=run, workspace_root=tmp_path)
    second = finish_existing_reaudits(changes, decisions, run_dir=run, workspace_root=tmp_path)

    updated = __import__("json").loads((live / "audit.json").read_text(encoding="utf-8"))
    assert first[0]["status"] == "completed_from_existing_artifacts"
    assert second[0]["status"] == "already_complete"
    assert updated["last_targeted_minute_repair_run"] == run.name
    assert updated["quality"]["price_unreliable_rows"] == 1
    assert updated["quality"]["total_minute_feature_exclusion_rows"] == 2
    assert updated["quality"]["open_exact_rate"] == 1.0
    assert (backup / "audit.json").is_file()
    assert all((backup / name).is_file() for name in artifacts)
