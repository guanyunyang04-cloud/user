from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from daily_research.data_lake import ResearchDataLake
from daily_research.data_lake.pool_views import PoolViewSpec, build_pool_view_from_policy_bundle
from daily_research.path_policy import mainboard_rebuild_baseline as mainboard


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_mainboard_training_tasks_use_corrected_pool_and_seed_manifest_reuse() -> None:
    tasks = mainboard.build_mainboard_training_tasks(output_root=Path("anchor"))

    assert [task["seed"] for task in tasks] == [7, 11, 19]
    assert tasks[0]["source_pool_view_id"] == mainboard.CORRECTED_POOL_VIEW_ID
    assert "--pool-view-id" in tasks[0]["command"]
    assert tasks[0]["command"][tasks[0]["command"].index("--pool-view-id") + 1] == mainboard.CORRECTED_POOL_VIEW_ID
    assert "--forecast-memmap-manifest" not in tasks[0]["command"]
    assert "--forecast-memmap-manifest" in tasks[1]["command"]
    assert tasks[1]["command"][tasks[1]["command"].index("--forecast-memmap-manifest") + 1].endswith(
        "forecast_dataset_manifest.json"
    )
    assert tasks[0]["active_execution_strategy_expected_diff"] == "none"


def test_validate_corrected_pool_view_rejects_excluded_board_prefixes(tmp_path: Path, monkeypatch) -> None:
    dates = pd.date_range("2026-01-05", periods=30, freq="B")
    mainboard_symbols = [f"600{i:03d}.SH" for i in range(500)]
    excluded_symbols = ["300001.SZ", "301001.SZ", "688001.SH", "689001.SH"]
    stocks = [*mainboard_symbols, *excluded_symbols]
    close = pd.DataFrame(10.0, index=dates, columns=stocks)
    amount = pd.DataFrame(
        {stock: [10000.0 + pos for pos in range(len(dates))] for stock in stocks},
        index=dates,
    )
    market_frames = {
        "Open": close - 0.1,
        "High": close + 0.2,
        "Low": close - 0.2,
        "Close": close,
        "Volume": pd.DataFrame(1000.0, index=dates, columns=stocks),
        "Amount": amount,
    }
    lake = ResearchDataLake(tmp_path)
    market_record = lake.save_market_data_bundle(
        spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
        market_frames=market_frames,
        benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
        membership_frame=pd.DataFrame(True, index=dates, columns=stocks),
        feature_frames={"score_none": close * 0.0, "score_v2": close * 0.0 + 0.1},
        source="synthetic",
    )
    view = build_pool_view_from_policy_bundle(
        lake=lake,
        spec=PoolViewSpec(
            source_market_dataset_id=market_record.dataset_id,
            view_kind="rolling_liquidity",
            view_name="rolling_liquid500_mainboard",
            pool_name="liquid500",
            start_date="2026-01-05",
            end_date="2026-02-13",
            rebalance_every_days=5,
            adv_window=2,
            exclude_symbol_prefixes=("300", "301", "688", "689"),
        ),
    )
    monkeypatch.setattr(mainboard, "DATASET_ID", market_record.dataset_id)
    monkeypatch.setattr(mainboard, "CORRECTED_POOL_VIEW_ID", view.dataset_id)

    payload = mainboard.validate_corrected_pool_view(pool_view_id=view.dataset_id, data_lake_root=tmp_path)

    assert payload["status"] == "ok"
    assert payload["active_universe_size"] == 500
    assert payload["daily_member_count_min"] == 500
    assert payload["daily_member_count_max"] == 500
    assert payload["active_excluded_prefix_total"] == 0


def test_validate_mainboard_memmap_manifest_checks_lineage_and_next_open(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mainboard, "DATASET_ID", "policy_input_bundle__unit")
    monkeypatch.setattr(mainboard, "CORRECTED_POOL_VIEW_ID", "policy_pool_view__unit")
    manifest_path = tmp_path / "forecast_dataset_manifest.json"
    feature_columns = [f"f{i}" for i in range(116)]
    _write_json(
        manifest_path,
        {
            "source_market_dataset_id": "policy_input_bundle__unit",
            "source_pool_view_id": "policy_pool_view__unit",
            "source_pool_view_kind": "rolling_liquidity",
            "source_pool_view_name": "rolling_liquid500_mainboard",
            "feature_profile": "raw_kline_context_no_alpha_prior_v1",
            "feature_count_before_cap": 116,
            "feature_count_after_cap": 116,
            "feature_store_shape": [1699, 500, 116],
            "feature_columns": feature_columns,
            "feature_group_counts": {"state": 59, "raw_kline": 15},
            "sample_count_by_role": {"train": 10, "validation": 5, "test": 5},
            "label_semantics": {"label_semantics": "next_open_entry_to_future_open"},
            "execution_mode": "next_open",
            "next_open_label_extra_trading_day": 1,
            "stock_values": ["000001.SZ", "600000.SH"],
        },
    )

    payload = mainboard.validate_mainboard_memmap_manifest(manifest_path)

    assert payload["status"] == "ok"
    assert payload["feature_count"] == 116
    assert payload["feature_store_shape"] == [1699, 500, 116]
    assert payload["sample_count_by_role"]["train"] == 10


def test_feature_schema_diff_keeps_unresolved_when_legacy156_not_found(tmp_path: Path) -> None:
    manifest_path = tmp_path / "current" / "forecast_dataset_manifest.json"
    _write_json(
        manifest_path,
        {
            "feature_store_shape": [1699, 500, 116],
            "feature_columns": [f"f{i}" for i in range(116)],
        },
    )

    payload = mainboard.build_feature_schema_diff(
        current_manifest_path=manifest_path,
        search_roots=(tmp_path / "missing",),
    )

    assert payload["status"] == "legacy156_not_recovered"
    assert payload["feature_schema_shift"] == "unresolved_root_cause"
    assert payload["legacy156_profile_allowed"] is False


def test_corrected_gate_status_reuses_stage28_gate_thresholds(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "role": "test",
                "score_name": "pred_decision_score",
                "seed_count": 3,
                "rank_ic_min": 0.05,
                "spread_min": 0.02,
                "hit_lift_min": 0.01,
                "monthly_positive_rate_mean": 0.8,
                "negative_month_count_max": 2,
                "thirty_d_concentration_mean": 0.70,
            }
        ]
    ).to_csv(tmp_path / "profile_aggregate.csv", index=False)

    payload = mainboard.corrected_gate_status(tmp_path)

    assert payload["status"] == "pass"
    assert all(payload["checks"].values())
