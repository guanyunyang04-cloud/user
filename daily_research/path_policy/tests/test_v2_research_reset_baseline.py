from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_data_platform.lake import ResearchDataLake
from quant_data_platform.lake.pool_views import PoolViewSpec, build_pool_view_from_policy_bundle
from daily_research.path_policy import v2_research_reset_baseline as v2


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_v2_training_tasks_use_strict_pool_and_amount_checked_profile() -> None:
    tasks = v2.build_v2_training_tasks(output_root=Path("anchor"))

    assert [task["seed"] for task in tasks] == [7, 11, 19]
    assert tasks[0]["source_pool_view_id"] == v2.V2_STRICT_POOL_VIEW_ID
    assert tasks[0]["source_status_sidecar_dataset_id"] == v2.V2_STATUS_SIDECAR_ID
    assert "--pool-view-id" in tasks[0]["command"]
    assert tasks[0]["command"][tasks[0]["command"].index("--pool-view-id") + 1] == v2.V2_STRICT_POOL_VIEW_ID
    assert tasks[0]["command"][tasks[0]["command"].index("--forecast-feature-profile") + 1] == v2.FEATURE_PROFILE
    assert "--forecast-memmap-manifest" not in tasks[0]["command"]
    assert "--forecast-memmap-manifest" in tasks[1]["command"]
    assert tasks[1]["command"][tasks[1]["command"].index("--forecast-memmap-manifest") + 1].endswith(
        "forecast_dataset_manifest.json"
    )
    assert tasks[0]["active_execution_strategy_expected_diff"] == "none"


def test_validate_v2_strict_pool_records_shortfall_and_filters_status(tmp_path: Path, monkeypatch) -> None:
    dates = pd.date_range("2026-01-05", periods=30, freq="B")
    stocks = ["000001.SZ", "000002.SZ", "600000.SH", "300001.SZ", "688001.SH"]
    close = pd.DataFrame(10.0, index=dates, columns=stocks)
    amount = pd.DataFrame(
        {
            "000001.SZ": [100000.0 + idx for idx in range(len(dates))],
            "000002.SZ": [90000.0 + idx for idx in range(len(dates))],
            "600000.SH": [80000.0 + idx for idx in range(len(dates))],
            "300001.SZ": [300000.0 + idx for idx in range(len(dates))],
            "688001.SH": [290000.0 + idx for idx in range(len(dates))],
        },
        index=dates,
    )
    lake = ResearchDataLake(tmp_path)
    market_record = lake.save_market_data_bundle(
        spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "synthetic"},
        market_frames={
            "Open": close - 0.1,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": pd.DataFrame(1000.0, index=dates, columns=stocks),
            "Amount": amount,
        },
        benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
        membership_frame=pd.DataFrame(True, index=dates, columns=stocks),
        feature_frames={"score_none": close * 0.0},
        source="synthetic",
    )
    status_rows = []
    for date in dates:
        for stock in stocks:
            status_rows.append(
                {
                    "trade_date": date.strftime("%Y-%m-%d"),
                    "symbol": stock,
                    "is_listed_on_date": True,
                    "is_mainboard": stock in {"000001.SZ", "000002.SZ", "600000.SH"},
                    "is_common_a_share": True,
                    "is_st": stock == "000002.SZ",
                    "is_suspended": False,
                    "is_delisted": stock == "600000.SH",
                    "is_tradeable": stock == "000001.SZ",
                    "has_bar": True,
                    "reject_reason": "" if stock == "000001.SZ" else "blocked",
                    "source": "unit",
                }
            )
    sidecar = lake.save_domain_dataset(
        domain="v2_status_sidecar",
        frame=pd.DataFrame(status_rows),
        spec={
            "dataset": "data_platform_v2_status_sidecar",
            "source_market_dataset_id": market_record.dataset_id,
            "start_date": "2026-01-05",
            "end_date": "2026-02-13",
        },
        source="unit",
    )
    strict_pool = build_pool_view_from_policy_bundle(
        lake=lake,
        spec=PoolViewSpec(
            source_market_dataset_id=market_record.dataset_id,
            view_kind="rolling_liquidity_tradeable_mainboard",
            view_name="rolling_liquid2_tradeable_mainboard_v2",
            pool_name="liquid2",
            start_date="2026-01-05",
            end_date="2026-02-13",
            rebalance_every_days=5,
            adv_window=2,
            exclude_symbol_prefixes=("300", "301", "688", "689"),
            status_sidecar_dataset_id=sidecar.dataset_id,
            require_tradeable=True,
        ),
    )
    monkeypatch.setattr(v2, "DATASET_ID", market_record.dataset_id)
    monkeypatch.setattr(v2, "V2_STATUS_SIDECAR_ID", sidecar.dataset_id)
    monkeypatch.setattr(v2, "V2_STRICT_POOL_VIEW_ID", strict_pool.dataset_id)

    payload = v2.validate_v2_strict_pool(data_lake_root=tmp_path)

    assert payload["status"] == "ok"
    assert payload["daily_member_count_min"] == 1
    assert payload["daily_shortfall_count"] == len(dates)
    assert "daily_member_count_below_500_on_some_dates" in payload["diagnostics"]
    assert payload["active_excluded_prefix_counts"] == {"300": 0, "301": 0, "688": 0, "689": 0}
    assert payload["active_not_tradeable_rows"] == 0
    assert payload["active_is_st_rows"] == 0


def test_v2_gate_status_reuses_strict_stage28_thresholds(tmp_path: Path) -> None:
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

    payload = v2.v2_gate_status(tmp_path)

    assert payload["status"] == "pass"
    assert all(payload["checks"].values())


def test_validate_v2_seed_manifest_blocks_alpha_prior_columns(tmp_path: Path) -> None:
    manifest = {
        "source_market_dataset_id": v2.DATASET_ID,
        "source_pool_view_id": v2.V2_STRICT_POOL_VIEW_ID,
        "feature_profile": v2.FEATURE_PROFILE,
        "feature_columns": ["raw_open_gap_1d", "alpha_prior_score_raw", "score_rank_pct"],
        "feature_profile_audit": {"amount_unit": {"amount_unit_policy": "as_is"}},
    }
    path = tmp_path / "forecast_dataset_manifest.json"
    _write_json(path, manifest)

    payload = v2.validate_v2_seed_manifest(path)

    assert payload["status"] == "blocked"
    assert "alpha_prior_or_score_columns_present" in payload["blockers"]
    assert payload["alpha_like_feature_count"] == 2


def test_write_v2_task_list_is_research_only(tmp_path: Path) -> None:
    path = v2.write_v2_task_list(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["research_program"] == v2.RESEARCH_PROGRAM
    assert payload["training_task_count"] == 3
    assert payload["training_tasks"][0]["promotion_allowed"] is False
    assert payload["training_tasks"][0]["active_execution_strategy_expected_diff"] == "none"
