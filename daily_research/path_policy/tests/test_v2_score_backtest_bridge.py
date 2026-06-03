from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from daily_research.path_policy import v2_score_backtest_bridge as bridge


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_source_study(
    studies_root: Path,
    tag: str,
    *,
    dataset_id: str = "dataset",
    pool_view_id: str = "pool",
    feature_profile: str = "feature",
    score_shift: float = 0.0,
) -> None:
    study = studies_root / tag
    _write_json(
        study / "forecast_dataset_manifest.json",
        {
            "source_market_dataset_id": dataset_id,
            "source_pool_view_id": pool_view_id,
            "source_pool_view_kind": "rolling_liquidity_tradeable_mainboard",
            "source_pool_view_name": "rolling_liquid500_tradeable_mainboard_v2",
            "feature_profile": feature_profile,
            "feature_columns": ["raw_open_gap_1d", "amount_norm"],
            "feature_store_shape": [10, 3, 2],
            "label_semantics": {"label_semantics": "next_open_entry_to_future_open"},
        },
    )
    rows = []
    for date in ["2024-01-02", "2024-01-03"]:
        for stock, score, future, hit in [
            ("000001.SZ", 0.30, 0.03, 1),
            ("000002.SZ", 0.20, 0.02, 1),
            ("600000.SH", 0.10, -0.01, 0),
        ]:
            rows.append(
                {
                    "date": date,
                    "stock": stock,
                    "role": "test",
                    "pred_decision_score": score + score_shift,
                    "future_decision_score": future,
                    "pred_best_horizon": 5,
                    "future_best_horizon": 5,
                    "future_hit_label_5d": hit,
                    "history_valid_ratio": 1.0,
                }
            )
    pd.DataFrame(rows).to_csv(study / "forecast_predictions_test.csv", index=False)


def test_validate_source_study_accepts_summary_dataset_manifest(tmp_path: Path) -> None:
    studies = tmp_path / "studies"
    _write_source_study(studies, "seed11")
    study = studies / "seed11"
    manifest = json.loads((study / "forecast_dataset_manifest.json").read_text(encoding="utf-8"))
    (study / "forecast_dataset_manifest.json").unlink()
    _write_json(study / "study_summary.json", {"dataset_manifest": manifest})

    payload = bridge.validate_source_study(
        study,
        dataset_id="dataset",
        pool_view_id="pool",
        feature_profile="feature",
    )

    assert payload["status"] == "ok"
    assert payload["manifest_source"] == "study_summary_dataset_manifest"


def test_build_v2_score_bridge_writes_research_only_ensemble_panel(tmp_path: Path) -> None:
    studies = tmp_path / "studies"
    _write_source_study(studies, "seed7", score_shift=0.00)
    _write_source_study(studies, "seed11", score_shift=0.02)

    report = bridge.build_v2_score_bridge(
        output_root=tmp_path / "out",
        run_tag="unit_bridge",
        seed_study_tags=("seed7", "seed11"),
        studies_root=studies,
        dataset_id="dataset",
        pool_view_id="pool",
        feature_profile="feature",
        require_all_seeds=True,
        enforce_active_artifact_clean=False,
    )

    assert report["status"] == "completed"
    assert report["boundary"]["promotion_allowed"] is False
    assert report["boundary"]["active_execution_strategy_expected_diff"] == "none"
    ensemble_path = Path(report["bridge"]["ensemble_score_panel_csv"])
    backtest_path = Path(report["bridge"]["backtest_score_panel_csv"])
    assert ensemble_path.exists()
    assert backtest_path.exists()

    ensemble = pd.read_csv(ensemble_path)
    row = ensemble.loc[(ensemble["date"] == "2024-01-02") & (ensemble["stock"] == "000001.SZ")].iloc[0]
    assert row["score"] == pytest.approx(0.31)
    assert int(row["seed_count"]) == 2
    backtest_panel = pd.read_csv(backtest_path)
    assert list(backtest_panel.columns) == ["date", "stock", "score"]


def test_build_v2_score_bridge_blocks_manifest_mismatch(tmp_path: Path) -> None:
    studies = tmp_path / "studies"
    _write_source_study(studies, "seed7", pool_view_id="wrong_pool")

    report = bridge.build_v2_score_bridge(
        output_root=tmp_path / "out",
        run_tag="unit_bridge",
        seed_study_tags=("seed7",),
        studies_root=studies,
        dataset_id="dataset",
        pool_view_id="pool",
        feature_profile="feature",
        enforce_active_artifact_clean=False,
    )

    assert report["status"] == "blocked"
    assert "seed7:source_pool_view_id_mismatch" in report["blockers"]
    assert Path(tmp_path / "out" / "v2_score_backtest_bridge_report.json").exists()


def test_v2_score_bridge_backtest_command_is_lake_research_candidate(tmp_path: Path) -> None:
    studies = tmp_path / "studies"
    _write_source_study(studies, "seed7")

    report = bridge.build_v2_score_bridge(
        output_root=tmp_path / "out",
        run_tag="unit_bridge",
        seed_study_tags=("seed7",),
        studies_root=studies,
        dataset_id="dataset",
        pool_view_id="pool",
        feature_profile="feature",
        enforce_active_artifact_clean=False,
    )

    command = report["shared_backtest"]["command"]
    assert "--data-source" in command
    assert command[command.index("--data-source") + 1] == "lake"
    assert command[command.index("--lake-dataset-id") + 1] == "dataset"
    assert command[command.index("--score-threshold") + 1] == "-999.0"
    assert "--no-market-regime-filter" in command
    assert report["boundary"]["target_weight_semantics"].endswith("not_production_panel")


def test_v2_score_bridge_active_artifact_guard_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge, "_active_artifact_has_diff", lambda: True)

    with pytest.raises(ValueError, match="active_artifact_diff_blocker"):
        bridge.build_v2_score_bridge(output_root=tmp_path / "out")


def test_external_score_backtest_parser_accepts_lake(monkeypatch: pytest.MonkeyPatch) -> None:
    from daily_research.baseline import backtest_external_score_panel

    monkeypatch.setattr(sys, "argv", ["prog", "--data-source", "lake", "--lake-dataset-id", "dataset"])
    args = backtest_external_score_panel.parse_args()

    assert args.data_source == "lake"
    assert args.lake_dataset_id == "dataset"


def test_external_score_backtest_score_panel_honors_all_offset_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from daily_research.baseline import backtest_external_score_panel

    score_csv = tmp_path / "scores.csv"
    score_csv.write_text(
        "\n".join(
            [
                "date,stock,score",
                "2024-01-02,AAA,0.30",
                "2024-01-02,BBB,0.10",
                "2024-01-03,AAA,0.10",
                "2024-01-03,BBB,0.30",
                "2024-01-04,AAA,0.30",
                "2024-01-04,BBB,0.10",
                "2024-01-05,AAA,0.10",
                "2024-01-05,BBB,0.30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    columns = ["AAA", "BBB", "000300.SH"]
    close = pd.DataFrame(
        {
            "AAA": [10.0, 10.1, 10.2, 10.3],
            "BBB": [20.0, 20.1, 20.2, 20.3],
            "000300.SH": [4000.0, 4010.0, 4020.0, 4030.0],
        },
        index=dates,
    )
    raw = {
        "Open": close - 0.1,
        "High": close + 0.2,
        "Low": close - 0.2,
        "Close": close,
        "Volume": pd.DataFrame(1000.0, index=dates, columns=columns),
        "Amount": pd.DataFrame(10000.0, index=dates, columns=columns),
    }

    monkeypatch.setattr(backtest_external_score_panel, "load_raw_data_with_cache", lambda **_: (raw, {"cache_hit": False}))
    monkeypatch.setattr(
        backtest_external_score_panel,
        "build_filter_mask",
        lambda _bundle, _cfg: pd.DataFrame(True, index=dates, columns=["AAA", "BBB"]),
    )
    monkeypatch.setattr(
        backtest_external_score_panel,
        "compute_market_regime_state",
        lambda _benchmark_close, _cfg: pd.DataFrame(
            {"regime_on": [True] * len(dates), "quadrant": ["trend_up_low_vol"] * len(dates)},
            index=dates,
        ),
    )
    monkeypatch.setattr(
        backtest_external_score_panel,
        "summarize_backtest_by_month",
        lambda _equity, _actions: pd.DataFrame({"month": ["2024-01"], "excess_return": [0.0]}),
    )
    monkeypatch.setattr(
        backtest_external_score_panel,
        "summarize_monthly_diagnostics",
        lambda _summary, return_column: {"return_column": return_column, "month_count": 1},
    )

    def fake_backtest(**kwargs):
        equity = pd.DataFrame(
            {
                "portfolio_equity": [1.0, 1.01, 1.02, 1.03],
                "benchmark_equity": [1.0, 1.0, 1.01, 1.01],
                "excess_equity": [1.0, 1.01, 1.01, 1.02],
            },
            index=dates,
        )
        actions = pd.DataFrame({"date": dates, "stock": ["AAA"] * len(dates), "action": ["hold"] * len(dates)})
        metrics = {
            "annual_return": 0.0,
            "excess_annual_return": 0.0,
            "excess_sharpe": 0.0,
            "max_drawdown": 0.0,
        }
        return equity, actions, metrics

    monkeypatch.setattr(backtest_external_score_panel, "backtest", fake_backtest)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--score-panel-csv",
            str(score_csv),
            "--data-source",
            "lake",
            "--lake-dataset-id",
            "dataset",
            "--benchmark",
            "000300.SH",
            "--rebalance-freq",
            "2d",
            "--rebalance-offset-mode",
            "all",
            "--output-dir",
            str(tmp_path / "runs"),
            "--experiment-tag",
            "unit",
            "--no-market-regime-filter",
        ],
    )

    backtest_external_score_panel.main()
    metrics = json.loads((tmp_path / "runs" / "unit" / "metrics.json").read_text(encoding="utf-8"))

    assert metrics["bridge_mode"] == "score_panel_ensemble"
    assert metrics["rebalance_offset_mode"] == "all"
    assert metrics["rebalance_sleeve_count"] == 2
