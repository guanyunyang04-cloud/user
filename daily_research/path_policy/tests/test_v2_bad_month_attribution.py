from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from daily_research.path_policy import v2_bad_month_attribution as attribution


def _write_backtest_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"month": "2024-01", "portfolio_return": 0.01, "benchmark_return": 0.00, "excess_return": 0.01, "total_trading_cost_return": 0.001},
            {"month": "2024-02", "portfolio_return": -0.05, "benchmark_return": 0.01, "excess_return": -0.06, "total_trading_cost_return": 0.002},
        ]
    ).to_csv(root / "monthly_backtest_summary.csv", index=False)
    pd.DataFrame(
        [
            {"date": "2024-01-31", "stock": "AAA", "target_weight": 0.5},
            {"date": "2024-01-31", "stock": "BBB", "target_weight": 0.5},
            {"date": "2024-02-01", "stock": "AAA", "target_weight": 0.5},
            {"date": "2024-02-01", "stock": "BBB", "target_weight": 0.5},
            {"date": "2024-02-02", "stock": "AAA", "target_weight": 0.0},
            {"date": "2024-02-02", "stock": "BBB", "target_weight": 1.0},
        ]
    ).to_csv(root / "aligned_daily_target_weight_panel.csv", index=False)
    pd.DataFrame(
        [
            {"date": "2024-01-31", "stock": "AAA", "score": 0.9},
            {"date": "2024-01-31", "stock": "BBB", "score": 0.1},
            {"date": "2024-02-01", "stock": "AAA", "score": 0.8},
            {"date": "2024-02-01", "stock": "BBB", "score": 0.2},
            {"date": "2024-02-02", "stock": "AAA", "score": 0.3},
            {"date": "2024-02-02", "stock": "BBB", "score": 0.7},
        ]
    ).to_csv(root / "aligned_daily_score_panel.csv", index=False)


def test_bad_month_attribution_selects_worst_month_and_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backtest_dir = tmp_path / "backtest"
    _write_backtest_fixture(backtest_dir)

    def fake_close(**kwargs):
        dates = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-02"])
        close = pd.DataFrame({"AAA": [100.0, 90.0, 81.0], "BBB": [100.0, 110.0, 121.0]}, index=dates)
        return close

    monkeypatch.setattr(attribution, "_load_close_panel", fake_close)

    report = attribution.build_bad_month_attribution(
        run_tag="unit_attr",
        backtest_dir=backtest_dir,
        output_root=tmp_path / "out",
        dataset_id="dataset",
        pool_view_id="pool",
        enforce_active_artifact_clean=False,
    )

    assert report["status"] == "completed"
    assert report["month"] == "2024-02"
    assert report["summary"]["row_count"] == 4
    assert report["summary"]["recent_return_window"] == 5
    assert report["summary"]["volatility_window"] == 20
    assert report["boundary"]["promotion_allowed"] is False
    assert report["boundary"]["risk_state_method"] == "previous_close_derived_recent_return_volatility_reversal"
    assert Path(report["outputs"]["contribution_csv"]).exists()
    assert Path(report["outputs"]["recent_return_bucket_summary_csv"]).exists()
    assert Path(report["outputs"]["volatility_bucket_summary_csv"]).exists()
    assert Path(report["outputs"]["reversal_bucket_summary_csv"]).exists()
    assert Path(report["outputs"]["score_volatility_bucket_summary_csv"]).exists()
    contributions = pd.read_csv(report["outputs"]["contribution_csv"])
    assert {"recent_return", "volatility", "reversal", "recent_return_bucket", "volatility_bucket", "reversal_bucket"}.issubset(contributions.columns)
    assert report["top_negative_contributors"][0]["stock"] == "AAA"


def test_bad_month_attribution_active_artifact_guard_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attribution, "_active_artifact_has_diff", lambda: True)

    with pytest.raises(ValueError, match="active_artifact_diff_blocker"):
        attribution.build_bad_month_attribution(output_root=tmp_path / "out")
