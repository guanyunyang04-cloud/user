from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from daily_research.path_policy import v2_state_sizing_matrix as sizing


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_source_backtest(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        "date,stock,target_weight",
        "2024-01-02,AAA.SH,0.5",
        "2024-01-02,BBB.SH,0.5",
        "2024-01-03,AAA.SH,0.6",
        "2024-01-03,BBB.SH,0.4",
        "2024-01-04,AAA.SH,0.7",
        "2024-01-04,BBB.SH,0.3",
    ]
    (root / "aligned_daily_target_weight_panel.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    score_rows = [
        "date,stock,score",
        "2024-01-02,AAA.SH,0.9",
        "2024-01-02,BBB.SH,0.8",
        "2024-01-03,AAA.SH,0.7",
        "2024-01-03,BBB.SH,0.6",
        "2024-01-04,AAA.SH,0.5",
        "2024-01-04,BBB.SH,0.4",
    ]
    (root / "aligned_daily_score_panel.csv").write_text("\n".join(score_rows) + "\n", encoding="utf-8")
    equity_rows = [
        "date,excess_equity",
        "2024-01-02,1.00",
        "2024-01-03,0.96",
        "2024-01-04,0.95",
    ]
    (root / "equity_curve.csv").write_text("\n".join(equity_rows) + "\n", encoding="utf-8")


def test_build_state_scale_flags_drawdown_stress() -> None:
    equity = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "excess_equity": [1.0, 0.97, 0.95],
        }
    )

    scale, meta = sizing.build_state_scale(
        equity,
        scale_mode="drawdown",
        stress_scale=0.5,
        drawdown_threshold=0.02,
        monthly_loss_threshold=0.01,
        min_scale=0.0,
    )

    assert scale.iloc[0] == 1.0
    assert scale.iloc[-1] == 0.5
    assert meta["stress_day_count"] == 2
    assert meta["min_observed_drawdown"] < -0.04


def test_apply_state_sizing_scales_target_weight_rows() -> None:
    target = pd.DataFrame(
        {"AAA.SH": [0.6, 0.6], "BBB.SH": [0.4, 0.4]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    scale = pd.Series([1.0, 0.5], index=target.index)

    out = sizing.apply_state_sizing(target, scale)

    assert out.loc[pd.Timestamp("2024-01-02")].sum() == pytest.approx(1.0)
    assert out.loc[pd.Timestamp("2024-01-03")].sum() == pytest.approx(0.5)


def test_state_sizing_matrix_dry_run_writes_target_panels(tmp_path: Path) -> None:
    source = tmp_path / "source_backtest"
    _write_source_backtest(source)

    report = sizing.build_state_sizing_matrix(
        run_tag="unit_state_sizing",
        source_backtest_dir=source,
        output_root=tmp_path / "out",
        dataset_id="dataset",
        pool_view_id="pool",
        data_lake_root=tmp_path / "lake",
        scale_modes=("drawdown",),
        stress_scales=(0.5,),
        drawdown_thresholds=(0.02,),
        monthly_loss_thresholds=(0.01,),
        run_backtests=False,
        enforce_active_artifact_clean=False,
    )

    assert report["status"] == "completed"
    assert report["summary"]["variant_count"] == 1
    result = report["results"][0]
    target_path = Path(result["target_weight_panel_csv"])
    assert target_path.exists()
    assert "--target-weight-panel-csv" in result["command"]
    assert result["command"][result["command"].index("--target-weight-panel-csv") + 1] == str(target_path)
    assert report["boundary"]["promotion_allowed"] is False


def test_state_sizing_matrix_active_artifact_guard_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sizing, "_active_artifact_has_diff", lambda: True)

    with pytest.raises(ValueError, match="active_artifact_diff_blocker"):
        sizing.build_state_sizing_matrix(output_root=tmp_path / "out")
