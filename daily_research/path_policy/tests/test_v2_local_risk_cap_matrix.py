from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from daily_research.path_policy import v2_local_risk_cap_matrix as local_caps


def _write_source_backtest(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"date": "2024-01-02", "stock": "AAA.SH", "target_weight": 0.50},
            {"date": "2024-01-02", "stock": "BBB.SH", "target_weight": 0.30},
            {"date": "2024-01-02", "stock": "CCC.SH", "target_weight": 0.20},
            {"date": "2024-01-03", "stock": "AAA.SH", "target_weight": 0.50},
            {"date": "2024-01-03", "stock": "BBB.SH", "target_weight": 0.30},
            {"date": "2024-01-03", "stock": "CCC.SH", "target_weight": 0.20},
            {"date": "2024-01-04", "stock": "AAA.SH", "target_weight": 0.50},
            {"date": "2024-01-04", "stock": "BBB.SH", "target_weight": 0.30},
            {"date": "2024-01-04", "stock": "CCC.SH", "target_weight": 0.20},
        ]
    ).to_csv(root / "aligned_daily_target_weight_panel.csv", index=False)
    pd.DataFrame(
        [
            {"date": "2024-01-02", "stock": "AAA.SH", "score": 0.90},
            {"date": "2024-01-02", "stock": "BBB.SH", "score": 0.50},
            {"date": "2024-01-02", "stock": "CCC.SH", "score": 0.40},
            {"date": "2024-01-03", "stock": "AAA.SH", "score": 0.90},
            {"date": "2024-01-03", "stock": "BBB.SH", "score": 0.50},
            {"date": "2024-01-03", "stock": "CCC.SH", "score": 0.40},
            {"date": "2024-01-04", "stock": "AAA.SH", "score": 0.90},
            {"date": "2024-01-04", "stock": "BBB.SH", "score": 0.50},
            {"date": "2024-01-04", "stock": "CCC.SH", "score": 0.40},
        ]
    ).to_csv(root / "aligned_daily_score_panel.csv", index=False)


def test_apply_local_risk_caps_scales_and_redistributes() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    target = pd.DataFrame({"AAA.SH": [0.5, 0.5, 0.5], "BBB.SH": [0.3, 0.3, 0.3], "CCC.SH": [0.2, 0.2, 0.2]}, index=dates)
    score = pd.DataFrame({"AAA.SH": [0.9, 0.9, 0.9], "BBB.SH": [0.5, 0.5, 0.5], "CCC.SH": [0.4, 0.4, 0.4]}, index=dates)
    close = pd.DataFrame({"AAA.SH": [10.0, 12.0, 12.2], "BBB.SH": [10.0, 10.1, 10.2], "CCC.SH": [10.0, 10.0, 10.1]}, index=dates)

    adjusted, meta, mask = local_caps.apply_local_risk_caps(
        target,
        score,
        close,
        cap_mode="score_reversal",
        stress_scale=0.5,
        score_quantile=0.80,
        recent_return_quantile=0.80,
        volatility_quantile=0.80,
        reversal_quantile=0.80,
        recent_return_window=1,
        volatility_window=2,
        redistribution_mode="source_weight",
        max_weight=0.80,
    )

    assert adjusted.loc[pd.Timestamp("2024-01-04"), "AAA.SH"] < target.loc[pd.Timestamp("2024-01-04"), "AAA.SH"]
    assert adjusted.loc[pd.Timestamp("2024-01-04")].sum() == pytest.approx(1.0)
    assert adjusted.max().max() <= 0.80
    assert meta["active_mask_cell_count"] >= 1
    assert not mask.empty


def test_local_risk_cap_matrix_dry_run_writes_target_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source_backtest"
    _write_source_backtest(source)

    def fake_close(**kwargs):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        return pd.DataFrame({"AAA.SH": [10.0, 12.0, 12.2], "BBB.SH": [10.0, 10.1, 10.2], "CCC.SH": [10.0, 10.0, 10.1]}, index=dates)

    from daily_research.path_policy import v2_bad_month_attribution as attribution

    monkeypatch.setattr(attribution, "_load_close_panel", fake_close)
    report = local_caps.build_local_risk_cap_matrix(
        run_tag="unit_local_caps",
        source_backtest_dir=source,
        output_root=tmp_path / "out",
        dataset_id="dataset",
        pool_view_id="pool",
        data_lake_root=tmp_path / "lake",
        cap_modes=("score_reversal",),
        stress_scales=(0.5,),
        score_quantiles=(0.80,),
        recent_return_quantiles=(0.80,),
        volatility_quantiles=(0.80,),
        reversal_quantiles=(0.80,),
        recent_return_windows=(1,),
        volatility_windows=(2,),
        redistribution_modes=("source_weight",),
        run_backtests=False,
        enforce_active_artifact_clean=False,
    )

    assert report["status"] == "completed"
    assert report["summary"]["variant_count"] == 1
    assert report["boundary"]["promotion_allowed"] is False
    result = report["results"][0]
    assert Path(result["target_weight_panel_csv"]).exists()
    assert Path(result["risk_mask_csv"]).exists()
    assert "--target-weight-panel-csv" in result["command"]


def test_local_risk_cap_matrix_active_artifact_guard_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_caps, "_active_artifact_has_diff", lambda: True)

    with pytest.raises(ValueError, match="active_artifact_diff_blocker"):
        local_caps.build_local_risk_cap_matrix(output_root=tmp_path / "out")
