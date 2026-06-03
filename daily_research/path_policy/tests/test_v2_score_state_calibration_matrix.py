from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from daily_research.path_policy import v2_score_state_calibration_matrix as calibration


def _score_panel(dates: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = {
        "AAA.SH": -1.0,
        "BBB.SH": 1.0,
        "CCC.SH": 0.5,
        "DDD.SH": 0.0,
        "EEE.SH": -0.5,
    }
    scores = {
        "AAA.SH": 1.0,
        "BBB.SH": 0.8,
        "CCC.SH": 0.6,
        "DDD.SH": 0.4,
        "EEE.SH": 0.2,
    }
    for date in dates:
        for stock in scores:
            rows.append(
                {
                    "date": date,
                    "stock": stock,
                    "score": scores[stock],
                    "future_decision_score": labels[stock],
                    "future_hit_label_5d": 1.0 if stock == "BBB.SH" else 0.0,
                    "payload_marker": f"{date}:{stock}",
                }
            )
    return pd.DataFrame(rows)


def _close_panel() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-06",
        ]
    )
    return pd.DataFrame(
        {
            "AAA.SH": [10.0, 12.0, 8.0, 12.0, 8.0, 12.0],
            "BBB.SH": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "CCC.SH": [10.0, 10.1, 10.1, 10.1, 10.1, 10.1],
            "DDD.SH": [10.0, 10.0, 10.1, 10.0, 10.1, 10.0],
            "EEE.SH": [10.0, 10.0, 10.0, 10.1, 10.0, 10.0],
        },
        index=dates,
    )


def test_adjust_score_panel_penalizes_local_volatility_and_preserves_payload() -> None:
    panel = _score_panel(["2024-01-04", "2024-01-05"])
    adjusted, meta = calibration.adjust_score_panel(
        panel,
        _close_panel(),
        calibration_mode="score_volatility",
        penalty=1.0,
        score_quantile=0.80,
        recent_return_quantile=0.80,
        volatility_quantile=0.80,
        reversal_quantile=0.80,
        recent_return_window=1,
        volatility_window=2,
    )

    daily = adjusted.loc[adjusted["date"].eq(pd.Timestamp("2024-01-04"))].set_index("stock")
    assert daily.loc["AAA.SH", "score"] < daily.loc["BBB.SH", "score"]
    assert daily.loc["AAA.SH", "payload_marker"] == "2024-01-04:AAA.SH"
    assert "future_decision_score" in adjusted.columns
    assert meta["guarded_cell_count"] >= 1
    assert meta["guarded_original_top20_cell_count"] >= 1


def test_validation_selection_prefers_penalty_that_improves_validation_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation_csv = tmp_path / "validation_scores.csv"
    test_csv = tmp_path / "test_scores.csv"
    _score_panel(["2024-01-04", "2024-01-05", "2024-01-06"]).to_csv(validation_csv, index=False)
    _score_panel(["2024-01-04", "2024-01-05", "2024-01-06"]).to_csv(test_csv, index=False)
    monkeypatch.setattr(calibration, "_load_close_for_panel", lambda *_, **__: _close_panel())

    report = calibration.build_score_state_calibration_matrix(
        run_tag="unit_score_state_calibration",
        output_root=tmp_path / "out",
        validation_ensemble_csv=validation_csv,
        test_ensemble_csv=test_csv,
        dataset_id="dataset",
        pool_view_id="pool",
        feature_profile="profile",
        data_lake_root=tmp_path / "lake",
        calibration_modes=("score_volatility",),
        penalties=(0.0, 1.0),
        score_quantiles=(0.80,),
        recent_return_quantiles=(0.80,),
        volatility_quantiles=(0.80,),
        reversal_quantiles=(0.80,),
        recent_return_windows=(1,),
        volatility_windows=(2,),
        top_n_test_backtests=1,
        run_backtests=False,
        enforce_active_artifact_clean=False,
    )

    selected = report["validation_selection"]
    assert len(selected) == 1
    assert selected[0]["penalty"] == pytest.approx(1.0)
    assert report["summary"]["variant_count"] == 2
    assert report["summary"]["selected_variant_count"] == 1
    adjusted_path = Path(report["results"][0]["adjusted_score_panel_csv"])
    assert adjusted_path.exists()
    assert "--score-panel-csv" in report["results"][0]["command"]
    assert report["boundary"]["promotion_allowed"] is False
    assert report["boundary"]["active_execution_strategy_expected_diff"] == "none"


def test_score_state_calibration_active_artifact_guard_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(calibration, "_active_artifact_has_diff", lambda: True)

    with pytest.raises(ValueError, match="active_artifact_diff_blocker"):
        calibration.build_score_state_calibration_matrix(output_root=tmp_path / "out")
