from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from daily_research.path_policy import v2_selective_throttle_matrix as throttle


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_source_bridge(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    score_panel = root / "scores.csv"
    score_panel.write_text(
        "\n".join(
            [
                "date,stock,score",
                "2024-01-02,AAA.SH,0.90",
                "2024-01-02,BBB.SH,0.80",
                "2024-01-03,AAA.SH,0.95",
                "2024-01-03,BBB.SH,0.70",
                "2024-01-04,AAA.SH,0.96",
                "2024-01-04,BBB.SH,0.75",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        root / "v2_score_backtest_bridge_report.json",
        {
            "status": "completed",
            "bridge": {
                "backtest_score_panel_csv": str(score_panel),
                "dataset_id": "dataset",
                "pool_view_id": "pool",
            },
        },
    )
    return score_panel


def test_build_adjusted_score_panel_penalizes_recent_runup() -> None:
    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
            "stock": ["AAA.SH"] * 3 + ["BBB.SH"] * 3,
            "score": [0.8, 0.9, 0.95, 0.7, 0.7, 0.7],
        }
    )
    close = pd.DataFrame(
        {
            "AAA.SH": [10.0, 12.0, 14.0],
            "BBB.SH": [10.0, 10.0, 10.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    adjusted, stats = throttle.build_adjusted_score_panel(
        scores,
        close,
        throttle_mode="recent_runup",
        penalty=0.5,
        recent_return_window=1,
        volatility_window=2,
        recent_return_quantile=0.75,
        volatility_quantile=0.95,
    )

    final = adjusted.loc[adjusted["date"].eq("2024-01-03")].set_index("stock")["score"]
    assert final["AAA.SH"] < final["BBB.SH"]
    assert stats["guarded_cell_count"] >= 1
    assert stats["guarded_original_top20_cell_count"] >= 1


def test_selective_throttle_matrix_dry_run_writes_adjusted_score_panels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_bridge = tmp_path / "source_bridge"
    _write_source_bridge(source_bridge)

    close = pd.DataFrame(
        {
            "AAA.SH": [10.0, 11.0, 12.0],
            "BBB.SH": [10.0, 10.0, 10.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    monkeypatch.setattr(throttle.attribution, "_load_close_panel", lambda **_: close)

    report = throttle.build_selective_throttle_matrix(
        run_tag="unit_throttle",
        source_bridge_root=source_bridge,
        output_root=tmp_path / "out",
        dataset_id="dataset",
        pool_view_id="pool",
        data_lake_root=tmp_path / "lake",
        holding_counts=(20,),
        max_weights=(0.08,),
        throttle_modes=("recent_runup",),
        penalties=(0.25,),
        recent_return_windows=(1,),
        volatility_windows=(2,),
        recent_return_quantiles=(0.75,),
        volatility_quantiles=(0.95,),
        run_backtests=False,
        enforce_active_artifact_clean=False,
    )

    assert report["status"] == "completed"
    assert report["summary"]["variant_count"] == 1
    result = report["results"][0]
    adjusted_path = Path(result["adjusted_score_panel_csv"])
    assert adjusted_path.exists()
    assert "--score-panel-csv" in result["command"]
    assert result["command"][result["command"].index("--score-panel-csv") + 1] == str(adjusted_path)
    assert report["boundary"]["promotion_allowed"] is False
    assert report["boundary"]["active_execution_strategy_expected_diff"] == "none"


def test_selective_throttle_matrix_active_artifact_guard_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(throttle, "_active_artifact_has_diff", lambda: True)

    with pytest.raises(ValueError, match="active_artifact_diff_blocker"):
        throttle.build_selective_throttle_matrix(output_root=tmp_path / "out")
