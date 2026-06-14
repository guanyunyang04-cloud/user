from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from daily_research.path_policy.personal_topk_diagnostics import build_personal_topk_diagnostics


def test_personal_topk_diagnostics_scores_concentrated_selection(tmp_path: Path) -> None:
    rows = []
    for date in ("2025-01-02", "2025-01-03"):
        for stock, score, future, rank in [
            ("AAA", 0.90, 0.050, 0.95),
            ("BBB", 0.70, 0.030, 0.80),
            ("CCC", 0.10, 0.000, 0.50),
            ("DDD", -0.10, -0.020, 0.20),
            ("EEE", -0.30, -0.040, 0.05),
        ]:
            rows.append(
                {
                    "date": date,
                    "stock": stock,
                    "pred_cum_mu_5d": score,
                    "future_cum_excess_return_5d": future,
                    "future_rank_5d": rank,
                }
            )
    prediction_csv = tmp_path / "forecast_predictions_test.csv"
    pd.DataFrame(rows).to_csv(prediction_csv, index=False)

    report = build_personal_topk_diagnostics(
        prediction_csv=prediction_csv,
        output_root=tmp_path / "diag",
        score_columns=("pred_cum_mu_5d",),
        horizons=(5,),
        top_ks=(1, 2),
        round_trip_cost_bps=10.0,
    )

    assert report["status"] == "completed"
    assert Path(report["outputs"]["summary_csv"]).exists()
    assert Path(report["outputs"]["daily_csv"]).exists()
    assert Path(report["outputs"]["report_json"]).exists()

    summary = pd.read_csv(report["outputs"]["summary_csv"])
    top1 = summary[(summary["score_column"] == "pred_cum_mu_5d") & (summary["top_k"] == 1)].iloc[0]
    top2 = summary[(summary["score_column"] == "pred_cum_mu_5d") & (summary["top_k"] == 2)].iloc[0]

    assert top1["gross_mean"] == pytest.approx(0.050)
    assert top1["net_mean"] == pytest.approx(0.049)
    assert top1["selected_future_rank_mean"] == pytest.approx(0.95)
    assert top2["gross_mean"] == pytest.approx(0.040)
    assert top2["hit_rate_mean"] == pytest.approx(1.0)

    saved = json.loads(Path(report["outputs"]["report_json"]).read_text(encoding="utf-8"))
    assert saved["contract"]["not_a_backtest"] is True
    assert saved["leaderboard"][0]["top_k"] == 1
