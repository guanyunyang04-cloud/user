from __future__ import annotations

from pathlib import Path

import pandas as pd

from traditional_quant_research.experiments import frontier_ml_signal_rebuild


class DummyRegressor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.feature_importances_ = []

    def fit(self, x_train, y_train):
        self.feature_importances_ = list(range(1, len(x_train.columns) + 1))
        return self

    def predict(self, x_eval):
        return x_eval.sum(axis=1).to_numpy()


def _ml_panel() -> pd.DataFrame:
    rows = []
    dates = pd.to_datetime(["2025-12-26", "2025-12-29", "2025-12-30", "2025-12-31", "2026-01-02", "2026-01-05"])
    for date_index, date in enumerate(dates):
        for code_index, code in enumerate(["A", "B"]):
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "feature_a_z": float(date_index + code_index),
                    "feature_b_z": float(10 - date_index + code_index),
                    "xsec_excess_ret_2d": 0.01 * (date_index + 1) - 0.001 * code_index,
                }
            )
    return pd.DataFrame(rows)


def test_label_cutoff_date_requires_horizon_buffer() -> None:
    dates = pd.to_datetime(["2025-12-26", "2025-12-29", "2025-12-30", "2025-12-31"])

    cutoff = frontier_ml_signal_rebuild.label_cutoff_date(dates, "2025-12-31", horizon=2)

    assert cutoff == pd.Timestamp("2025-12-29")


def test_build_ml_signal_artifacts_uses_prior_years_and_label_buffer() -> None:
    plan, predictions, importance, audit = frontier_ml_signal_rebuild.build_ml_signal_artifacts(
        _ml_panel(),
        years=(2026,),
        final_end_date="2026-01-05",
        horizon=2,
        feature_columns=("feature_a_z", "feature_b_z"),
        target_col="xsec_excess_ret_2d",
        model_class=DummyRegressor,
        model_params={"random_state": 42},
        max_train_years=1,
        min_train_years=1,
        ml_signal_name="ml_lgbm_xsec_excess_score_h2_prior_fit",
    )

    row = plan.iloc[0]
    assert row["label_cutoff_date"] == "2025-12-29"
    assert row["train_years"] == "2025"
    assert row["train_row_count"] == 4
    assert bool(row["fit_uses_eval_year"]) is False
    assert set(predictions["eval_year"]) == {2026}
    assert len(predictions) == 4
    assert predictions["fit_uses_eval_year"].eq(False).all()
    assert set(importance["feature"]) == {"feature_a_z", "feature_b_z"}
    assert audit.iloc[0]["prediction_row_count"] == 4


def test_run_ml_signal_rebuild_dependency_missing_writes_skip_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(frontier_ml_signal_rebuild, "_load_lgbm_regressor", lambda: None)

    result = frontier_ml_signal_rebuild.run_frontier_ml_signal_rebuild(
        years=(2026,),
        output_dir=tmp_path,
        write_research_log=True,
        research_log_path=tmp_path / "ml.md",
    )

    run_dir = Path(result["run_dir"])
    assert result["decision"] == "skipped/dependency_missing"
    assert result["candidate_count"] == 0
    assert result["prediction_rows"] == 0
    assert (run_dir / "ml_signal_plan.csv").exists()
    assert (run_dir / "ml_signal_predictions.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (tmp_path / "ml.md").exists()
