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


class WeightedDummyRegressor(DummyRegressor):
    last_sample_weight = None

    def fit(self, x_train, y_train, sample_weight=None):
        type(self).last_sample_weight = sample_weight
        return super().fit(x_train, y_train)


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
                    "ml_market_ret_20d_mean": -0.01 if date_index < 3 else 0.02,
                    "ml_breadth_20d_positive_rate": 0.40 if date_index < 3 else 0.60,
                    "ml_market_volatility_20d_mean": 0.03 if date_index < 3 else 0.01,
                    "xsec_excess_ret_2d": 0.01 * (date_index + 1) - 0.001 * code_index,
                }
            )
    return pd.DataFrame(rows)


def test_label_cutoff_date_requires_horizon_buffer() -> None:
    dates = pd.to_datetime(["2025-12-26", "2025-12-29", "2025-12-30", "2025-12-31"])

    cutoff = frontier_ml_signal_rebuild.label_cutoff_date(dates, "2025-12-31", horizon=2)

    assert cutoff == pd.Timestamp("2025-12-29")


def test_build_ml_signal_artifacts_uses_prior_years_and_label_buffer() -> None:
    plan, predictions, importance, audit, regime_audit = frontier_ml_signal_rebuild.build_ml_signal_artifacts(
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
    assert set(predictions["training_mode"]) == {"baseline"}
    assert set(predictions["model_head"]) == {"global"}
    assert set(importance["feature"]) == {"feature_a_z", "feature_b_z"}
    assert {"training_mode", "model_head"}.issubset(importance.columns)
    assert audit.iloc[0]["prediction_row_count"] == 4
    assert not regime_audit.empty
    assert set(regime_audit["regime_label"]) == {"weak_regime", "normal_regime"}


def test_weak_weighted_training_passes_prior_fit_sample_weights() -> None:
    WeightedDummyRegressor.last_sample_weight = None

    plan, predictions, _importance, audit, _regime_audit = frontier_ml_signal_rebuild.build_ml_signal_artifacts(
        _ml_panel(),
        years=(2026,),
        final_end_date="2026-01-05",
        horizon=2,
        feature_columns=("feature_a_z", "feature_b_z"),
        target_col="xsec_excess_ret_2d",
        model_class=WeightedDummyRegressor,
        model_params={"random_state": 42},
        max_train_years=1,
        min_train_years=1,
        training_mode="weak_weighted",
        weak_years=(2025,),
        weak_sample_weight=3.0,
        ml_signal_name="ml_lgbm_xsec_excess_score_h2_prior_fit",
    )

    assert plan.iloc[0]["training_mode"] == "weak_weighted"
    assert bool(plan.iloc[0]["fit_uses_eval_year"]) is False
    assert WeightedDummyRegressor.last_sample_weight is not None
    assert set(WeightedDummyRegressor.last_sample_weight.tolist()) == {3.0}
    assert audit.iloc[0]["sample_weight_mean"] == 3.0
    assert predictions["fit_uses_eval_year"].eq(False).all()


def test_regime_heads_skip_when_prior_head_samples_are_insufficient() -> None:
    plan, predictions, _importance, audit, regime_audit = frontier_ml_signal_rebuild.build_ml_signal_artifacts(
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
        training_mode="regime_heads",
        min_regime_head_rows=10,
        ml_signal_name="ml_lgbm_xsec_excess_score_h2_prior_fit",
    )

    assert plan.iloc[0]["status"] == "skipped/insufficient_regime_heads"
    assert bool(plan.iloc[0]["fit_uses_eval_year"]) is False
    assert predictions.empty
    assert audit.iloc[0]["prediction_row_count"] == 0
    assert regime_audit["fit_uses_eval_year"].eq(False).all()


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
    assert (run_dir / "ml_regime_training_audit.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (tmp_path / "ml.md").exists()
