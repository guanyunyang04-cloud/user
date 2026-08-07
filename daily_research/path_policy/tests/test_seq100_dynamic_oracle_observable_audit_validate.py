from __future__ import annotations

import json

import pandas as pd

from daily_research.path_policy import seq100_dynamic_oracle_observable_audit as audit
from daily_research.path_policy import (
    seq100_dynamic_oracle_observable_audit_validate as validate,
)


def test_validation_rejects_missing_source_output(tmp_path) -> None:
    audit.load_study()
    root = tmp_path / "missing"
    root.mkdir()
    try:
        validate.validate(study_path=audit.DEFAULT_STUDY_PATH, output_root=root)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing manifest should be rejected")


def test_validation_accepts_minimal_well_formed_fixture(tmp_path) -> None:
    audit.load_study()
    root = tmp_path / "fixture"
    root.mkdir()
    feature_names = ["f0", "f1"]
    outputs = {}
    frames = {
        "feature_metrics": pd.DataFrame(
            {
                "feature_name": feature_names,
                "column_index": [0, 1],
                "winner_percentile_effect": [0.1, -0.1],
                "mean_daily_linear_ic": [0.01, -0.01],
            }
        ),
        "feature_annual_metrics": pd.DataFrame(
            {
                "year": [2012, 2012],
                "feature_name": feature_names,
                "winner_percentile_effect": [0.1, -0.1],
                "mean_daily_linear_ic": [0.01, -0.01],
            }
        ),
        "family_metrics": pd.DataFrame(
            {"analytic_family": ["a"], "feature_count": [2]}
        ),
        "matched_neighbors": pd.DataFrame(
            {
                "date_idx": [1],
                "selected_symbol_idx": [2],
                "selected_advantage": [0.2],
                "nearest_distance": [0.5],
                "nearest_failure_distance": [0.6],
                "nearest_failure_advantage": [-0.1],
            }
        ),
        "cost_overlap": pd.DataFrame({"metric": ["x"], "value": [1.0]}),
        "perturbation_stability": pd.DataFrame(
            {
                "cost_scenario": ["base"],
                "perturbation_bps_per_action_value": [5],
                "argmax_guaranteed_stable_fraction": [1.0],
                "buy_vs_cash_guaranteed_stable_fraction": [1.0],
            }
        ),
    }
    for name, frame in frames.items():
        path = root / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        outputs[name] = {
            "path": str(path),
            "sha256": audit.replay.sha256(path),
            "rows": len(frame),
        }
    manifest = {
        "status": "completed",
        "study_id": audit.STUDY_ID,
        "audit": {
            "feature_count": 557,
            "forbidden_2026_rows": 0,
            "duplicate_input_keys": 0,
            "aligned_rows": 1,
            "model_input_rows": 1,
        },
        "outputs": outputs,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    # The fixture intentionally has fewer than 557 feature rows and therefore
    # exercises the validator's blocking path without touching the real output.
    try:
        validate.validate(study_path=audit.DEFAULT_STUDY_PATH, output_root=root)
    except ValueError as exc:
        assert "feature_metrics_row_count" in str(exc)
    else:
        raise AssertionError("invalid feature count should be rejected")
