from __future__ import annotations

import pandas as pd
import pytest

from daily_research.path_policy import v2_horizon_concentration_repair_scout as scout


HORIZONS = (1, 2, 3, 5, 8, 10, 15, 20, 30)


def _frame(*, rows: int = 90, tiny_30d_edge: bool = True) -> pd.DataFrame:
    items = []
    for idx in range(rows):
        strength = (idx % 30) / 30.0
        date = pd.Timestamp("2024-01-02") + pd.Timedelta(days=idx // 10)
        row = {
            "date": date.strftime("%Y-%m-%d"),
            "stock": f"{idx:06d}.SZ",
            "role": "test",
            "future_decision_score": strength + 0.01,
            "future_best_horizon": HORIZONS[idx % len(HORIZONS)],
        }
        for horizon in HORIZONS:
            base = strength + 0.001 * horizon
            if tiny_30d_edge and horizon == 30:
                base = strength + 0.0205
            row[f"pred_decision_utility_{horizon}d"] = base
            row[f"future_hit_label_{horizon}d"] = 1 if idx % 3 == 0 or strength > 0.75 else 0
        row["pred_decision_score"] = max(row[f"pred_decision_utility_{horizon}d"] for horizon in HORIZONS)
        row["trade_utility_score"] = row["pred_decision_score"]
        row["pred_best_horizon"] = 30
        items.append(row)
    return pd.DataFrame(items)


def test_penalty_30d_recomputes_pred_best_horizon() -> None:
    frame = _frame(rows=30, tiny_30d_edge=True)
    stats = scout.fit_validation_stats(frame)

    repaired = scout.apply_horizon_repair_variant(frame, variant="penalty_30d_0p002", validation_stats=stats)

    assert repaired["pred_best_horizon"].eq(30).mean() < frame["pred_best_horizon"].eq(30).mean()
    assert set(repaired["pred_best_horizon"].unique()).issubset(set(HORIZONS))


def test_validation_center_mean_uses_validation_stats_to_reduce_horizon_bias() -> None:
    validation = _frame(rows=45, tiny_30d_edge=True)
    test = _frame(rows=45, tiny_30d_edge=True)
    stats = scout.fit_validation_stats(validation)

    repaired = scout.apply_horizon_repair_variant(test, variant="validation_center_mean", validation_stats=stats)

    assert repaired["pred_best_horizon"].eq(30).mean() < 1.0
    assert "horizon_repair_variant" in repaired.columns


def test_build_repair_report_marks_posthoc_boundary_and_gate() -> None:
    validation = _frame(rows=90, tiny_30d_edge=True)
    test = _frame(rows=90, tiny_30d_edge=True)
    frames = {seed: {"validation": validation.copy(), "test": test.copy()} for seed in (7, 11, 19)}

    report = scout.build_repair_report_from_frames(frames, variants=("baseline", "penalty_30d_0p002", "validation_center_mean"))

    assert report["research_program"] == scout.RESEARCH_PROGRAM
    assert report["boundary"]["promotion_allowed"] is False
    assert report["boundary"]["posthoc_output_calibration_only"] is True
    assert report["gate"]["status"] in {"repair_pass", "validation_only_needs_confirmation", "repair_not_found"}
    assert {row["variant"] for row in report["aggregate_rows"]} == {"baseline", "penalty_30d_0p002", "validation_center_mean"}


def test_unknown_variant_rejected() -> None:
    with pytest.raises(ValueError, match="unknown_horizon_repair_variants"):
        scout.build_repair_report_from_frames({7: {"validation": _frame(), "test": _frame()}}, variants=("nope",))


def test_active_artifact_guard_blocks(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(scout, "_active_artifact_has_diff", lambda: True)

    with pytest.raises(ValueError, match="active_artifact_diff_blocker"):
        scout.run_repair_scout(output_root=tmp_path)
