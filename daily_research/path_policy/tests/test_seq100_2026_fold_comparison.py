from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy import seq100_2026_fold_comparison as comparison
from daily_research.path_policy.seq100_development import _parser


def test_registered_boundaries_and_cli_surface() -> None:
    contract = comparison._study_semantic_contract()

    assert [comparison.SIGNAL_START, comparison.SIGNAL_END] == [
        "2026-01-05",
        "2026-03-19",
    ]
    assert comparison.EXPECTED_SIGNAL_DATE_COUNT == 48
    assert comparison.EXPECTED_CANDIDATE_COUNT == 143_840
    assert comparison.EXPECTED_SYMBOL_COUNT == 3_034
    assert comparison.EXPECTED_TRAIN_ROW_COUNT == 7_773_480
    assert comparison.EXPECTED_PURGED_ROW_COUNT == 239_081
    assert comparison.EXPECTED_PURGED_DATE_COUNT == 80
    assert contract["protected_boundaries"] == {
        "update_qdp": False,
        "call_provider": False,
        "mutate_base_pack": False,
        "change_active_execution": False,
        "create_final_model": False,
    }

    prepare = _parser().parse_args(["prepare-2026-fold"])
    run = _parser().parse_args(["run-2026-fold", "--max-tasks", "1"])
    evaluate = _parser().parse_args(["evaluate-2026-vintages", "--max-tasks", "2"])
    worker = _parser().parse_args(
        [
            "evaluate-2026-vintage-task",
            "--suite-contract",
            "contract.json",
            "--profile",
            "legal_flat_baseline",
            "--vintage",
            "2026",
            "--force-inference",
        ]
    )
    assert prepare.command == "prepare-2026-fold"
    assert run.max_tasks == 1
    assert evaluate.max_tasks == 2
    assert worker.vintage == 2026
    assert worker.force_inference is True


def test_candidate_material_does_not_filter_on_future_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(comparison, "EXPECTED_CANDIDATE_COUNT", 3)
    monkeypatch.setattr(comparison, "EXPECTED_SIGNAL_DATE_COUNT", 2)
    monkeypatch.setattr(comparison, "EXPECTED_SYMBOL_COUNT", 4)
    monkeypatch.setattr(comparison, "SIGNAL_START", "2026-01-05")
    monkeypatch.setattr(comparison, "SIGNAL_END", "2026-01-06")
    path = tmp_path / "candidates.parquet"
    pd.DataFrame(
        {
            "candidate_id": [0, 1, 2],
            "trade_date": ["2026-01-05", "2026-01-05", "2026-01-06"],
            "date_idx": [10, 10, 11],
            "symbol_idx": [0, 1, 2],
            "symbol": ["A", "B", "C"],
            "entry_filled": [True, False, True],
            "label_valid": [True, False, False],
            "price_label_valid": [True, False, False],
            "va_aux_valid": [True, False, False],
        }
    ).to_parquet(path, index=False)

    material = comparison._candidate_material(
        {"candidate_index_path": str(path), "symbol_values": ["A", "B", "C", "D"]}
    )

    assert material["row_count"] == 3
    assert material["pack_symbol_count"] == 4
    assert sum(row["label_valid_count"] for row in material["label_coverage_by_date"]) == 1


def test_fold_index_uses_strict_dependency_purge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(comparison, "DEPENDENCY_DAYS", 2)
    monkeypatch.setattr(comparison, "FORWARD_DAYS", 1)
    monkeypatch.setattr(comparison, "SIGNAL_START", "2026-01-05")
    monkeypatch.setattr(comparison, "SAFE_TRAIN_SIGNAL_END", "2025-12-29")
    monkeypatch.setattr(comparison, "PURGE_START", "2025-12-30")
    monkeypatch.setattr(comparison, "PURGE_END", "2025-12-31")
    monkeypatch.setattr(comparison, "EXPECTED_TRAIN_ROW_COUNT", 1)
    monkeypatch.setattr(comparison, "EXPECTED_PURGED_ROW_COUNT", 2)
    monkeypatch.setattr(comparison, "EXPECTED_PURGED_DATE_COUNT", 2)
    source_path = tmp_path / "source.parquet"
    pd.DataFrame(
        {
            "sample_id": [0, 1, 2],
            "split": ["train", "train", "train"],
            "trade_date": ["2025-12-29", "2025-12-30", "2025-12-31"],
            "date_idx": [0, 1, 2],
            "symbol_idx": [0, 0, 0],
            "symbol": ["A", "A", "A"],
        }
    ).to_parquet(source_path, index=False)
    candidates = pd.DataFrame(
        {
            "candidate_id": [0, 1],
            "split": ["development", "development"],
            "year": [2026, 2026],
            "trade_date": ["2026-01-05", "2026-01-05"],
            "date_idx": [3, 3],
            "symbol_idx": [0, 1],
            "symbol": ["A", "B"],
        }
    )
    supervised = candidates.iloc[[0]].rename(columns={"candidate_id": "sample_id"})

    sample_path, candidate_path, audit = comparison._build_fold_indexes(
        base={
            "sample_index_path": str(source_path),
            "date_values": [
                "2025-12-29",
                "2025-12-30",
                "2025-12-31",
                "2026-01-05",
            ],
        },
        candidates=candidates,
        supervised=supervised,
        study_root=tmp_path / "study",
    )

    samples = pd.read_parquet(sample_path)
    observed_candidates = pd.read_parquet(candidate_path)
    assert samples["split"].value_counts().to_dict() == {"train": 1, "development": 1}
    assert audit["purged_row_count"] == 2
    assert audit["purged_date_count"] == 2
    assert len(observed_candidates) == 2


def test_mapped_overlap_checks_execution_and_turnover_panels(tmp_path: Path) -> None:
    base_path = tmp_path / "base.dat"
    compact_path = tmp_path / "compact.dat"
    base = np.memmap(base_path, dtype=np.float32, mode="w+", shape=(4, 3))
    base[:] = np.arange(12, dtype=np.float32).reshape(4, 3)
    base.flush()
    compact = np.memmap(compact_path, dtype=np.float32, mode="w+", shape=(4, 4))
    compact[:] = np.nan
    symbol_map = np.asarray([2, 0, 3], dtype=np.int64)
    for date_idx in range(4):
        compact[date_idx, symbol_map] = base[date_idx]
    compact.flush()
    del base, compact

    comparison._assert_mapped_panel_overlap(
        name="fixture",
        base_meta={"path": str(base_path), "shape": [4, 3]},
        compact_meta={"path": str(compact_path), "shape": [4, 4]},
        dtype="float32",
        base_date_indices=[1, 2],
        compact_date_indices=[1, 2],
        compact_symbol_indices=symbol_map,
    )

    compact = np.memmap(compact_path, dtype=np.float32, mode="r+", shape=(4, 4))
    compact[2, symbol_map[1]] += 1.0
    compact.flush()
    del compact
    with pytest.raises(ValueError, match="overlap regression failed"):
        comparison._assert_mapped_panel_overlap(
            name="fixture",
            base_meta={"path": str(base_path), "shape": [4, 3]},
            compact_meta={"path": str(compact_path), "shape": [4, 4]},
            dtype="float32",
            base_date_indices=[1, 2],
            compact_date_indices=[1, 2],
            compact_symbol_indices=symbol_map,
        )


def _topk_frame() -> pd.DataFrame:
    rows = []
    for top_k in comparison.TOP_K_VALUES:
        rows.append(
            {
                "top_k": top_k,
                "execution_cost_contract_sha256": "cost",
                "alpha_net_realized_plan_return_base": 0.01,
                "alpha_net_realized_plan_return_stress": 0.009,
                "alpha_opportunity_value": 0.02,
                "selected_net_realized_plan_return_base": 0.03,
                "universe_net_realized_plan_return_base": 0.02,
                "selected_net_realized_plan_return_stress": 0.029,
                "universe_net_realized_plan_return_stress": 0.02,
                "selected_label_coverage": 1.0,
                "universe_label_coverage": 0.99,
                "selected_opportunity_value_coverage": 1.0,
                "universe_opportunity_value_coverage": 0.99,
                "selected_oracle_regret": 0.10,
                "universe_oracle_regret": 0.12,
                "selected_oracle_regret_coverage": 1.0,
                "universe_oracle_regret_coverage": 0.99,
                "selected_entry_fill_rate": 0.98,
                "universe_entry_fill_rate": 0.97,
                "selected_realized_plan_entry_filled_coverage": 1.0,
                "universe_realized_plan_entry_filled_coverage": 1.0,
                "selected_realized_plan_return_coverage": 1.0,
                "universe_realized_plan_return_coverage": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_extended_metrics_include_stress_returns_and_topk_coverages() -> None:
    daily = pd.DataFrame(
        [
            {
                "trade_date": "2026-01-05",
                "top_k": top_k,
                "selected_realized_plan_return_coverage": 1.0,
                "universe_realized_plan_return_coverage": 1.0,
                "universe_count": 100,
                "universe_hash": "same",
                "execution_cost_contract_sha256": "cost",
            }
            for top_k in comparison.TOP_K_VALUES
        ]
    )
    split_metrics = {
        "row_count": 100,
        "date_count": 1,
        "candidate_complete_score_coverage": 1.0,
        "rank_ic_mean": 0.1,
        "rank_ic_positive_day_rate": 1.0,
        "path_mae": 0.2,
        "path_open_mae": 0.21,
        "path_high_mae": 0.22,
        "path_low_mae": 0.23,
        "path_close_mae": 0.24,
        "entry_fill_rate": 0.97,
    }

    metrics = comparison._evaluation_metrics_2026(
        topk=_topk_frame(), daily_topk=daily, split_metrics=split_metrics
    )

    assert metrics["top3_selected_stress"] == pytest.approx(0.029)
    assert metrics["top10_universe_stress"] == pytest.approx(0.02)
    assert metrics["top5_selected_exit_regret_coverage"] == 1.0
    assert metrics["top1_universe_execution_coverage"] == 1.0


def test_expanded_gate_requires_2026_to_beat_all_three_older_vintages() -> None:
    results: list[dict[str, object]] = []
    for profile in comparison.PROFILE_ORDER:
        for vintage, offset in ((2023, 0.00), (2024, 0.01), (2025, 0.02), (2026, 0.04)):
            results.append(
                {
                    "profile": profile,
                    "checkpoint_vintage": vintage,
                    "metrics": {
                        "top1_base_alpha": offset + 0.01,
                        "top3_base_alpha": offset + 0.02,
                        "top5_base_alpha": offset + 0.03,
                        "top10_base_alpha": offset + 0.04,
                        "rank_ic": offset + 0.05,
                        "top3_opportunity_alpha": offset + 0.06,
                        "exit_regret": 0.30 - offset,
                        "path_mae": 0.20 - offset,
                    },
                }
            )
    assert comparison._expanded_freshness_gate(results)["metric_gate_pass"] is True
    target = next(
        row
        for row in results
        if row["profile"] == "legal_flat_baseline"
        and row["checkpoint_vintage"] == 2026
    )
    target["metrics"]["top3_base_alpha"] = -1.0  # type: ignore[index]
    assert comparison._expanded_freshness_gate(results)["metric_gate_pass"] is False


def test_task_reset_is_confined_to_its_registered_directory(tmp_path: Path) -> None:
    root = tmp_path / "suite"
    output = root / "tasks" / "task_a"
    output.mkdir(parents=True)
    (output / "partial.txt").write_text("partial", encoding="utf-8")
    task = {"task_id": "task_a", "output_dir": str(output)}

    comparison._reset_task_output(task=task, suite_root=root)
    assert not output.exists()

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="refusing to reset"):
        comparison._reset_task_output(
            task={"task_id": "task_a", "output_dir": str(outside)},
            suite_root=root,
        )
