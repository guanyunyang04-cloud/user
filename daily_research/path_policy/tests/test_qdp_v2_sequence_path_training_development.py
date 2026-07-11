from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from daily_research.path_policy import qdp_v2_sequence_path_training as training
from daily_research.path_policy.qdp_v2_sequence_path_pack import _bind_research_contract
from daily_research.path_policy.tests.test_qdp_v2_sequence_path_training_fixed_oos import (
    _build_tiny_pack,
    _config,
    _write_float32,
)


APPROVED_CONTRACT = Path(
    "daily_research/brain/references/seq100_candidate_complete_development_walkforward_contract_20260711.json"
).resolve()


def _build_development_pack(tmp_path: Path) -> Path:
    manifest_path = _build_tiny_pack(tmp_path, fixed_oos=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_path = Path(manifest["sample_index_path"])
    source = pd.read_parquet(sample_path)
    source = source[source["split"].isin(["train", "validation"])].copy()
    source.loc[source["split"].eq("train"), "date_idx"] = 1
    source.loc[source["split"].eq("validation"), ["split", "date_idx", "trade_date"]] = [
        "development",
        5,
        "2022-01-04",
    ]
    source["entry_filled"] = True
    source["label_valid"] = True
    source["price_label_valid"] = True
    source["va_aux_valid"] = True
    source["sample_id"] = np.arange(len(source), dtype=np.int64)

    candidate = source[source["split"].eq("development")].copy().reset_index(drop=True)
    candidate["candidate_id"] = np.arange(len(candidate), dtype=np.int64)
    # This candidate must remain in the scoring slate although it has no future
    # opportunity/path label.  It is an unfilled entry, so execution is cash 0.
    candidate.loc[3, ["entry_filled", "label_valid", "price_label_valid", "va_aux_valid"]] = False
    supervised = source[
        source["split"].eq("train")
        | (source["split"].eq("development") & source["label_valid"])
    ].copy()
    supervised.to_parquet(sample_path, index=False)
    candidate_path = tmp_path / "candidate_index.parquet"
    candidate.to_parquet(candidate_path, index=False)

    date_count = 9
    symbol_count = 4
    raw_entry = np.full((date_count, symbol_count), 10.0, dtype=np.float32)
    raw_exit = np.full((date_count, symbol_count), 11.0, dtype=np.float32)
    exit_sellable = np.ones((date_count, symbol_count), dtype=bool)
    manifest["candidate_index_path"] = str(candidate_path.resolve())
    manifest["execution_tail_days"] = 1
    manifest["max_label_dependency_days"] = 3
    manifest["dependency_padding_complete"] = True
    manifest["available_dependency_padding_days"] = 3
    manifest["execution_arrays"] = {
        "entry_open_raw": _write_float32(tmp_path / "entry_open.float32.dat", raw_entry),
        "exit_close_raw": _write_float32(tmp_path / "exit_close.float32.dat", raw_exit),
    }
    sellable_path = tmp_path / "exit_sellable.bool.dat"
    exit_sellable.tofile(sellable_path)
    manifest.setdefault("masks", {})["exit_sellable"] = {
        "path": str(sellable_path.resolve()),
        "shape": list(exit_sellable.shape),
    }
    manifest["terminal_execution_contract"] = {
        "unresolved_after_tail": "apply_precommitted_recovery_fraction",
        "recovery_fraction_of_entry_notional": 0.0,
    }
    manifest["research_contract"] = _bind_research_contract(APPROVED_CONTRACT)
    manifest["development_walkforward"] = {
        "method": "expanding_train_development_walkforward",
        "split_roles": {"fit": "train", "evaluation": "development"},
        "development_year": 2022,
        "development_start": "2022-01-04",
        "development_start_date_idx": 5,
        "max_label_dependency_days": 3,
        "label_dependency_overlap_count": 0,
    }
    manifest["normalization"].update(
        {
            "fit_scope": "feature_dates_before_development_start",
            "fit_date_end_exclusive": "2022-01-04",
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _install_development_diagnostics(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def fake_predict_split(**kwargs: Any):
        split = str(kwargs["split"])
        calls.append(split)
        ic = pd.DataFrame([{"trade_date": "2022-01-04", "rank_ic": 0.1, "count": 3}])
        topk = pd.DataFrame(
            [
                {
                    "top_k": 1,
                    "day_count": 1,
                    "selected_realized_plan_return_coverage": 1.0,
                    "selected_realized_plan_value_coverage": 1.0,
                    "selected_realized_plan_coverage": 1.0,
                }
            ]
        )
        daily = topk.assign(trade_date="2022-01-04")
        candidates = pd.DataFrame(
            [{"trade_date": "2022-01-04", "symbol": "000001.SZ", "score": 0.2}]
        )
        metrics = {
            "split": split,
            "row_count": 4,
            "date_count": 1,
            "rank_ic_mean": 0.1,
            "rank_ic_median": 0.1,
            "rank_ic_positive_day_rate": 1.0,
            "value_column": "path_trade_value_v2_2d",
            "score_column": "score",
            "target_mean": 0.1,
            "prediction_mean": 0.2,
            "prediction_csv": "",
        }
        return ic, topk, daily, candidates, metrics

    monkeypatch.setattr(training, "_predict_split", fake_predict_split)
    monkeypatch.setattr(training, "_write_report", lambda output_dir, **_: str(output_dir / "report.md"))
    monkeypatch.setattr(training, "_find_latest_baseline_summary", lambda _: {})
    monkeypatch.setattr(training, "_trim_working_set", lambda: None)
    monkeypatch.setattr(training, "_validate_development_split_contract", lambda **_: None)
    return calls


def test_development_early_stops_on_total_loss_and_restores_best_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _build_development_pack(tmp_path)
    calls = _install_development_diagnostics(monkeypatch)
    loss_values = iter([3.0, 2.0, 2.1, 2.2])

    def fake_loss(**_: Any) -> dict[str, Any]:
        value = next(loss_values)
        return {
            **{key: value for key in training.VALIDATION_LOSS_KEYS},
            "sample_count": 3,
            "batch_count": 1,
            "seconds": 0.01,
            "aggregation": "sample_weighted_batch_mean_all_supervised_rows",
        }

    monkeypatch.setattr(training, "_evaluate_development_loss", fake_loss)
    config = replace(
        _config(tmp_path, manifest_path, evaluation_mode=training.EVALUATION_MODE_STANDARD),
        evaluation_mode=training.EVALUATION_MODE_DEVELOPMENT,
        epochs=5,
        early_stopping_patience=2,
        early_stopping_metric=training.EARLY_STOPPING_METRIC_DEVELOPMENT_TOTAL_LOSS,
        early_stopping_mode=training.EARLY_STOPPING_MODE_MIN,
        min_complete_epochs=1,
        development_contract=APPROVED_CONTRACT,
    )

    summary = training.train_sequence_path_model(config)

    assert summary["completed_epochs"] == 4
    assert summary["best_epoch"] == 2
    assert summary["checkpoint_policy"] == "best_development_total_loss"
    assert summary["early_stopping"]["best_value"] == 2.0
    assert summary["early_stopping"]["stopped_early"] is True
    assert summary["early_stopping"]["topk_selects_checkpoint"] is False
    assert calls == ["development"] * 5  # four epoch diagnostics plus restored-best final evaluation
    assert summary["sample_selection"]["train"]["policy"] == "all_rows"
    assert summary["sample_selection"]["development_supervised"]["selected_row_count"] == 3
    assert summary["sample_selection"]["development_candidates"]["policy"] == "all_candidates"
    assert summary["sample_selection"]["development_candidates"]["selected_row_count"] == 4
    assert summary["candidate_index_sha256"]
    assert summary["research_contract"]["contract_sha256"] == (
        "e461c3e4654b0e97b31508874f2eacf89258d5219bf0fb452d8f1690ee193f11"
    )
    checkpoint = torch.load(summary["best_checkpoint"], map_location="cpu", weights_only=False)
    assert checkpoint["best_epoch"] == 2
    assert checkpoint["best_development_total_loss"] == 2.0
    assert checkpoint["checkpoint_policy"] == "best_development_total_loss"
    assert all("development_total_loss" in row for row in summary["history"])
    assert all("development_path_loss" in row for row in summary["history"])


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"max_samples_per_split": 1}, "full training data"),
        ({"early_stopping_patience": 0}, "patience"),
        ({"early_stopping_metric": "validation_rank_ic_mean"}, "development_total_loss"),
        ({"early_stopping_mode": "max"}, "mode=min"),
        ({"min_complete_epochs": 0}, "at least 1"),
    ],
)
def test_development_rejects_non_contract_training_controls(
    tmp_path: Path,
    updates: dict[str, Any],
    message: str,
) -> None:
    manifest_path = _build_development_pack(tmp_path)
    config = replace(
        _config(tmp_path, manifest_path, evaluation_mode=training.EVALUATION_MODE_STANDARD),
        evaluation_mode=training.EVALUATION_MODE_DEVELOPMENT,
        epochs=3,
        early_stopping_patience=2,
        early_stopping_metric=training.EARLY_STOPPING_METRIC_DEVELOPMENT_TOTAL_LOSS,
        early_stopping_mode=training.EARLY_STOPPING_MODE_MIN,
        min_complete_epochs=1,
        development_contract=APPROVED_CONTRACT,
    )
    config = replace(config, **updates)

    with pytest.raises(ValueError, match=message):
        training.train_sequence_path_model(config)


def test_topk_ranks_finite_score_even_when_future_label_is_unavailable() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["2025-12-31"] * 3,
            "symbol": ["UNLABELED", "LABELED_A", "LABELED_B"],
            "score": [1.0, 0.5, 0.1],
            "path_trade_value_2d": [np.nan, 0.2, 0.1],
            "future_max_return_2d": [np.nan, 0.3, 0.2],
            "future_min_return_2d": [np.nan, -0.1, -0.05],
            "future_final_return_2d": [np.nan, 0.1, 0.05],
            "future_peak_day_2d": [np.nan, 1.0, 2.0],
            "drawdown_after_peak_2d": [np.nan, -0.1, -0.05],
        }
    )

    daily = training._topk_daily_rows(
        frame,
        top_k_values=(1,),
        forward_days=2,
        value_column="path_trade_value_2d",
    )[0]
    candidates = training._topk_candidate_rows(
        frame,
        max_top_k=1,
        forward_days=2,
        value_column="path_trade_value_2d",
    )

    assert daily["universe_count"] == 3
    assert daily["selected_label_coverage"] == 0.0
    assert candidates[0]["symbol"] == "UNLABELED"
    assert candidates[0]["label_available"] is False


def test_raw_execution_enforces_t_plus_one_cash_and_terminal_recovery() -> None:
    forward_days = 2
    summary_columns = training.legacy_derived_path_summary_columns(forward_days)
    predicted = np.zeros((3, len(summary_columns)), dtype=np.float32)
    predicted[:, summary_columns.index("best_exit_day_2d")] = 1.0
    true_path = np.zeros((3, forward_days, 4), dtype=np.float32)
    entry = np.asarray([10.0, 10.0, 10.0], dtype=np.float32)
    exits = np.asarray(
        [[10.0, 11.0, 12.0], [10.0, 11.0, 12.0], [10.0, 11.0, 12.0]],
        dtype=np.float32,
    )
    sellable = np.asarray(
        [[False, False, False], [False, False, True], [False, False, False]],
        dtype=bool,
    )

    result = training._realize_path_value_v2_plan_numpy(
        predicted,
        true_path,
        forward_days=forward_days,
        entry_filled=np.asarray([False, True, True]),
        entry_open_raw=entry,
        exit_close_raw_path=exits,
        exit_sellable_path=sellable,
        execution_tail_days=1,
        terminal_recovery_fraction=0.0,
    )

    assert result["predicted_exit_day"].tolist() == [2.0, 2.0, 2.0]
    assert result["realized_plan_return"].tolist() == pytest.approx([0.0, 0.2, -1.0])
    assert np.isnan(result["realized_plan_exit_day"][0])
    assert result["realized_plan_exit_day"][1:].tolist() == [3.0, 3.0]
    assert result["realized_plan_covered"].tolist() == [1.0, 1.0, 1.0]
    assert result["realized_plan_terminal_recovery"].tolist() == [0.0, 0.0, 1.0]
