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

    # This candidate must remain in the scoring slate although it has no future
    # opportunity/path label.  It is an unfilled entry, so execution is cash 0.
    development_source_rows = source.index[source["split"].eq("development")]
    source.loc[
        development_source_rows[-1],
        ["entry_filled", "label_valid", "price_label_valid", "va_aux_valid"],
    ] = False
    candidate = source[source["split"].eq("development")].copy().reset_index(drop=True)
    candidate["candidate_id"] = np.arange(len(candidate), dtype=np.int64)
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
    manifest["date_values"] = [
        "2021-12-27",
        "2021-12-28",
        "2021-12-29",
        "2021-12-30",
        "2021-12-31",
        "2022-01-04",
        "2022-01-05",
        "2022-01-06",
        "2022-01-07",
    ]
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
    manifest["execution_cost_contract"] = {
        "contract": "a_share_round_trip_cashflow_v1",
        "lot_size": 100,
        "commission_bps": 3.0,
        "minimum_commission_cny": 5.0,
        "stamp_tax_bps": 5.0,
        "stamp_tax_schedule": [
            {"effective_date": "1900-01-01", "stamp_tax_bps": 10.0},
            {"effective_date": "2023-08-28", "stamp_tax_bps": 5.0},
        ],
        "transfer_fee_bps": 0.1,
        "slippage_bps": 7.0,
        "stress_slippage_multiplier": 2.0,
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
                    "universe_realized_plan_return_coverage": 1.0,
                    "universe_realized_plan_value_coverage": 0.75,
                    "universe_realized_plan_coverage": 1.0,
                    "execution_cost_contract_sha256": "0" * 64,
                    **{
                        f"{scope}_net_realized_plan_return_{scenario}": 0.0
                        for scope in ("selected", "universe", "alpha")
                        for scenario in ("base", "stress")
                    },
                    **{
                        f"{scope}_net_realized_plan_value_{scenario}": 0.0
                        for scope in ("selected", "universe", "alpha")
                        for scenario in ("base", "stress")
                    },
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


def test_v2_development_contract_purges_only_the_prediction_horizon(tmp_path: Path) -> None:
    manifest_path = _build_development_pack(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_path = Path(manifest["sample_index_path"])
    candidate_path = Path(manifest["candidate_index_path"])
    samples = pd.read_parquet(sample_path)
    candidates = pd.read_parquet(candidate_path)
    samples["year"] = samples["trade_date"].astype(str).str[:4].astype(int)
    candidates["year"] = candidates["trade_date"].astype(str).str[:4].astype(int)
    samples["entry_trade_date"] = "2022-01-05"
    candidates["entry_trade_date"] = "2022-01-05"
    samples.to_parquet(sample_path, index=False)
    candidates.to_parquet(candidate_path, index=False)
    manifest["max_label_dependency_days"] = 2
    manifest["max_execution_dependency_days"] = 3
    manifest["development_contract"] = dict(manifest["research_contract"])
    manifest["development_walkforward"].update(
        {
            "schema_version": 2,
            "forward_days": 2,
            "execution_tail_days": 1,
            "max_label_dependency_days": 2,
            "training_label_dependency_days": 2,
            "execution_dependency_days": 3,
            "purge_rule": "training_label_end_date_idx < development_start_date_idx",
        }
    )
    manifest["normalization"]["development_feature_date_count"] = 0

    from daily_research.path_policy.seq100_fold_contract import (
        compute_development_fold_training_contract,
    )

    contract = compute_development_fold_training_contract(manifest)

    assert contract["payload"]["training_label_dependency_days"] == 2
    assert contract["payload"]["execution_dependency_days"] == 3
    manifest["development_walkforward"]["training_label_dependency_days"] = 3
    with pytest.raises(ValueError, match="purge must equal forward_days"):
        compute_development_fold_training_contract(manifest)


def test_build_development_fold_view_keeps_execution_tail_out_of_purge(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    manifest_path = _build_development_pack(source_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_path = Path(manifest["sample_index_path"])
    candidate_path = Path(manifest["candidate_index_path"])
    for path in (sample_path, candidate_path):
        frame = pd.read_parquet(path)
        frame["year"] = frame["trade_date"].astype(str).str[:4].astype(int)
        frame["entry_trade_date"] = "2022-01-05"
        frame.to_parquet(path, index=False)
    manifest["artifact_type"] = "qdp_v2_sequence_path_pack"
    manifest["max_label_dependency_days"] = 2
    manifest["max_execution_dependency_days"] = 3
    manifest["research_contract"] = _bind_research_contract(APPROVED_CONTRACT)
    manifest["development_contract"] = dict(manifest["research_contract"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    from daily_research.path_policy.seq100_fold_contract import build_development_fold_view

    result = build_development_fold_view(
        source_manifest=manifest_path,
        output_root=tmp_path / "folds",
        development_year=2022,
    )
    view = json.loads(Path(result["view_path"]).read_text(encoding="utf-8"))

    assert view["development_walkforward"]["training_label_dependency_days"] == 2
    assert view["development_walkforward"]["execution_dependency_days"] == 3
    assert view["development_walkforward"]["purged_signal_date_count"] == 2
    assert view["development_fold_training_contract"]["schema_version"] == 2


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
        minimum_complete_epochs=1,
        development_contract=APPROVED_CONTRACT,
    )

    summary = training.train_sequence_path_model(config)

    assert summary["completed_epochs"] == 4
    assert summary["best_epoch"] == 2
    assert summary["checkpoint_policy"] == "best_development_total_loss"
    assert summary["early_stopping"]["best_value"] == 2.0
    assert summary["early_stopping"]["stopped_early"] is True
    assert summary["early_stopping"]["topk_selects_checkpoint"] is False
    assert 0 < summary["best_optimizer_step"] <= summary["optimizer_step_count"]
    assert summary["resolved_training_config"]["minimum_complete_epochs"] == 1
    assert "min_complete_epochs" not in summary["resolved_training_config"]
    assert len(summary["execution_cost_contract_sha256"]) == 64
    assert calls == ["development"]  # Only the restored-best checkpoint gets candidate diagnostics.
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
    assert checkpoint["best_optimizer_step"] == summary["best_optimizer_step"]
    assert checkpoint["optimizer_step_count"] == summary["best_optimizer_step"]
    assert checkpoint["execution_cost_contract_sha256"] == summary["execution_cost_contract_sha256"]
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
        ({"minimum_complete_epochs": 0}, "at least 1"),
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
        minimum_complete_epochs=1,
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


class _DeterministicCandidatePathModel(torch.nn.Module):
    uses_derived_path_value = True
    uses_direct_value = False
    uses_ohlcva_path = False
    uses_ohlcva_aux_path = False
    residual_weight = 0.0

    def __init__(self, *, nonfinite_symbol_idx: int | None = None) -> None:
        super().__init__()
        self.nonfinite_symbol_idx = nonfinite_symbol_idx

    def forward(self, x: torch.Tensor, *, symbol_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = int(x.shape[0])
        scale = (symbol_idx.to(dtype=torch.float32) + 1.0) * 0.02
        path = torch.zeros((batch, 2, 4), dtype=x.dtype, device=x.device)
        path[:, 0, 1] = scale
        path[:, 0, 2] = -0.01
        path[:, 0, 3] = scale * 0.5
        path[:, 1, 0] = scale * 0.5
        path[:, 1, 1] = scale * 1.5
        path[:, 1, 2] = -0.005
        path[:, 1, 3] = scale
        if self.nonfinite_symbol_idx is not None:
            path[symbol_idx.eq(int(self.nonfinite_symbol_idx))] = torch.nan
        return {"future_path": path}


def test_predict_split_merges_candidate_complete_execution_after_full_date(
    tmp_path: Path,
) -> None:
    manifest_path = _build_development_pack(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = training.SequencePathPackDataset(
        manifest,
        split="development",
        max_samples=0,
        input_channel_profile=training.INPUT_CHANNEL_PROFILE_DAILY_ONLY,
        index_role="candidate",
    )

    _ic, topk, daily, candidates, metrics = training._predict_split(
        model=_DeterministicCandidatePathModel(),
        dataset=dataset,
        device=torch.device("cpu"),
        output_dir=tmp_path / "predict",
        split="development",
        batch_size=2,
        amp_enabled=False,
        top_k=(1, 3),
        write_predictions=False,
        write_path_predictions=False,
    )

    assert len(daily) == 2
    assert daily["universe_count"].eq(4).all()
    assert daily["universe_hash"].nunique() == 1
    assert daily["execution_cost_contract_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert daily["selected_realized_plan_return_coverage"].eq(1.0).all()
    assert daily["universe_realized_plan_return_coverage"].eq(1.0).all()
    assert daily["selected_realized_plan_coverage"].eq(1.0).all()
    assert np.isfinite(daily["selected_net_realized_plan_return_base"]).all()
    assert np.isfinite(daily["selected_net_realized_plan_return_stress"]).all()
    assert {
        "alpha_opportunity_value",
        "selected_oracle_regret",
        "selected_execution_cost_base_cny",
        "selected_cash_utilization_base",
        "selected_exit_status_counts_json",
    }.issubset(daily.columns)

    top1 = daily.loc[daily["top_k"].eq(1)].iloc[0]
    assert top1["selected_symbols"] == ["000004.SZ"]
    assert top1["selected_entry_fill_rate"] == 0.0
    assert top1["selected_net_realized_plan_return_base"] == 0.0
    assert top1["selected_realized_plan_value_coverage"] == 0.0
    assert pd.isna(top1["selected_net_realized_plan_value_base"])
    top3 = daily.loc[daily["top_k"].eq(3)].iloc[0]
    assert top3["selected_realized_plan_value_coverage"] == pytest.approx(2.0 / 3.0)

    assert candidates.iloc[0]["symbol"] == "000004.SZ"
    assert not bool(candidates.iloc[0]["label_available"])
    assert candidates.iloc[0]["value_label_available"] == 0.0
    assert pd.isna(candidates.iloc[0]["net_realized_plan_value_base"])
    assert candidates.iloc[0]["realized_plan_exit_status"] == "entry_unfilled_cash"
    assert {
        "net_realized_plan_return_base",
        "net_realized_plan_return_stress",
        "execution_cost_base_cny",
        "cash_utilization_base",
        "realized_plan_covered",
    }.issubset(candidates.columns)
    training._validate_development_topk_execution_coverage(topk)
    assert topk.loc[topk["top_k"].eq(1), "selected_realized_plan_value_coverage"].iloc[0] == 0.0
    assert metrics["candidate_complete_score_coverage"] == 1.0
    assert metrics["candidate_universe_date_count_verified"] == 1


def test_predict_split_rejects_nonfinite_development_candidate_score(tmp_path: Path) -> None:
    manifest_path = _build_development_pack(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = training.SequencePathPackDataset(
        manifest,
        split="development",
        max_samples=0,
        input_channel_profile=training.INPUT_CHANNEL_PROFILE_DAILY_ONLY,
        index_role="candidate",
    )

    with pytest.raises(ValueError, match="finite score for every candidate"):
        training._predict_split(
            model=_DeterministicCandidatePathModel(nonfinite_symbol_idx=3),
            dataset=dataset,
            device=torch.device("cpu"),
            output_dir=tmp_path / "predict_nonfinite",
            split="development",
            batch_size=2,
            amp_enabled=False,
            top_k=(1, 3),
            write_predictions=False,
            write_path_predictions=False,
        )


def test_adaptive_working_set_trim_uses_margin_and_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available = iter([1.5, 3.25, 3.0])
    trim_calls: list[bool] = []
    monkeypatch.setattr(training, "_available_physical_memory_gb", lambda: next(available))
    monkeypatch.setattr(
        training,
        "_current_process_memory_gb",
        lambda: {"working_set_gb": 7.0, "private_gb": 4.0},
    )
    monkeypatch.setattr(training, "_trim_working_set", lambda: trim_calls.append(True) or True)

    last_trim, event = training._maybe_trim_training_working_set(
        batch_count=31,
        last_trim_batch=0,
    )
    assert (last_trim, event) == (0, None)

    last_trim, event = training._maybe_trim_training_working_set(
        batch_count=32,
        last_trim_batch=last_trim,
    )
    assert last_trim == 32
    assert event is not None
    assert event["reason"] == "low_available_memory"
    assert event["trim_succeeded"] is True
    assert event["available_before_gb"] == 1.5
    assert event["available_after_gb"] == 3.25
    assert len(trim_calls) == 1

    cooled_last, cooled_event = training._maybe_trim_training_working_set(
        batch_count=64,
        last_trim_batch=last_trim,
    )
    assert (cooled_last, cooled_event) == (last_trim, None)

    final_last, final_event = training._maybe_trim_training_working_set(
        batch_count=288,
        last_trim_batch=last_trim,
    )
    assert (final_last, final_event) == (last_trim, None)
    assert len(trim_calls) == 1


def test_working_set_trim_fallback_and_failed_api_do_not_fake_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training, "_available_physical_memory_gb", lambda: None)
    monkeypatch.setattr(
        training,
        "_current_process_memory_gb",
        lambda: {"working_set_gb": None, "private_gb": None},
    )
    monkeypatch.setattr(training, "_trim_working_set", lambda: False)

    last_trim, event = training._maybe_trim_training_working_set(
        batch_count=1024,
        last_trim_batch=0,
    )
    assert last_trim == 0
    assert event is not None
    assert event["reason"] == "periodic_fallback"
    assert event["trim_succeeded"] is False


def test_development_loss_records_working_set_trim_without_training_progress_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _build_development_pack(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = training.SequencePathPackDataset(
        manifest,
        split="development",
        max_samples=0,
        input_channel_profile=training.INPUT_CHANNEL_PROFILE_DAILY_ONLY,
    )
    config = replace(
        _config(tmp_path, manifest_path, evaluation_mode=training.EVALUATION_MODE_DEVELOPMENT),
        batch_size=4,
    )
    model = training.SequencePathModel(
        input_dim=dataset.input_dim,
        hidden_dim=int(config.hidden_dim),
        layers=int(config.layers),
        forward_days=dataset.forward_days,
        summary_dim=len(dataset.path_summary_columns),
        dropout=float(config.dropout),
        model_type=str(config.model_type),
        symbol_count=int(dataset.symbol_count),
        symbol_embedding_dim=int(config.symbol_embedding_dim),
        richer_path_dim=int(dataset.richer_path_dim),
    )
    trim_event = {"reason": "low_available_memory", "trim_succeeded": True}
    monkeypatch.setattr(
        training,
        "_maybe_trim_training_working_set",
        lambda **kwargs: (int(kwargs["batch_count"]), trim_event),
    )

    result = training._evaluate_development_loss(
        model=model,
        dataset=dataset,
        device=torch.device("cpu"),
        config=config,
        amp_enabled=False,
        path_value_gradient_profile=training.PATH_VALUE_GRADIENT_PROFILE_SMOOTH,
        rank_training_profile=training.RANK_TRAINING_PROFILE_LOCAL_CHUNK,
    )

    assert result["sample_count"] == len(dataset)
    assert result["working_set_trim_events"] == [trim_event]
    assert np.isfinite(result["loss"])


def test_training_batch_can_skip_unused_observation_and_execution_arrays(tmp_path: Path) -> None:
    manifest_path = _build_development_pack(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = training.SequencePathPackDataset(
        manifest,
        split="train",
        input_channel_profile=training.INPUT_CHANNEL_PROFILE_DAILY_ONLY,
    )

    batch = dataset.get_batch(
        [0],
        include_metadata=False,
        include_observation=False,
        include_execution=False,
    )

    assert batch["y_observed_price_path"] is None
    assert batch["entry_open_raw"] is None
    assert batch["exit_close_raw_path"] is None
    assert batch["exit_sellable_path"] is None
    assert batch["entry_filled"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP is unavailable")
def test_finite_smooth_l1_preserves_fp32_math_under_cuda_amp() -> None:
    shape = (512, 60, 4)
    element_count = int(np.prod(shape))
    pred = torch.linspace(-0.08, 0.08, element_count, device="cuda", dtype=torch.float16).requires_grad_(True)
    target = torch.linspace(-0.03, 0.03, element_count, device="cuda", dtype=torch.float32)

    with torch.amp.autocast(device_type="cuda", enabled=True):
        actual = training._finite_smooth_l1(pred, target)
        fields_actual = training._finite_smooth_l1_fields_equal(
            pred.reshape(shape),
            target.reshape(shape),
        )
    expected = torch.nn.functional.smooth_l1_loss(
        pred.float(),
        target.float(),
        reduction="mean",
    )

    assert actual.dtype == torch.float32
    assert float(actual) > 0.0
    assert torch.isinf(torch.tensor(element_count, device="cuda", dtype=torch.float16))
    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-9)
    assert fields_actual.dtype == torch.float32
    assert float(fields_actual) > 0.0
    torch.testing.assert_close(fields_actual, expected, rtol=1.0e-6, atol=1.0e-9)
    actual.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().sum()) > 0.0
