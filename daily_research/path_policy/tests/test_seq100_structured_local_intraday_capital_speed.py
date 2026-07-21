from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from daily_research.path_policy import qdp_v2_sequence_path_training as training
from daily_research.path_policy import seq100_finite_capital_backtest as finite
from daily_research.path_policy import seq100_structured_global_intraday as global_study
from daily_research.path_policy import (
    seq100_structured_local_intraday_capital_speed as study,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_factorial_contract_and_single_epoch_task_order() -> None:
    assert list(study.MODEL_SPECS) == ["l35v2", "l43v2", "l35v3", "l43v3"]
    assert study.TRAINING_MODEL_ORDER == ("l43v2", "l35v3", "l43v3")
    assert study.MODEL_SPECS["l35v2"].reuse_legacy is True
    assert study.MODEL_SPECS["l43v2"].input_dim == 43
    assert study.MODEL_SPECS["l35v3"].is_v3 is True
    assert study.MODEL_SPECS["l43v3"].input_channel_profile == (
        training.INPUT_CHANNEL_PROFILE_DAILY_ONLY_TURNOVER_INTRADAY
    )
    contract = study._semantic_contract()
    assert contract["training"]["epochs"] == 1
    assert contract["training"]["batch_size"] == 512
    assert contract["training"]["rank_training_profile"] == "local_chunk"
    assert contract["training"]["hard_negative_mining"] is False
    assert contract["early_stopping"]["development_labels_select_epoch"] is False
    assert contract["evaluation"]["d7_special_weight"] is False


def test_state_machine_advances_one_factorial_model_at_a_time(tmp_path: Path) -> None:
    root = tmp_path / "study"
    root.mkdir()
    _write_json(root / "study.json", {"study_id": study.STUDY_ID})
    study._set_state(root, "l43v2_training")
    assert study._training_spec_for_state(root).model_id == "l43v2"
    assert study._next_evaluation_state(study.MODEL_SPECS["l43v2"]) == (
        "l43v2_evaluating"
    )
    assert study._post_evaluation_state(study.MODEL_SPECS["l43v2"]) == (
        "l35v3_training"
    )
    assert study._post_evaluation_state(study.MODEL_SPECS["l35v3"]) == (
        "l43v3_training"
    )
    assert study._post_evaluation_state(study.MODEL_SPECS["l43v3"]) == (
        "factorial_summarizing"
    )


def test_capital_speed_v3_uses_net_log_growth_per_capital_day() -> None:
    path = np.zeros((1, 60, 4), dtype=np.float32)
    path[0, 1, 3] = 0.04
    path[0, 59, 3] = 0.20
    cost = np.full((1, 60), 0.998, dtype=np.float32)
    result = training._derive_path_summary_numpy(
        path,
        path_value_semantic=training.PATH_VALUE_SEMANTIC_CAPITAL_SPEED_V3,
        path_value_growth_multiplier=cost,
    )
    expected = math.log((1.0 + 0.04) * 0.998) / 2.0
    assert int(result[0, 8]) == 2
    assert math.isclose(float(result[0, -1]), expected, rel_tol=1e-6)


def test_unit_time_alpha_is_computed_after_daily_cohort_aggregation() -> None:
    plan = {
        "planned_day": np.full((3, 1), 2, dtype=np.int16),
        "exit_day": np.asarray([[2], [4], [4]], dtype=np.int16),
        "terminal_recovery": np.zeros((3, 1), dtype=bool),
    }
    cash = {
        "net_return": np.asarray([[0.04], [0.02], [0.00]], dtype=np.float64),
        "order_filled": np.ones((3, 1), dtype=bool),
        "total_cost": np.zeros((3, 1), dtype=np.float64),
        "cash_utilization": np.ones((3, 1), dtype=np.float64),
    }
    row = study._unit_metric_rows(
        model_id="l35v2",
        year=2023,
        trade_date="2023-01-03",
        policy_names=["fixed_d2"],
        policy_kinds=["fixed"],
        selected_idx=np.asarray([0], dtype=np.int64),
        selected_symbols=["000001.SZ"],
        top_k=1,
        cost_scenario="base",
        plan=plan,
        cash=cash,
        universe_count=3,
    )[0]
    expected = (
        math.log1p(0.04) / 2.0
        - math.log1p((0.04 + 0.02) / 3.0) / ((2.0 + 4.0 + 4.0) / 3.0)
    ) * 252.0
    assert math.isclose(row["annualized_unit_time_alpha"], expected, rel_tol=1e-12)


def test_rolling_path_keeps_latest_request_when_forecast_is_missing() -> None:
    book = finite.ForecastBook("test")
    # A day-1 refresh shortens the original D5 request to absolute D3.  Later
    # forecasts are absent, so the request must remain D3 instead of being reset.
    book.add_day(
        date_idx=101,
        symbol_idx=np.asarray([7]),
        score=np.asarray([0.1]),
        planned_day=np.asarray([2]),
    )
    result = study._rolling_plan_matrix(
        book=book,
        signal_date_idx=100,
        symbol_idx=np.asarray([7]),
        initial_planned_day=np.asarray([5]),
        entry_filled=np.asarray([True]),
        entry_prices=np.asarray([10.0]),
        exit_prices=np.full((1, 80), 11.0, dtype=np.float64),
        exit_sellable=np.ones((1, 80), dtype=bool),
        terminal_recovery_fraction=0.0,
    )
    assert int(result["exit_day"][0, 0]) == 3
    assert float(result["exit_price"][0, 0]) == 11.0
    assert bool(result["terminal_recovery"][0, 0]) is False


def test_factorial_effects_include_interaction() -> None:
    growth = {"l35v2": 0.10, "l43v2": 0.13, "l35v3": 0.16, "l43v3": 0.21}
    selection = {
        "frontier": [
            {
                "model_id": model_id,
                "selected": {"annualized_log_growth": value},
            }
            for model_id, value in growth.items()
        ]
    }
    effects = study._factorial_effects(selection).set_index("effect")
    assert math.isclose(
        effects.loc["intraday_main_effect_v2", "annualized_log_growth_delta"],
        0.03,
    )
    assert math.isclose(
        effects.loc["v3_main_effect_35", "annualized_log_growth_delta"], 0.06
    )
    assert math.isclose(
        effects.loc["intraday_v3_interaction", "annualized_log_growth_delta"],
        0.02,
    )


def test_global_retirement_is_permanent_and_preserves_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "global"
    _write_json(root / "study.json", {"study_id": global_study.STUDY_ID})
    _write_json(root / "state.json", {"status": "v2_training", "current_task": "x"})
    evidence = root / "runs/g35v2/partial/evidence.bin"
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(b"preserve-me")
    monkeypatch.setattr(global_study, "_classify_runs", lambda *_: ([], [evidence.parent]))

    retirement = global_study.retire_structured_global_intraday(
        study_root=root, successor_study_id=study.STUDY_ID
    )
    assert retirement["status"] == "abandoned"
    assert evidence.read_bytes() == b"preserve-me"
    assert global_study.run_structured_global_intraday(study_root=root)["status"] == (
        "abandoned"
    )
    assert global_study.evaluate_structured_global_intraday(study_root=root)[
        "training_allowed"
    ] is False
    assert json.loads((root / "state.json").read_text())["current_task"] is None


def test_batch1024_probe_falls_back_only_to_activation_checkpointing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "study"
    root.mkdir()
    _write_json(
        root / "stage_decisions.json",
        {"batch512_selection": {"winner_model_id": "l35v2"}},
    )
    view_path = root / "view.json"
    _write_json(view_path, {})

    class FakeDataset:
        input_dim = 35
        forward_days = 60
        path_summary_columns = ["value"]
        symbol_count = 3
        richer_path_dim = 4
        value_index = 0
        price_anchor = "next_open"
        sample_index = pd.DataFrame({"date_idx": np.zeros(1024, dtype=np.int64)})

        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def get_batch(self, *_: Any, **__: Any) -> dict[str, Any]:
            return {"sentinel": True}

    class FakeSampler:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def __iter__(self):
            yield list(range(1024))

    calls: list[str] = []

    def fake_probe(**kwargs: Any) -> dict[str, Any]:
        profile = str(kwargs["activation_profile"])
        calls.append(profile)
        if profile == training.ACTIVATION_CHECKPOINT_PROFILE_NONE:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return {
            "status": "success",
            "activation_checkpoint_profile": profile,
            "batch_size": 1024,
        }

    monkeypatch.setattr(study, "_view_for", lambda *_: view_path)
    monkeypatch.setattr(training, "SequencePathPackDataset", FakeDataset)
    monkeypatch.setattr(training, "DateGroupedBatchSampler", FakeSampler)
    monkeypatch.setattr(study, "_probe_backward", fake_probe)
    result = study._run_batch1024_probe(root)

    assert result["status"] == "success"
    assert calls == [
        training.ACTIVATION_CHECKPOINT_PROFILE_NONE,
        training.ACTIVATION_CHECKPOINT_PROFILE_STRUCTURED_GRU,
    ]
    assert result["tried_batch768"] is False
    assert result["used_gradient_accumulation"] is False
    batch_spec = study._batch1024_spec(root)
    assert batch_spec is not None
    assert batch_spec.batch_size == 1024
    assert batch_spec.activation_checkpoint_profile == (
        training.ACTIVATION_CHECKPOINT_PROFILE_STRUCTURED_GRU
    )


def test_structured_gru_activation_checkpoint_preserves_forward_gradient_and_state() -> None:
    torch.manual_seed(7)
    plain = training.SequencePathModel(
        input_dim=35,
        hidden_dim=8,
        layers=2,
        forward_days=5,
        summary_dim=10,
        dropout=0.0,
        model_type="gru_structured_joint_turnover",
        symbol_count=3,
        richer_path_dim=4,
        activation_checkpoint_profile=training.ACTIVATION_CHECKPOINT_PROFILE_NONE,
    )
    checkpointed = training.SequencePathModel(
        input_dim=35,
        hidden_dim=8,
        layers=2,
        forward_days=5,
        summary_dim=10,
        dropout=0.0,
        model_type="gru_structured_joint_turnover",
        symbol_count=3,
        richer_path_dim=4,
        activation_checkpoint_profile=training.ACTIVATION_CHECKPOINT_PROFILE_STRUCTURED_GRU,
    )
    checkpointed.load_state_dict(plain.state_dict())
    assert list(plain.state_dict()) == list(checkpointed.state_dict())
    plain.train()
    checkpointed.train()
    source = torch.randn(3, 12, 35)
    x_plain = source.clone().requires_grad_(True)
    x_checkpointed = source.clone().requires_grad_(True)
    plain_output = plain(x_plain)
    checkpointed_output = checkpointed(x_checkpointed)
    for key in plain_output:
        assert torch.allclose(plain_output[key], checkpointed_output[key], atol=1e-6)
    sum(value.sum() for value in plain_output.values()).backward()
    sum(value.sum() for value in checkpointed_output.values()).backward()
    assert torch.allclose(x_plain.grad, x_checkpointed.grad, atol=1e-6)
    for plain_parameter, checkpointed_parameter in zip(
        plain.parameters(), checkpointed.parameters(), strict=True
    ):
        assert plain_parameter.grad is not None
        assert checkpointed_parameter.grad is not None
        assert torch.allclose(
            plain_parameter.grad, checkpointed_parameter.grad, atol=1e-6
        )
