from __future__ import annotations

from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest
import torch

from daily_research.path_policy.seq100_qcurve import (
    QCurveCostContract,
    QCurveModel,
    structured_qcurve_auxiliary_loss,
)
from daily_research.path_policy.seq100_qcurve_backtest import (
    PortfolioState,
    _apply_turnover_gate,
    _execute_targets,
)
from daily_research.path_policy.seq100_qcurve_lgbm import (
    ENTER_ANCHORS,
    HOLD_ANCHORS,
    _curves_from_anchors,
    _interpolate,
)
from daily_research.path_policy.seq100_qcurve_training import (
    _early_stopping_reached,
    _train_gradient_cache_day,
)
from daily_research.path_policy.seq100_qcurve_development import (
    DEVELOPMENT_YEARS,
    PROFILES,
    _job_command,
    select_profiles,
)


CONTRACT = QCurveCostContract(
    lot_size=100,
    commission_bps=3.0,
    minimum_commission_cny=5.0,
    transfer_fee_bps=0.1,
    slippage_bps=7.0,
    stress_slippage_multiplier=2.0,
    stamp_tax_schedule=(("1900-01-01", 10.0), ("2023-08-28", 5.0)),
)


def test_multiscale_grouped_model_enforces_structured_path() -> None:
    model = QCurveModel(
        input_dim=12,
        hidden_dim=16,
        dropout=0.0,
        encoder_type="multiscale_tcn",
        input_group_dims=(4, 4, 4),
    )
    outputs = model(torch.randn(3, 24, 12))
    assert torch.all(outputs["path_high_log_return"] >= outputs["path_open_log_return"])
    assert torch.all(outputs["path_high_log_return"] >= outputs["path_close_log_return"])
    assert torch.all(outputs["path_low_log_return"] <= outputs["path_open_log_return"])
    assert torch.all(outputs["path_low_log_return"] <= outputs["path_close_log_return"])
    assert torch.allclose(
        outputs["path_log_amount"],
        outputs["path_log_volume"] + outputs["path_log_vwap"],
    )
    target = torch.zeros(3, 60, 6)
    price_loss, va_loss, _ = structured_qcurve_auxiliary_loss(outputs, target)
    assert torch.isfinite(price_loss)
    assert torch.isfinite(va_loss)


class _TinyPack:
    def candidate_symbols(self, start: int, stop: int) -> np.ndarray:
        return np.arange(start, stop, dtype=np.int32)

    def q_targets(self, start: int, stop: int, *, cost: str = "base") -> tuple[np.ndarray, np.ndarray]:
        count = stop - start
        base = np.linspace(-0.02, 0.03, count, dtype=np.float32)[:, None]
        return (
            base + np.linspace(0.0, 0.02, 59, dtype=np.float32)[None, :],
            base + np.linspace(0.0, 0.02, 60, dtype=np.float32)[None, :],
        )

    def future_ohlcva(self, date_idx: int, symbols: np.ndarray) -> np.ndarray:
        target = np.zeros((symbols.size, 60, 6), dtype=np.float32)
        target[..., 0] = 0.001
        target[..., 1] = 0.002
        target[..., 2] = -0.001
        target[..., 3] = 0.001
        return target

    def sequence_symbols(
        self,
        *,
        date_idx: int,
        symbols: np.ndarray,
        profile: str,
        normalization: object,
    ) -> np.ndarray:
        generator = np.random.default_rng(int(date_idx) * 100 + int(symbols[0]))
        return generator.normal(size=(symbols.size, 12, 8)).astype(np.float32)


def test_gradient_cache_uses_one_complete_day_and_backpropagates() -> None:
    model = QCurveModel(input_dim=8, hidden_dim=12, layers=1, dropout=0.0, encoder_type="gru")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    fold = {
        "normalization": {
            "inputs": {},
            "targets": {
                "enter": {"median": [0.0] * 59, "scale": [0.05] * 59},
                "hold": {"median": [0.0] * 60, "scale": [0.05] * 60},
            },
        }
    }
    span = {"date_idx": 20, "candidate_start": 0, "candidate_stop": 7}
    before = [parameter.detach().clone() for parameter in model.parameters()]
    parts, diagnostics = _train_gradient_cache_day(
        model=model,
        optimizer=optimizer,
        pack=_TinyPack(),
        fold=fold,
        span=span,
        profile="qcurve_gru",
        device=torch.device("cpu"),
        microbatch_size=3,
        loss_scales={
            "mean_loss": 1.0,
            "quantile_loss": 1.0,
            "positive_loss": 1.0,
            "soft_exit_loss": 1.0,
            "rank_loss": 1.0,
            "structured_price_trend_aux_loss": 1.0,
            "va_vwap_aux_loss": 1.0,
        },
        epoch=1,
        seed=7,
        record_gradient_diagnostics=True,
    )
    assert np.isfinite(parts["total_loss"])
    assert diagnostics is not None
    assert 0.0 <= diagnostics["auxiliary_gradient_fraction"] <= 1.0
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_lgbm_anchor_interpolation_and_quantile_projection() -> None:
    values = np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32)
    interpolated = _interpolate(values, (1, 3, 5), (1, 2, 3, 4, 5))
    assert interpolated.tolist() == [[0.0, 0.5, 1.0, 1.5, 2.0]]

    anchors: dict[str, np.ndarray] = {}
    for action, horizons in (("enter", ENTER_ANCHORS), ("hold", HOLD_ANCHORS)):
        count = len(horizons)
        anchors[f"{action}_mean"] = np.zeros((2, count), dtype=np.float32)
        anchors[f"{action}_q20"] = np.ones((2, count), dtype=np.float32)
        anchors[f"{action}_q50"] = np.zeros((2, count), dtype=np.float32)
        anchors[f"{action}_q80"] = -np.ones((2, count), dtype=np.float32)
        anchors[f"{action}_p_positive"] = np.full((2, count), 0.5, dtype=np.float32)
    curves = _curves_from_anchors(anchors)
    for action in ("enter", "hold"):
        assert np.all(curves[f"{action}_q20"] <= curves[f"{action}_q50"])
        assert np.all(curves[f"{action}_q50"] <= curves[f"{action}_q80"])


def test_stateful_execution_respects_lots_costs_and_unsellable_lock() -> None:
    open_raw = np.asarray([[10.0], [10.0], [11.0]], dtype=np.float32)
    pack = SimpleNamespace(
        execution={"open_raw": open_raw, "signal_close_raw": open_raw},
        masks={
            "open_buyable": np.asarray([[True], [True], [True]]),
            "open_sellable": np.asarray([[True], [True], [False]]),
        },
        date_values=np.asarray(["2024-01-02", "2024-01-03", "2024-01-04"], dtype="datetime64[D]"),
    )
    state = PortfolioState()
    _, trades, _ = _execute_targets(
        pack,
        state,
        execution_date_idx=1,
        targets={0: 0.6},
        contract=CONTRACT,
        slippage_multiplier=1.0,
    )
    assert len(trades) == 1 and trades[0]["side"] == "buy"
    assert state.positions[0].shares % 100 == 0
    bought = state.positions[0].shares
    assert state.fees_paid > 0.0

    _, trades, _ = _execute_targets(
        pack,
        state,
        execution_date_idx=2,
        targets={},
        contract=CONTRACT,
        slippage_multiplier=1.0,
    )
    assert trades == []
    assert state.positions[0].shares == bought


def test_mandatory_exit_turnover_gate_is_strict_json_serializable() -> None:
    allocation = pd.DataFrame(
        [
            {
                "symbol": "0",
                "is_held": True,
                "sellable_next_open": True,
                "path_value": float("-inf"),
                "current_weight": 0.5,
                "target_weight": 0.0,
                "selected": False,
            }
        ]
    )
    _, diagnostics = _apply_turnover_gate(
        allocation,
        equity=1_000_000.0,
        contract=CONTRACT,
        slippage_multiplier=1.0,
    )
    assert diagnostics["expected_gain"] is None
    json.dumps(diagnostics, allow_nan=False)


def test_minimum_complete_epoch_allows_one_recorded_diagnostic_extension() -> None:
    assert not _early_stopping_reached(
        wait=2,
        patience=2,
        completed_epoch=5,
        minimum_complete_epochs=6,
    )
    assert _early_stopping_reached(
        wait=2,
        patience=2,
        completed_epoch=6,
        minimum_complete_epochs=6,
    )
    command = _job_command(
        {
            "profile": "qcurve_gru",
            "fold_path": "fold.json",
            "output_dir": "output",
            "max_epochs": 6,
            "minimum_complete_epochs": 6,
        },
        python="C:/Users/ASUS/miniconda3/envs/yolos/python.exe",
    )
    assert command[command.index("--max-epochs") + 1] == "6"
    assert command[command.index("--minimum-complete-epochs") + 1] == "6"


def test_grouped_projection_rejects_wrong_dimensions() -> None:
    with pytest.raises(ValueError, match="summing to input_dim"):
        QCurveModel(input_dim=10, encoder_type="multiscale_tcn", input_group_dims=(4, 4, 4))


def test_selection_requires_all_frozen_gates_and_uses_primary_order(tmp_path) -> None:
    entries = {}
    scores = {"qcurve_lgbm": 0.01, "qcurve_gru": 0.02, "qcurve_multiscale_ma": 0.03}
    for profile in PROFILES:
        for year in DEVELOPMENT_YEARS:
            evaluation = {
                "portfolio": {
                    "base": {
                        "total_log_return": scores[profile],
                        "net_return": float(np.expm1(scores[profile])),
                        "alpha_total_log_return": scores[profile] / 2,
                        "turnover": 1.0,
                        "maximum_drawdown": -0.1,
                    },
                    "stress": {
                        "total_log_return": scores[profile] / 2,
                        "net_return": float(np.expm1(scores[profile] / 2)),
                    },
                },
                "topk": {
                    "top3": {
                        "cost_adjusted_absolute_mean_log_return": 0.01,
                        "cost_adjusted_alpha_mean_log_return": 0.005,
                    }
                },
                "calibration": {
                    "quantile_coverage": {
                        "enter": {"q20": {"actual_breach_rate": 0.20, "count": 100}}
                    }
                },
                "diagnostics": {
                    "candidate_coverage": 1.0,
                    "execution_mask_coverage": 1.0,
                    "q_label_coverage": 1.0,
                    "prediction_coverage": 1.0,
                },
            }
            evaluation_path = tmp_path / f"{profile}_{year}.json"
            evaluation_path.write_text(__import__("json").dumps(evaluation), encoding="utf-8")
            key = f"{profile}:development{year}:seed7"
            entries[key] = {"status": "completed", "evaluation_path": str(evaluation_path)}
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        __import__("json").dumps({"root": str(tmp_path), "entries": entries}),
        encoding="utf-8",
    )
    result = select_profiles(registry_path=registry_path)
    assert result["winner"] == "qcurve_multiscale_ma"
    selection = __import__("json").loads((tmp_path / "selection.json").read_text(encoding="utf-8"))
    assert selection["freeze_allowed"] is True
