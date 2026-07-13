from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from daily_research.path_policy.seq100_qcurve import QCurveModel
from daily_research.path_policy.seq100_qcurve_qonly_development import _deployment_epoch
from daily_research.path_policy.seq100_qcurve_qonly_development import current_contract
from daily_research.path_policy.seq100_qcurve_qonly_training import (
    NormalizedInputCache,
    QONLY_LOSS_WEIGHT_MAP,
    _checkpoint_contract,
    _model_config,
    _objective_digest,
    _set_seed,
    _validate_checkpoint_contract,
)


def test_normalized_cache_window_matches_reference_without_second_host_copy() -> None:
    cache = NormalizedInputCache.__new__(NormalizedInputCache)
    cache.panel = np.arange(110 * 7 * 3, dtype=np.float32).reshape(110, 7, 3)
    symbols = np.asarray([0, 2, 6], dtype=np.int32)
    actual = cache.sequence_symbols(date_idx=105, symbols=symbols)
    expected = np.ascontiguousarray(cache.panel[6:106][:, symbols, :].transpose(1, 0, 2))
    assert np.array_equal(actual, expected)
    assert not actual.flags.c_contiguous


def test_qonly_weights_preserve_primary_ratios_and_sum_to_one() -> None:
    assert sum(QONLY_LOSS_WEIGHT_MAP.values()) == pytest.approx(1.0)
    assert QONLY_LOSS_WEIGHT_MAP["mean_loss"] / QONLY_LOSS_WEIGHT_MAP["quantile_loss"] == pytest.approx(1.25)
    assert QONLY_LOSS_WEIGHT_MAP["rank_loss"] == pytest.approx(QONLY_LOSS_WEIGHT_MAP["mean_loss"])
    assert set(QONLY_LOSS_WEIGHT_MAP) == {
        "mean_loss",
        "quantile_loss",
        "positive_loss",
        "soft_exit_loss",
        "rank_loss",
    }


def test_qonly_model_does_not_create_or_emit_auxiliary_heads() -> None:
    model = QCurveModel(
        input_dim=38,
        hidden_dim=16,
        layers=1,
        dropout=0.0,
        encoder_type="gru",
        horizon_embedding_dim=4,
        enable_auxiliary_heads=False,
    )
    assert model.path_aux_head is None
    assert model.trend_aux_head is None
    outputs = model(torch.randn(2, 100, 38))
    assert set(outputs) == {
        f"{action}_{head}"
        for action in ("enter", "hold")
        for head in ("mean", "q20", "q50", "q80", "p_positive")
    }
    assert torch.all(outputs["enter_q20"] <= outputs["enter_q50"])
    assert torch.all(outputs["enter_q50"] <= outputs["enter_q80"])


def test_qonly_objective_digest_is_profile_specific_and_stable() -> None:
    gru = _objective_digest("qcurve_gru_qonly")
    multiscale = _objective_digest("qcurve_multiscale_ma_qonly")
    assert len(gru) == 64
    assert len(multiscale) == 64
    assert gru != multiscale
    assert gru == _objective_digest("qcurve_gru_qonly")


def test_checkpoint_contract_mismatch_is_rejected() -> None:
    expected = {"profile": "qcurve_gru_qonly", "seed": 7}
    _validate_checkpoint_contract({"checkpoint_contract": expected}, expected)
    with pytest.raises(ValueError, match="contract mismatch"):
        _validate_checkpoint_contract(
            {"checkpoint_contract": {"profile": "qcurve_gru_qonly", "seed": 8}},
            expected,
        )


def test_deployment_epoch_uses_complete_loss_curves_not_average_best_epoch(tmp_path: Path) -> None:
    losses = {
        2022: [0.7035438372, 0.6999893597, 0.6925439308, 0.7088007457, 0.6935246188],
        2023: [0.6286062687, 0.6281935819, 0.6236575817, 0.6225604629, 0.6276041828],
        2024: [0.7608303620, 0.7572948226, 0.7481532520, 0.7643351794, 0.7387726373],
        2025: [0.6427550480, 0.6386324302, 0.6237651637, 0.6961965482, 0.6319249213],
    }
    entries = {}
    for year, curve in losses.items():
        path = tmp_path / f"history_{year}.json"
        path.write_text(
            json.dumps(
                {
                    "history": [
                        {"epoch": epoch, "development_total_loss": value}
                        for epoch, value in enumerate(curve, start=1)
                    ]
                }
            ),
            encoding="utf-8",
        )
        entries[f"qcurve_gru_qonly:development{year}:seed7"] = {"history_path": str(path)}
    result = _deployment_epoch("qcurve_gru_qonly", entries)
    assert result["best_epoch_by_year"] == {"2022": 3, "2023": 4, "2024": 5, "2025": 3}
    assert result["deployment_epoch"] == 3
    assert result["fixed_epoch_fold_replay"] is False


def test_qonly_contract_has_no_test_or_multiseed(tmp_path: Path) -> None:
    contract = current_contract(root=tmp_path)
    assert contract["training"]["historical_test"] is None
    assert contract["training"]["multi_seed"] is False
    assert contract["objective"]["auxiliary_heads"] is False
    assert contract["deployment_epoch"]["fixed_epoch_fold_replay"] is False


def test_qonly_model_configs_keep_frozen_encoder_shapes() -> None:
    class Pack:
        @staticmethod
        def sequence_group_dims(profile: str) -> tuple[int, int, int]:
            return (9, 4, 25) if profile == "qcurve_gru" else (19, 8, 26)

    gru = _model_config(Pack(), "qcurve_gru_qonly")  # type: ignore[arg-type]
    multiscale = _model_config(Pack(), "qcurve_multiscale_ma_qonly")  # type: ignore[arg-type]
    assert gru["input_dim"] == 38 and gru["encoder_type"] == "gru"
    assert gru["enable_auxiliary_heads"] is False and gru["input_group_dims"] is None
    assert multiscale["input_dim"] == 53 and multiscale["encoder_type"] == "multiscale_tcn"
    assert multiscale["input_group_dims"] == (19, 8, 26)


def test_qonly_fast_cudnn_is_explicit_not_accidental() -> None:
    _set_seed(7, fast_cudnn=True)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True
    _set_seed(7, fast_cudnn=False)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_checkpoint_contract_records_numeric_and_cache_semantics() -> None:
    class Cache:
        digest = "cache-digest"

    fold = {"fold_contract_sha256": "fold-digest"}
    contract = _checkpoint_contract(
        profile="qcurve_gru_qonly",
        fold=fold,
        input_cache=Cache(),  # type: ignore[arg-type]
        precision="amp_fp16",
        microbatch_size=1024,
        seed=7,
    )
    assert contract["fold_contract_sha256"] == "fold-digest"
    assert contract["input_cache_digest"] == "cache-digest"
    assert contract["precision"] == "amp_fp16"
    assert contract["microbatch_size"] == 1024
