from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from daily_research.path_policy import seq100_short_horizon_target_reaudit as reaudit


def test_d3_numeric_labels_obey_t_plus_one_and_pre_peak_adversity() -> None:
    close = np.asarray(
        [
            [-0.02, 0.08, 0.03],
            [0.01, 0.12, 0.05],
        ],
        dtype=np.float64,
    )
    sellable = np.asarray(
        [
            [True, True, True],
            [True, False, True],
        ],
        dtype=bool,
    )

    g, mfe, mae, has_sellable = reaudit._numeric_path_labels(
        close, sellable, np.asarray([True, True])
    )

    assert g.tolist() == pytest.approx([0.03, 0.05])
    assert mfe.tolist() == pytest.approx([0.08, 0.05])
    assert mae.tolist() == pytest.approx([-0.02, 0.0])
    assert has_sellable.tolist() == [True, True]


def test_short_batch_keeps_d1_separate_from_d3_sellability() -> None:
    future = np.zeros((2, 3, 4), dtype=np.float64)
    future[:, :, 2] = np.asarray([[0.00, -0.01, 0.00], [0.00, 0.00, 0.00]])
    future[:, :, 3] = np.asarray([[0.02, 0.05, 0.03], [0.01, 0.12, 0.05]])
    source = SimpleNamespace(
        cutoff_idx=3,
        entry_filled=np.asarray([[True, True]], dtype=bool),
        exit_sellable=np.asarray(
            [
                [True, True],
                [True, True],
                [True, False],
                [True, True],
            ],
            dtype=bool,
        ),
        future_ohlc_prefix=lambda _date, symbols, horizon: future[
            np.asarray(symbols), :horizon
        ],
    )
    model = {
        "clip_low": np.full(3, -1.0, dtype=np.float32),
        "clip_high": np.full(3, 1.0, dtype=np.float32),
        "center": np.zeros(3, dtype=np.float32),
        "scale": np.ones(3, dtype=np.float32),
        "pca_mean": np.zeros(3, dtype=np.float32),
        "pca_components": np.eye(3, dtype=np.float32),
        "fixed_k3_centers": np.asarray(
            [[-0.1, -0.1, -0.1], [0.0, 0.0, 0.0], [0.1, 0.1, 0.1]],
            dtype=np.float32,
        ),
        "raw_to_state": np.asarray([0, 1, 2], dtype=np.int8),
    }

    labels, states, flags = reaudit._derive_short_batch(
        source=source,
        d3_model=model,
        date_idx=0,
        symbol_idx=np.asarray([0, 1]),
    )

    assert labels[:, reaudit.SHORT_LABEL_INDEX["g_1"]].tolist() == pytest.approx(
        [0.02, 0.01]
    )
    assert labels[:, reaudit.SHORT_LABEL_INDEX["g_3"]].tolist() == pytest.approx(
        [0.03, 0.05]
    )
    assert labels[:, reaudit.SHORT_LABEL_INDEX["mfe_3"]].tolist() == pytest.approx(
        [0.05, 0.05]
    )
    assert states.tolist() == [1, 2]
    assert bool(flags[0, 0] & reaudit.FLAG_G_VALID)
    assert bool(flags[1, 1] & reaudit.FLAG_MFE_VALID)


def test_revised_regression_gate_does_not_reject_one_imperfect_decile_year() -> None:
    annual = [
        {
            "rank_ic": rank_ic,
            "top_1pct_lift": 0.02,
            "top_5pct_lift": 0.01,
            "top_bottom_spread": 0.01,
            "decile_spearman": decile,
            "rank_ic_hac_p_value": 0.01,
        }
        for rank_ic, decile in ((0.21, 0.78), (0.13, 0.95), (0.09, 0.418))
    ]

    gate = reaudit._regression_gate(
        annual,
        {
            "minimum_worst_year_rank_ic": 0.01,
            "minimum_hac_supported_years": 2,
            "minimum_top1_positive_years": 2,
            "minimum_monotone_years": 2,
            "minimum_decile_spearman": 0.5,
            "minimum_no_reversal_decile_spearman": -0.2,
        },
        require_upper_tail=False,
    )

    assert gate["monotone_years"] == 2
    assert gate["no_material_decile_reversal"] is True
    assert gate["qualified"] is True


def test_mfe_gate_uses_realized_upper_tail_not_mfe_above_zero() -> None:
    annual = [
        {
            "rank_ic": 0.08,
            "top_1pct_lift": 0.04,
            "top_5pct_lift": 0.03,
            "top_bottom_spread": 0.03,
            "decile_spearman": 0.8,
            "rank_ic_hac_p_value": 0.01,
            "daily_tail_top5_lift": 0.2,
        }
        for _year in range(3)
    ]

    gate = reaudit._regression_gate(
        annual, {}, require_upper_tail=True
    )

    assert gate["all_years_daily_top_quintile_enrichment"] is True
    assert gate["qualified"] is True


def test_sequential_temperature_scaling_softens_overconfident_probabilities() -> None:
    probability = np.asarray(
        [
            [0.99, 0.005, 0.005],
            [0.99, 0.005, 0.005],
            [0.005, 0.99, 0.005],
            [0.005, 0.005, 0.99],
        ],
        dtype=np.float64,
    )
    actual = np.asarray([0, 1, 1, 2], dtype=np.int8)
    weights = np.ones(4, dtype=np.float64)

    fit = reaudit._fit_temperature(probability, actual, weights)
    calibrated = reaudit._temperature_scale(probability, fit["temperature"])
    raw_loss = -float(
        np.mean(np.log(probability[np.arange(len(actual)), actual]))
    )
    calibrated_loss = -float(
        np.mean(np.log(calibrated[np.arange(len(actual)), actual]))
    )

    assert fit["temperature"] > 1.0
    assert calibrated_loss < raw_loss


def test_daily_tertiles_and_contingency_are_label_permutation_invariant() -> None:
    dates = np.repeat(np.asarray([0, 1], dtype=np.int32), 6)
    values = np.tile(np.arange(6, dtype=np.float64), 2)
    tertile = reaudit._daily_tertiles(
        date_idx=dates,
        values=values,
        valid=np.ones(len(values), dtype=bool),
    )
    permuted = np.asarray([2, 1, 0], dtype=np.int8)[tertile]

    metrics = reaudit._contingency_metrics(tertile, permuted)

    assert tertile.tolist() == [0, 0, 1, 1, 2, 2] * 2
    assert metrics["adjusted_rand_index"] == pytest.approx(1.0)
    assert metrics["normalized_mutual_information"] == pytest.approx(1.0)
