from __future__ import annotations

from pathlib import Path

import pandas as pd

from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.generate_daily_trade_plan import (
    _assess_external_model_retrain_freshness,
    _build_scores_from_external_target_weight_csv,
    _load_external_panel_row,
    _validate_external_signal_panels_fresh,
)


def test_external_target_weights_carry_latest_valid_signal_to_market_date(tmp_path: Path) -> None:
    weights = tmp_path / "weights.csv"
    weights.write_text(
        "\n".join(
            [
                "date,stock,target_weight",
                "2026-05-19,000001.SZ,0.10",
                "2026-05-19,600000.SH,0.20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    scores = tmp_path / "scores.csv"
    scores.write_text(
        "\n".join(
            [
                "date,stock,score",
                "2026-05-19,000001.SZ,0.70",
                "2026-05-19,600000.SH,0.80",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    close = pd.DataFrame(
        {
            "000001.SZ": [10.0, 10.1, 10.2],
            "600000.SH": [20.0, 20.1, 20.2],
        },
        index=pd.to_datetime(["2026-05-19", "2026-05-20", "2026-05-22"]),
    )
    prepared_bundle = {
        "factor_bundle": {"raw_inputs": {"Close": close}},
        "regime_state": pd.DataFrame(
            {"quadrant": ["trend_up_low_vol"] * 3, "regime_on": [True] * 3},
            index=close.index,
        ),
    }

    *_, model_score, _final_score_raw, _final_score_filtered, target_weights, training_log = (
        _build_scores_from_external_target_weight_csv(
        ResearchConfig(rebalance_freq="1d", enable_market_regime_filter=False),
        prepared_bundle,
        external_target_weight_csv=weights,
        external_target_weight_column="target_weight",
        candidate_label="candidate",
        rebalance_offset=0,
        rebalance_offset_mode="single",
        rebalance_anchor_date="2026-05-19",
        target_weight_top_k=0,
        target_weight_min_weight=0.0,
        target_weight_power=1.0,
        target_weight_full_invest=False,
        external_score_csv=scores,
        external_score_column="score",
    )
    )

    latest = pd.Timestamp("2026-05-22")
    assert pd.Timestamp(target_weights.index.max()) == latest
    assert target_weights.loc[latest, "000001.SZ"] == 0.10
    assert target_weights.loc[latest, "600000.SH"] == 0.20
    assert training_log.iloc[0]["source_signal_date"] == "2026-05-19"
    assert training_log.iloc[0]["signal_date"] == "2026-05-22"
    assert training_log.iloc[0]["score_context_status"] == "external_score"
    assert training_log.iloc[0]["score_context_warning"] == ""
    assert model_score.loc[latest, "000001.SZ"] == 0.70
    assert model_score.loc[latest, "600000.SH"] == 0.80


def test_external_panel_row_uses_latest_non_empty_row_before_requested_date(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    scores.write_text(
        "\n".join(
            [
                "date,stock,score",
                "2026-05-19,000001.SZ,0.70",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    row, signal_date = _load_external_panel_row(
        panel_csv=scores,
        value_column="score",
        panel_label="external score",
        value_name="score",
        latest_market_date=pd.Timestamp("2026-05-22"),
        requested_date=pd.Timestamp("2026-05-22"),
        market_index=pd.DatetimeIndex(pd.to_datetime(["2026-05-19", "2026-05-20", "2026-05-22"])),
        allowed_columns=pd.Index(["000001.SZ", "600000.SH"]),
    )

    assert signal_date == pd.Timestamp("2026-05-19")
    assert row["000001.SZ"] == 0.70
    assert pd.isna(row["600000.SH"])


def test_external_signal_panel_preflight_blocks_stale_default_trade_plan(tmp_path: Path) -> None:
    weights = tmp_path / "weights.csv"
    weights.write_text("date,stock,target_weight\n2026-05-19,000001.SZ,0.10\n", encoding="utf-8")
    scores = tmp_path / "scores.csv"
    scores.write_text("date,stock,score\n2026-05-19,000001.SZ,0.70\n", encoding="utf-8")

    try:
        _validate_external_signal_panels_fresh(
            target_weight_csv=weights,
            score_csv=scores,
            required_date=pd.Timestamp("2026-05-22"),
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("stale production signal panels must block default trade-plan generation")

    assert "生产信号面板已过期" in message
    assert "2026-05-19" in message
    assert "2026-05-22" in message


def test_external_model_without_retrain_thresholds_is_informational(tmp_path: Path) -> None:
    manifest = tmp_path / "production_retrain_manifest.json"
    manifest.write_text(
        "{"
        '"train_end_date": "2026-04-03",'
        '"launch_cutoff_date": "2026-04-21",'
        '"retrain_frequency_policy": {"auto_retrain_enabled": false}'
        "}",
        encoding="utf-8",
    )

    payload = _assess_external_model_retrain_freshness(
        manifest_path=manifest,
        latest_signal_date=pd.Timestamp("2026-05-22"),
        trading_dates=pd.DatetimeIndex(pd.to_datetime(["2026-04-21", "2026-05-22"])),
        warn_trading_days=0,
        max_trading_days=0,
    )

    assert payload["status_text"] == "informational"
    assert payload["trading_day_lag"] == 1
    assert payload["warnings"] == []
