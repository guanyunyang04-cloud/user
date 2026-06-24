from __future__ import annotations

from pathlib import Path

import pandas as pd

from daily_research.path_policy.shortline_upside_feature_profile import (
    build_shortline_upside_feature_profile_from_frame,
    classify_feature_semantics,
)


def _profile_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for date in ("2025-01-02", "2025-01-03", "2025-01-06"):
        for idx in range(20):
            strength = float(idx)
            future = (strength - 9.5) / 100.0
            rows.append(
                {
                    "role": "validation",
                    "date": date,
                    "stock": f"S{idx:03d}",
                    "good_signal": strength,
                    "bad_signal": -strength,
                    "flat_signal": 1.0,
                    "future_cum_return_1d": future,
                    "future_cum_excess_return_1d": future - 0.002,
                    "future_cum_return_2d": future + 0.01,
                    "future_cum_excess_return_2d": future + 0.008,
                }
            )
    return rows


def test_upside_profile_finds_positive_and_negative_feature_direction(tmp_path: Path) -> None:
    frame = pd.DataFrame(_profile_rows())

    report = build_shortline_upside_feature_profile_from_frame(
        frame=frame,
        feature_columns=("good_signal", "bad_signal", "flat_signal"),
        output_root=tmp_path / "profile",
        label_horizons=(1,),
        target_kinds=("net_abs",),
        group_scopes=("role",),
        round_trip_cost_bps=0.0,
    )

    assert report["status"] == "completed"
    metrics = pd.read_csv(report["outputs"]["feature_metrics_csv"])
    good = metrics.loc[metrics["feature"].eq("good_signal")].iloc[0]
    bad = metrics.loc[metrics["feature"].eq("bad_signal")].iloc[0]
    flat = metrics.loc[metrics["feature"].eq("flat_signal")].iloc[0]
    assert good["rank_ic_mean"] > 0.99
    assert good["top_minus_bottom_decile_target_mean"] > 0.0
    assert good["direction"] == "high_is_good"
    assert bad["rank_ic_mean"] < -0.99
    assert bad["top_minus_bottom_decile_target_mean"] < 0.0
    assert bad["direction"] == "low_is_good"
    assert abs(float(flat["rank_ic_mean"])) < 1.0e-9


def test_upside_profile_writes_deciles_and_target_summary(tmp_path: Path) -> None:
    frame = pd.DataFrame(_profile_rows())

    report = build_shortline_upside_feature_profile_from_frame(
        frame=frame,
        feature_columns=("good_signal", "bad_signal"),
        output_root=tmp_path / "profile_multi",
        label_horizons=(1, 2),
        target_kinds=("net_abs", "net_excess"),
        group_scopes=("all", "role", "year"),
        round_trip_cost_bps=20.0,
    )

    deciles = pd.read_csv(report["outputs"]["decile_profile_csv"])
    target = pd.read_csv(report["outputs"]["target_summary_csv"])
    assert set(deciles["decile"].astype(int)) == set(range(10))
    assert {"D+2_open", "D+3_open"} == set(target["exit_label"].astype(str))
    assert {"net_abs", "net_excess"} == set(target["target_kind"].astype(str))
    assert {"all", "role", "year"} == set(target["scope"].astype(str))
    assert Path(report["outputs"]["report_json"]).exists()


def test_feature_semantics_classifies_non_continuous_features() -> None:
    cases = {
        "intraday_last_30m_ret": "continuous_ordinal",
        "cs_rank_ret_20d": "already_ranked",
        "cs_z_turn": "zscore_continuous",
        "peer_adv_bucket_id": "ordinal_bucket",
        "local_high_runup_flag": "binary_flag",
        "index_000300_SH_member_lag1": "membership_flag",
        "market_positive_share_1d": "market_level",
    }

    for feature, expected in cases.items():
        assert classify_feature_semantics(feature).semantic_type == expected


def test_upside_profile_emits_type_aware_outputs(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for date_idx, date in enumerate(("2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07")):
        market_state = float(date_idx)
        for idx in range(12):
            flag = 1.0 if idx >= 6 else 0.0
            bucket = float(idx % 3)
            future = 0.01 * flag + 0.002 * bucket + 0.001 * market_state
            rows.append(
                {
                    "role": "validation",
                    "date": date,
                    "stock": f"S{idx:03d}",
                    "local_high_runup_flag": flag,
                    "peer_adv_bucket_id": bucket,
                    "market_positive_share_1d": market_state,
                    "future_cum_return_1d": future,
                    "future_cum_excess_return_1d": future,
                }
            )
    report = build_shortline_upside_feature_profile_from_frame(
        frame=pd.DataFrame(rows),
        feature_columns=("local_high_runup_flag", "peer_adv_bucket_id", "market_positive_share_1d"),
        output_root=tmp_path / "type_aware",
        label_horizons=(1,),
        target_kinds=("net_abs",),
        group_scopes=("role",),
        round_trip_cost_bps=0.0,
    )

    semantics = pd.read_csv(report["outputs"]["feature_semantics_csv"])
    categorical = pd.read_csv(report["outputs"]["categorical_profile_csv"])
    market = pd.read_csv(report["outputs"]["market_profile_csv"])
    metrics = pd.read_csv(report["outputs"]["feature_metrics_csv"])

    assert set(semantics["semantic_type"].astype(str)) == {"binary_flag", "ordinal_bucket", "market_level"}
    flag_rows = categorical.loc[categorical["feature"].eq("local_high_runup_flag")]
    assert {"0", "1"} == set(flag_rows["category_value"].astype(str))
    assert flag_rows.loc[flag_rows["category_value"].astype(str).eq("1"), "target_mean"].iloc[0] > flag_rows.loc[
        flag_rows["category_value"].astype(str).eq("0"), "target_mean"
    ].iloc[0]
    assert market.loc[market["feature"].eq("market_positive_share_1d"), "date_level_corr"].iloc[0] > 0.9
    market_metric = metrics.loc[metrics["feature"].eq("market_positive_share_1d")].iloc[0]
    assert market_metric["preferred_diagnostic"] == "date_level_target_correlation"
