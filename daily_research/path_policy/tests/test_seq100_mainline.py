from __future__ import annotations

import json
from pathlib import Path

import daily_research.path_policy as path_policy
from daily_research.path_policy.seq100_mainline import (
    ACTIVE_CONCEPTS,
    ARCHIVED_CONCEPTS,
    DEFAULT_DAILY_ONLY_RUN_TAG,
    DEFAULT_DIRECT_VALUE_5D_RUN_TAG,
    DEFAULT_STORE_VIEW,
    DEFAULT_SUMMARY_V2_RUN_TAG,
    TodayClosePathOnlyProfile,
    build_todayclose_daily_only_train_argv,
    build_todayclose_direct_value_train_argv,
    build_todayclose_summary_v2_train_argv,
    build_todayclose_path_only_train_argv,
    main,
    mainline_contract,
    summarize_sequence_run,
)


def test_mainline_contract_keeps_default_surface_small() -> None:
    contract = mainline_contract()

    assert contract["mainline_id"] == "seq100_todayclose_path_only"
    assert contract["default_store_view"] == str(DEFAULT_STORE_VIEW)
    assert set(contract["active_concepts"]) == set(ACTIVE_CONCEPTS)
    assert contract["default_train_profile"]["early_stopping_patience"] == 3
    assert contract["summary_v2_train_profile"]["summary_loss_profile"] == "multi_horizon_ohlc"
    assert contract["summary_v2_train_profile"]["early_stopping_patience"] == 2
    assert contract["daily_only_train_profile"]["input_channel_profile"] == "daily_only"
    assert contract["daily_only_train_profile"]["early_stopping_patience"] == 2
    assert contract["direct_value_train_profiles"]["5d"]["model_type"] == "gru_direct_value"
    assert contract["direct_value_train_profiles"]["5d"]["direct_value_horizon"] == 5
    assert contract["direct_value_train_profiles"]["5d"]["batch_size"] == 2048
    assert contract["direct_value_train_profiles"]["5d"]["early_stopping_patience"] == 2
    assert contract["direct_value_train_profiles"]["10d"]["direct_value_horizon"] == 10
    assert contract["direct_value_train_profiles"]["60d"]["direct_value_horizon"] == 60
    assert "symbol_embedding" in ARCHIVED_CONCEPTS
    assert "active_execution_strategy.json" in contract["evidence_boundary"]


def test_path_policy_package_default_points_to_seq100_mainline() -> None:
    assert path_policy.PATH_POLICY_DEFAULT_MAINLINE == "seq100_todayclose_path_only"
    assert path_policy.SEQ100_MAINLINE_PROFILE == "seq100_todayclose_path_only_mainline"


def test_todayclose_path_only_train_argv_has_no_experimental_branches() -> None:
    profile = TodayClosePathOnlyProfile(run_tag="unit_mainline", epochs=1, max_samples_per_split=32, device="cpu")
    argv = build_todayclose_path_only_train_argv(profile)

    assert argv[:3] == ["train", "--store-view", str(DEFAULT_STORE_VIEW)]
    assert "--model-type" in argv
    assert argv[argv.index("--model-type") + 1] == "gru_path_value"
    assert argv[argv.index("--path-loss-weight") + 1] == "0.45"
    assert argv[argv.index("--rank-loss-weight") + 1] == "0.15"
    assert argv[argv.index("--summary-loss-profile") + 1] == "base"
    assert argv[argv.index("--input-channel-profile") + 1] == "all"
    assert argv[argv.index("--prediction-mode") + 1] == "compact"
    assert argv[argv.index("--top-k") + 1] == "1,3,5,10,20,50,100"
    assert "--with-symbol" not in argv
    assert "--richer-path" not in argv
    assert "gru_path_value_residual" not in argv
    assert "gru_ohlcva_path_value" not in argv


def test_todayclose_summary_v2_train_argv_changes_only_summary_profile() -> None:
    profile = TodayClosePathOnlyProfile(
        run_tag=DEFAULT_SUMMARY_V2_RUN_TAG,
        epochs=1,
        max_samples_per_split=32,
        device="cpu",
        early_stopping_patience=2,
    )
    argv = build_todayclose_summary_v2_train_argv(profile)

    assert argv[argv.index("--run-tag") + 1] == DEFAULT_SUMMARY_V2_RUN_TAG
    assert argv[argv.index("--model-type") + 1] == "gru_path_value"
    assert argv[argv.index("--summary-loss-profile") + 1] == "multi_horizon_ohlc"
    assert argv[argv.index("--input-channel-profile") + 1] == "all"
    assert argv[argv.index("--early-stopping-patience") + 1] == "2"
    assert "--with-symbol" not in argv
    assert "--richer-path" not in argv
    assert "gru_path_value_residual" not in argv
    assert "gru_ohlcva_path_value" not in argv


def test_todayclose_daily_only_train_argv_removes_minute_channels() -> None:
    profile = TodayClosePathOnlyProfile(
        run_tag=DEFAULT_DAILY_ONLY_RUN_TAG,
        epochs=1,
        max_samples_per_split=32,
        device="cpu",
        early_stopping_patience=2,
    )
    argv = build_todayclose_daily_only_train_argv(profile)

    assert argv[argv.index("--run-tag") + 1] == DEFAULT_DAILY_ONLY_RUN_TAG
    assert argv[argv.index("--model-type") + 1] == "gru_path_value"
    assert argv[argv.index("--summary-loss-profile") + 1] == "base"
    assert argv[argv.index("--input-channel-profile") + 1] == "daily_only"
    assert argv[argv.index("--early-stopping-patience") + 1] == "2"
    assert "--with-symbol" not in argv
    assert "--richer-path" not in argv


def test_todayclose_direct_value_train_argv_switches_to_score_ranker() -> None:
    profile = TodayClosePathOnlyProfile(
        run_tag=DEFAULT_DIRECT_VALUE_5D_RUN_TAG,
        epochs=1,
        max_samples_per_split=32,
        device="cpu",
        early_stopping_patience=2,
    )
    argv = build_todayclose_direct_value_train_argv(profile, horizon=5)

    assert argv[argv.index("--run-tag") + 1] == DEFAULT_DIRECT_VALUE_5D_RUN_TAG
    assert argv[argv.index("--model-type") + 1] == "gru_direct_value"
    assert argv[argv.index("--direct-value-horizon") + 1] == "5"
    assert argv[argv.index("--path-loss-weight") + 1] == "0.0"
    assert argv[argv.index("--summary-loss-weight") + 1] == "0.0"
    assert argv[argv.index("--value-loss-weight") + 1] == "0.5"
    assert argv[argv.index("--rank-loss-weight") + 1] == "0.5"
    assert argv[argv.index("--early-stopping-patience") + 1] == "2"
    assert argv[argv.index("--input-channel-profile") + 1] == "all"
    assert "--with-symbol" not in argv
    assert "--richer-path" not in argv


def test_train_summary_v2_cli_defaults_patience_to_two(capsys) -> None:
    assert main(["train", "--dry-run", "--json"]) == 0
    base = json.loads(capsys.readouterr().out)
    assert base["profile"]["early_stopping_patience"] == 3
    assert base["argv"][base["argv"].index("--early-stopping-patience") + 1] == "3"

    assert main(["train-summary-v2", "--dry-run", "--json"]) == 0
    summary_v2 = json.loads(capsys.readouterr().out)
    assert summary_v2["profile"]["early_stopping_patience"] == 2
    assert summary_v2["profile"]["summary_loss_profile"] == "multi_horizon_ohlc"
    assert summary_v2["argv"][summary_v2["argv"].index("--early-stopping-patience") + 1] == "2"

    assert main(["train-daily-only", "--dry-run", "--json"]) == 0
    daily_only = json.loads(capsys.readouterr().out)
    assert daily_only["profile"]["early_stopping_patience"] == 2
    assert daily_only["profile"]["input_channel_profile"] == "daily_only"
    assert daily_only["argv"][daily_only["argv"].index("--input-channel-profile") + 1] == "daily_only"

    assert main(["train-direct-value-5d", "--dry-run", "--json"]) == 0
    direct_5d = json.loads(capsys.readouterr().out)
    assert direct_5d["profile"]["early_stopping_patience"] == 2
    assert direct_5d["profile"]["model_type"] == "gru_direct_value"
    assert direct_5d["profile"]["direct_value_horizon"] == 5
    assert direct_5d["argv"][direct_5d["argv"].index("--model-type") + 1] == "gru_direct_value"
    assert direct_5d["argv"][direct_5d["argv"].index("--direct-value-horizon") + 1] == "5"
    assert direct_5d["argv"][direct_5d["argv"].index("--batch-size") + 1] == "2048"
    assert direct_5d["argv"][direct_5d["argv"].index("--early-stopping-patience") + 1] == "2"

    assert main(["train-direct-value-10d", "--dry-run", "--json"]) == 0
    direct_10d = json.loads(capsys.readouterr().out)
    assert direct_10d["profile"]["direct_value_horizon"] == 10
    assert direct_10d["argv"][direct_10d["argv"].index("--direct-value-horizon") + 1] == "10"

    assert main(["train-direct-value-60d", "--dry-run", "--json"]) == 0
    direct_60d = json.loads(capsys.readouterr().out)
    assert direct_60d["profile"]["direct_value_horizon"] == 60
    assert direct_60d["argv"][direct_60d["argv"].index("--direct-value-horizon") + 1] == "60"


def test_summarize_sequence_run_reads_only_compact_metrics(tmp_path: Path) -> None:
    (tmp_path / "sequence_path_training_summary.json").write_text(
        """{
  "generated_at": "2026-07-07T00:00:00+08:00",
  "pack_manifest": "daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva.json",
  "price_anchor": "today_close",
  "value_column": "path_trade_value_v2_60d",
  "best_epoch": 1,
  "loss_weights": {"path": 0.45, "summary": 0.2, "value": 0.2, "rank": 0.15}
}""",
        encoding="utf-8",
    )
    (tmp_path / "split_metrics.csv").write_text(
        "split,row_count,date_count,rank_ic_mean,rank_ic_positive_day_rate,value_column\n"
        "validation,100,10,0.12,0.7,path_trade_value_v2_60d\n"
        "test,110,11,0.13,0.8,path_trade_value_v2_60d\n",
        encoding="utf-8",
    )
    (tmp_path / "topk_metrics.csv").write_text(
        "split,top_k,alpha_path_trade_value_v2_60d,selected_best_exit_day_mean,selected_hit_10pct_rate,selected_loss_5pct_rate\n"
        "validation,1,0.18,24.5,0.77,0.80\n"
        "test,10,0.10,28.4,0.85,0.70\n",
        encoding="utf-8",
    )

    summary = summarize_sequence_run(tmp_path)

    assert summary["price_anchor"] == "today_close"
    assert summary["splits"]["validation"]["rank_ic_mean"] == 0.12
    assert summary["topk"]["validation"]["1"]["path_value_spread"] == 0.18
    assert summary["topk"]["test"]["10"]["best_exit_day_mean"] == 28.4


def test_summarize_sequence_run_reads_direct_value_horizon_metric(tmp_path: Path) -> None:
    (tmp_path / "sequence_path_training_summary.json").write_text(
        """{
  "generated_at": "2026-07-08T00:00:00+08:00",
  "pack_manifest": "daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva.json",
  "input_channel_profile": "all",
  "price_anchor": "today_close",
  "value_column": "path_trade_value_v2_5d",
  "direct_value_horizon": 5,
  "best_epoch": 1,
  "model": {"input_dim": 46, "uses_direct_value": true},
  "loss_weights": {"path": 0.0, "summary": 0.0, "value": 0.5, "rank": 0.5}
}""",
        encoding="utf-8",
    )
    (tmp_path / "split_metrics.csv").write_text(
        "split,row_count,date_count,rank_ic_mean,rank_ic_positive_day_rate,value_column\n"
        "test,110,11,0.13,0.8,path_trade_value_v2_5d\n",
        encoding="utf-8",
    )
    (tmp_path / "topk_metrics.csv").write_text(
        "split,top_k,alpha_path_trade_value_v2_5d,selected_best_exit_day_mean,selected_hit_10pct_rate,selected_loss_5pct_rate\n"
        "test,1,0.03,3.4,0.45,0.20\n",
        encoding="utf-8",
    )

    summary = summarize_sequence_run(tmp_path)

    assert summary["value_column"] == "path_trade_value_v2_5d"
    assert summary["topk"]["test"]["1"]["path_value_spread"] == 0.03
