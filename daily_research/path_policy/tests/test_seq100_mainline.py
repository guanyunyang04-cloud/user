from __future__ import annotations

from pathlib import Path

import daily_research.path_policy as path_policy
from daily_research.path_policy.seq100_mainline import (
    ACTIVE_CONCEPTS,
    ARCHIVED_CONCEPTS,
    DEFAULT_STORE_VIEW,
    TodayClosePathOnlyProfile,
    build_todayclose_path_only_train_argv,
    mainline_contract,
    summarize_sequence_run,
)


def test_mainline_contract_keeps_default_surface_small() -> None:
    contract = mainline_contract()

    assert contract["mainline_id"] == "seq100_todayclose_path_only"
    assert contract["default_store_view"] == str(DEFAULT_STORE_VIEW)
    assert set(contract["active_concepts"]) == set(ACTIVE_CONCEPTS)
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
    assert argv[argv.index("--prediction-mode") + 1] == "compact"
    assert argv[argv.index("--top-k") + 1] == "1,3,5,10,20,50,100"
    assert "--with-symbol" not in argv
    assert "--richer-path" not in argv
    assert "gru_path_value_residual" not in argv
    assert "gru_ohlcva_path_value" not in argv


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
