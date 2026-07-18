from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

import daily_research.path_policy.seq100_walkforward as walkforward

from daily_research.path_policy.seq100_walkforward import (
    PROFILE_COMMANDS,
    STUDY_RUN_TAG,
    _resolved_profile_config,
    bind_legacy_walkforward_run_contracts,
    approved_development_contract_binding,
    build_development_walkforward_fold,
    build_purged_walkforward_fold,
    hac_mean_interval,
    moving_block_bootstrap_mean,
    paired_ic_comparison,
    paired_topk_comparison,
    reconcile_bound_walkforward_fold_metadata,
    run_walkforward_study,
    summarize_walkforward_study,
    verify_purged_walkforward_view,
    verify_development_walkforward_view,
)


DATES = [
    "2021-12-27",
    "2021-12-28",
    "2021-12-29",
    "2021-12-30",
    "2021-12-31",
    "2022-01-03",
    "2022-01-04",
    "2022-01-05",
    "2022-01-06",
    "2022-01-07",
]
SYMBOLS = ["000001.SZ", "600000.SH"]


def _write_array(path: Path, values: np.ndarray, *, dtype: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype=dtype)
    array.tofile(path)
    return {"path": str(path.resolve()), "shape": list(array.shape)}


def _build_source_view(store_root: Path) -> Path:
    arrays_root = store_root / "tiny_arrays"
    # Pre-OOS feature dates have mean=3 and std=sqrt(2). Large OOS values make
    # accidental normalization leakage immediately visible.
    date_values = np.asarray([1, 2, 3, 4, 5, 1000, 2000, 3000, 4000, 5000], dtype=np.float32)
    panel = np.repeat(date_values[:, None, None], len(SYMBOLS), axis=1)
    feature_channels: dict[str, Any] = {}
    for channel in ("daily_raw", "daily_state", "intraday_summary", "limit_structure"):
        feature_channels[channel] = {
            **_write_array(arrays_root / f"{channel}.float32.dat", panel, dtype="float32"),
            "columns": [f"{channel}_feature"],
        }

    forward_days = 2
    future_path = np.zeros((len(DATES), len(SYMBOLS), forward_days, 6), dtype=np.float32)
    path_summary = np.zeros((len(DATES), len(SYMBOLS), 9), dtype=np.float32)
    valid_mask = np.ones((len(DATES), len(SYMBOLS)), dtype=bool)
    label_arrays = {
        "future_ohlcva_path": {
            **_write_array(arrays_root / "future_ohlcva_path.float32.dat", future_path, dtype="float32"),
            "fields": ["open", "high", "low", "close", "volume", "amount"],
            "price_anchor": "today_close",
        },
        "path_summary": {
            **_write_array(arrays_root / "path_summary.float32.dat", path_summary, dtype="float32"),
            "columns": [
                "future_max_return_2d",
                "future_min_return_2d",
                "future_final_return_2d",
                "future_peak_day_2d",
                "future_trough_day_2d",
                "drawdown_after_peak_2d",
                "time_above_zero_2d",
                "time_below_zero_2d",
                "path_trade_value_2d",
            ],
        },
    }
    masks = {
        name: _write_array(arrays_root / f"{name}.bool.dat", valid_mask, dtype="bool")
        for name in (
            "input_valid",
            "entry_buyable",
            "label_valid",
            "long_suspension",
            "continuity_break",
        )
    }

    # The source split is deliberately unsuitable. The fold builder must derive
    # train/oos solely from date_idx and the future-label horizon.
    rows = [
        {
            "trade_date": DATES[date_idx],
            "date_idx": date_idx,
            "symbol": symbol,
            "symbol_idx": symbol_idx,
            "split": "legacy",
        }
        for date_idx in range(8)
        for symbol_idx, symbol in enumerate(SYMBOLS)
    ]
    sample_index_path = store_root / "sample_index" / "tiny_source.parquet"
    sample_index_path.parent.mkdir(parents=True, exist_ok=True)
    sample_frame = pd.DataFrame(rows)
    sample_frame.to_parquet(sample_index_path, index=False)
    candidate = sample_frame.copy()
    candidate["candidate_id"] = np.arange(len(candidate), dtype=np.int64)
    candidate["year"] = candidate["trade_date"].str.slice(0, 4).astype(np.int16)
    candidate["entry_trade_date"] = np.asarray(DATES, dtype=object)[
        np.minimum(candidate["date_idx"].to_numpy(dtype=np.int64) + 1, len(DATES) - 1)
    ]
    for column in ("entry_filled", "label_valid", "price_label_valid", "va_aux_valid"):
        candidate[column] = True
    candidate_index_path = store_root / "candidate_index" / "tiny_source.parquet"
    candidate_index_path.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_parquet(candidate_index_path, index=False)

    qdp_root = store_root / "tiny_qdp"
    qdp_manifest = qdp_root / "datasets" / "fixture" / "fixture-v1" / "dataset.json"
    qdp_manifest.parent.mkdir(parents=True, exist_ok=True)
    qdp_manifest.write_text('{"dataset_id":"fixture-v1"}', encoding="utf-8")
    qdp_hash = hashlib.sha256(qdp_manifest.read_bytes()).hexdigest()

    source_view = {
        "artifact_type": "qdp_v2_sequence_path_pack",
        "lookback_days": 1,
        "forward_days": forward_days,
        "start_date": DATES[0],
        "end_date": DATES[-1],
        "date_values": DATES,
        "symbol_values": SYMBOLS,
        "date_count": len(DATES),
        "symbol_count": len(SYMBOLS),
        "sample_count": len(rows),
        "sample_count_by_split": {"legacy": len(rows)},
        "feature_channels": feature_channels,
        "label_arrays": label_arrays,
        "masks": masks,
        "sample_index_path": str(sample_index_path.resolve()),
        "candidate_index_path": str(candidate_index_path.resolve()),
        "candidate_count": int(len(candidate)),
        "candidate_count_by_split": {"legacy": int(len(candidate))},
        "qdp_root": str(qdp_root.resolve()),
        "research_source_datasets": {"fixture": "fixture-v1"},
        "qdp_source_manifests": {
            "fixture": {
                "dataset_id": "fixture-v1",
                "manifest_path": str(qdp_manifest.resolve()),
                "dataset_json_sha256": qdp_hash,
            }
        },
        "dependency_padding_complete": True,
        "data_semantics": {
            "candidate_selection_uses_future_labels": False,
            "continuity_break_rule": {
                "minimum_consecutive_suspended_open_days": 20,
                "break_position": "first_non_suspended_open_day_after_qualifying_run",
            },
        },
        "normalization": {
            "fit_scope": "intentionally_stale",
            **{channel: {"mean": [999.0], "std": [999.0]} for channel in feature_channels},
        },
        "artifact_view": {"schema_version": 1, "view_id": "tiny_source"},
    }
    source_path = store_root / "views" / "seq100_path60_todayclose_ohlcva.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(json.dumps(source_view), encoding="utf-8")
    return source_path


def test_development_fold_preserves_source_research_contract(tmp_path: Path) -> None:
    store_root = tmp_path / "research_store"
    source_view = _build_source_view(store_root)
    source = json.loads(source_view.read_text(encoding="utf-8"))
    binding = {
        "path": str(tmp_path / "study.json"),
        "contract_id": "current-study",
        "contract_sha256": "a" * 64,
        "contract_file_sha256": "a" * 64,
    }
    source["research_contract"] = binding
    source["development_contract"] = binding
    source_view.write_text(json.dumps(source), encoding="utf-8")

    result = build_development_walkforward_fold(
        source_view=source_view,
        development_year=2022,
        train_start_year=2021,
        store_root=store_root,
    )
    fold = json.loads(Path(result["view_path"]).read_text(encoding="utf-8"))

    assert fold["research_contract"] == binding
    assert fold["development_contract"] == binding


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")


def _build_completed_tiny_study(tmp_path: Path) -> tuple[Path, Path, Path]:
    store_root = tmp_path / "custom_research_store"
    source_view = _build_source_view(store_root)
    build_purged_walkforward_fold(
        source_view=source_view,
        oos_year=2022,
        train_start_year=2021,
        store_root=store_root,
    )
    view_path = store_root / "views" / "seq100_path60_todayclose_ohlcva_purged_oos2022.json"
    view = json.loads(view_path.read_text(encoding="utf-8"))
    fold_contract = view["fold_training_contract"]
    study_root = tmp_path / "study"
    entries: dict[str, Any] = {}
    dates = ["2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06"]
    value_column = "path_trade_value_v2_2d"
    for profile_index, (profile, profile_command) in enumerate(PROFILE_COMMANDS.items()):
        run_tag = f"{profile}_purged_oos2022_seed7"
        run_dir = study_root / "runs" / f"{run_tag}_tiny"
        run_dir.mkdir(parents=True, exist_ok=True)
        resolved = _resolved_profile_config(profile, epochs=1, seed=7, device="cpu")
        alpha = 0.2 if profile_index == 0 else 0.1
        split_metrics = pd.DataFrame(
            [{"split": "oos", "row_count": 8, "date_count": 4, "rank_ic_mean": 0.05 - profile_index * 0.01}]
        )
        topk_metrics = pd.DataFrame(
            [{"split": "oos", "top_k": 3, f"alpha_{value_column}": alpha}]
        )
        daily_topk = pd.DataFrame(
            [
                {
                    "split": "oos",
                    "trade_date": trade_date,
                    "top_k": 3,
                    "score_column": "score",
                    "universe_count": 2,
                    "universe_hash": f"universe-{trade_date}",
                    f"alpha_{value_column}": alpha,
                    "selected_hit_10pct_rate": 0.6 + alpha,
                    "selected_loss_5pct_rate": 0.4 - alpha,
                }
                for trade_date in dates
            ]
        )
        daily_ic = pd.DataFrame(
            [
                {"split": "oos", "trade_date": trade_date, "rank_ic": 0.05 - profile_index * 0.01, "count": 2}
                for trade_date in dates
            ]
        )
        split_path = run_dir / "split_metrics.csv"
        topk_path = run_dir / "topk_metrics.csv"
        daily_topk_path = run_dir / "daily_topk_metrics.csv"
        daily_ic_path = run_dir / "daily_rank_ic.csv"
        history_path = run_dir / "training_history.csv"
        candidates_path = run_dir / "topk_candidates.parquet"
        checkpoint_path = run_dir / "final_model.pt"
        split_metrics.to_csv(split_path, index=False)
        topk_metrics.to_csv(topk_path, index=False)
        daily_topk.to_csv(daily_topk_path, index=False)
        daily_ic.to_csv(daily_ic_path, index=False)
        pd.DataFrame([{"epoch": 1}]).to_csv(history_path, index=False)
        candidates_path.write_bytes(b"tiny")
        torch.save(
            {
                "model_state_dict": {},
                "config": {
                    **resolved,
                    "pack_manifest": view_path,
                    "output_root": study_root / "runs",
                    "run_tag": run_tag,
                },
                "input_dim": 1,
                "summary_columns": [],
                "feature_channels": fold_contract["payload"]["feature_channels"],
                "best_epoch": 1,
                "checkpoint_policy": "final_epoch",
            },
            checkpoint_path,
        )
        input_channels = (
            ["daily_raw", "daily_state", "intraday_summary", "limit_structure"]
            if resolved["input_channel_profile"] == "all"
            else ["daily_raw", "daily_state"]
        )
        summary = {
            "artifact_type": "qdp_v2_sequence_path_training",
            "run_tag": run_tag,
            "seed": 7,
            "top_k": resolved["top_k"],
            "pack_manifest": str(view_path.resolve()),
            "output_dir": str(run_dir.resolve()),
            "lookback_days": 1,
            "forward_days": 2,
            "value_column": value_column,
            "prediction_mode": "none",
            "evaluation_mode": "fixed_oos",
            "evaluation_splits": ["oos"],
            "evaluation_split": "oos",
            "checkpoint_policy": "final_epoch",
            "fold_year": 2022,
            "train_label_end_before": view["purged_walkforward"]["oos_start"],
            "normalization_cutoff": view["purged_walkforward"]["normalization_cutoff_exclusive"],
            "epochs": 1,
            "completed_epochs": 1,
            "best_epoch": 1,
            "batch_size": resolved["batch_size"],
            "max_samples_per_split": 0,
            "input_channel_profile": resolved["input_channel_profile"],
            "input_channels": input_channels,
            "direct_value_horizon": 0,
            "device": "cpu",
            "amp_enabled": False,
            "model": {
                "type": f"SequencePathModel_{resolved['model_type']}",
                "input_dim": len(input_channels),
                "hidden_dim": resolved["hidden_dim"],
                "layers": resolved["layers"],
                "dropout": resolved["dropout"],
            },
            "loss_weights": {
                "path": resolved["path_loss_weight"],
                "path_profile": resolved["path_loss_profile"],
                "summary": resolved["summary_loss_weight"],
                "richer": resolved["richer_loss_weight"],
                "price_delta": resolved["price_delta_loss_weight"],
                "value": resolved["value_loss_weight"],
                "rank": resolved["rank_loss_weight"],
                "rank_max_per_side": resolved["rank_max_per_side"],
                "summary_profile": resolved["summary_loss_profile"],
                "va_level": resolved["va_level_loss_weight"],
                "va_delta": resolved["va_delta_loss_weight"],
            },
            "early_stopping": {"patience": 0, "min_delta": 0.0},
            "resolved_training_config": resolved,
            "fold_training_contract": fold_contract,
            "best_checkpoint": str(checkpoint_path.resolve()),
            "outputs": {
                "split_metrics_csv": str(split_path.resolve()),
                "topk_metrics_csv": str(topk_path.resolve()),
                "daily_rank_ic_csv": str(daily_ic_path.resolve()),
                "daily_topk_metrics_csv": str(daily_topk_path.resolve()),
                "topk_candidates_parquet": str(candidates_path.resolve()),
                "training_history_csv": str(history_path.resolve()),
            },
        }
        summary_path = run_dir / "sequence_path_training_summary.json"
        _write_json(summary_path, summary)
        command = [
            "python",
            "-m",
            "daily_research.path_policy.seq100_mainline",
            profile_command,
            "--store-view",
            str(view_path.resolve()),
            "--output-root",
            str((study_root / "runs").resolve()),
            "--run-tag",
            run_tag,
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--seed",
            "7",
            "--early-stopping-patience",
            "0",
            "--prediction-mode",
            "none",
            "--evaluation-mode",
            "fixed_oos",
            "--json",
        ]
        entries[f"{profile}:oos2022:seed7"] = {
            "status": "completed",
            "profile": profile,
            "profile_command": profile_command,
            "oos_year": 2022,
            "seed": 7,
            "view_path": str(view_path.resolve()),
            "run_tag": run_tag,
            "command": command,
            "run_dir": str(run_dir.resolve()),
            "summary_path": str(summary_path.resolve()),
            "fold_training_contract": fold_contract,
            "resolved_profile_config": resolved,
        }
    study_manifest = {
        "schema_version": 2,
        "artifact_type": "seq100_purged_walkforward_study_manifest",
        "source_view": str(source_view.resolve()),
        "oos_years": [2022],
        "profiles": list(PROFILE_COMMANDS),
        "seed": 7,
        "epochs": 1,
        "train_start_year": 2021,
        "store_root": str(store_root.resolve()),
        "device": "cpu",
        "run_tag": STUDY_RUN_TAG,
        "evaluation_mode": "fixed_oos",
        "early_stopping_patience": 0,
        "prediction_mode": "none",
        "entries": entries,
    }
    _write_json(study_root / "study_manifest.json", study_manifest)
    return study_root, store_root, view_path


def _strip_tiny_study_to_legacy(study_root: Path, view_path: Path) -> None:
    view = json.loads(view_path.read_text(encoding="utf-8"))
    view.pop("fold_training_contract")
    _write_json(view_path, view)
    manifest_path = study_root / "study_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("store_root", "train_start_year", "device", "run_tag"):
        manifest.pop(key)
    for entry in manifest["entries"].values():
        entry.pop("fold_training_contract")
        entry.pop("resolved_profile_config")
        summary_path = Path(entry["summary_path"])
        run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_summary.pop("fold_training_contract")
        run_summary.pop("resolved_training_config")
        _write_json(summary_path, run_summary)
    _write_json(manifest_path, manifest)


def test_build_purged_fold_enforces_label_end_before_oos_and_refits_normalization(tmp_path: Path) -> None:
    store_root = tmp_path / "research_store"
    source_view = _build_source_view(store_root)

    build_purged_walkforward_fold(
        source_view=source_view,
        oos_year=2022,
        train_start_year=2021,
        store_root=store_root,
    )

    view_path = store_root / "views" / "seq100_path60_todayclose_ohlcva_purged_oos2022.json"
    view = json.loads(view_path.read_text(encoding="utf-8"))
    sample_index = pd.read_parquet(view["sample_index_path"])
    train = sample_index.loc[sample_index["split"].eq("train")]
    oos = sample_index.loc[sample_index["split"].eq("oos")]

    assert set(sample_index["split"]) == {"train", "oos"}
    assert view["train_years"] == [2021]
    assert view["validation_years"] == []
    assert view["test_years"] == []
    assert view["oos_years"] == [2022]
    assert view["split_roles"] == {"fit": "train", "evaluation": "oos"}
    assert view["end_date"] == "2022-01-05"
    assert len(train) == 6
    assert len(oos) == 6
    assert set(pd.to_datetime(oos["trade_date"]).dt.year) == {2022}
    oos_start_idx = int(oos["date_idx"].min())
    assert int((train["date_idx"].astype(int) + 2).max()) < oos_start_idx
    # Exact boundary: idx2 ends on idx4 and remains; idx3 ends at OOS idx5 and is purged.
    assert set(train["date_idx"].astype(int)) == {0, 1, 2}

    assert view["artifact_view"]["view_type"] == "purged_walk_forward_research_store_view"
    audit = view["artifact_view"]["purge_audit"]
    assert audit["oos_year"] == 2022
    assert audit["oos_start_trade_date"] == "2022-01-03"
    assert audit["safe_train_signal_end_trade_date"] == "2021-12-29"
    assert audit["max_train_label_end_trade_date"] == "2021-12-31"
    assert audit["purged_row_count"] == 4
    assert audit["purged_signal_date_count"] == 2
    assert audit["label_overlap_count"] == 0

    assert view["normalization"]["fit_scope"] == "feature_dates_before_oos_start"
    assert view["normalization"]["fit_date_end"] == "2021-12-31"
    for channel in ("daily_raw", "daily_state", "intraday_summary", "limit_structure"):
        assert np.isclose(view["normalization"][channel]["mean"][0], 3.0)
        assert np.isclose(view["normalization"][channel]["std"][0], np.sqrt(2.0))

    verification = verify_purged_walkforward_view(view_path)
    assert verification["status"] == "ok"
    assert verification["label_overlap_count"] == 0


def test_purged_fold_rejects_changed_source_manifest(tmp_path: Path) -> None:
    store_root = tmp_path / "research_store"
    source_view = _build_source_view(store_root)
    result = build_purged_walkforward_fold(
        source_view=source_view,
        oos_year=2022,
        train_start_year=2021,
        store_root=store_root,
    )
    view_path = Path(result["view_path"])
    source = json.loads(source_view.read_text(encoding="utf-8"))
    source["tampered_after_fold_build"] = True
    source_view.write_text(json.dumps(source), encoding="utf-8")

    verification = verify_purged_walkforward_view(view_path)

    assert verification["status"] == "blocked"
    assert any("source_view_provenance" in item for item in verification["blockers"])


def test_purged_fold_rejects_sample_date_index_mapping_mismatch(tmp_path: Path) -> None:
    store_root = tmp_path / "research_store"
    source_view = _build_source_view(store_root)
    source = json.loads(source_view.read_text(encoding="utf-8"))
    sample_path = Path(source["sample_index_path"])
    sample = pd.read_parquet(sample_path)
    sample.loc[0, "trade_date"] = DATES[1]
    sample.to_parquet(sample_path, index=False)

    with pytest.raises(ValueError, match=r"trade_date.*date_values\[date_idx\]"):
        build_purged_walkforward_fold(
            source_view=source_view,
            oos_year=2022,
            train_start_year=2021,
            store_root=store_root,
        )


def test_purged_fold_rejects_changed_source_backing_file(tmp_path: Path) -> None:
    store_root = tmp_path / "research_store"
    source_view = _build_source_view(store_root)
    result = build_purged_walkforward_fold(
        source_view=source_view,
        oos_year=2022,
        train_start_year=2021,
        store_root=store_root,
    )
    source = json.loads(source_view.read_text(encoding="utf-8"))
    backing_path = Path(source["feature_channels"]["daily_raw"]["path"])
    payload = bytearray(backing_path.read_bytes())
    payload[0] ^= 1
    backing_path.write_bytes(payload)

    verification = verify_purged_walkforward_view(Path(result["view_path"]))

    assert verification["status"] == "blocked"
    assert any("backing-file identity changed" in item for item in verification["blockers"])


def test_hac_and_moving_block_intervals_are_finite_and_reproducible() -> None:
    values = np.sin(np.arange(64, dtype=np.float64) / 5.0) + np.arange(64, dtype=np.float64) / 100.0

    hac_first = hac_mean_interval(values, max_lag=7, confidence=0.90)
    hac_second = hac_mean_interval(values, max_lag=7, confidence=0.90)
    assert hac_first == hac_second
    assert hac_first["n"] == 64
    assert np.isclose(hac_first["mean"], values.mean())
    assert hac_first["se"] >= 0.0
    assert hac_first["ci_low"] <= hac_first["mean"] <= hac_first["ci_high"]

    bootstrap_first = moving_block_bootstrap_mean(
        values,
        block_length=8,
        replications=500,
        seed=7,
        confidence=0.90,
    )
    bootstrap_second = moving_block_bootstrap_mean(
        values,
        block_length=8,
        replications=500,
        seed=7,
        confidence=0.90,
    )
    assert bootstrap_first == bootstrap_second
    assert bootstrap_first["n"] == 64
    assert np.isclose(bootstrap_first["mean"], values.mean())
    assert bootstrap_first["bootstrap_std"] > 0.0
    assert bootstrap_first["ci_low"] < bootstrap_first["ci_high"]


def test_paired_topk_comparison_is_strict_and_recovers_constant_difference() -> None:
    rows: list[dict[str, Any]] = []
    for profile, offset in (("summary_v2_all_channels", 0.10), ("daily_only_summary_v2_ohlcva_aux_low", 0.0)):
        for year in (2022, 2023):
            for day in range(4):
                rows.append(
                    {
                        "profile": profile,
                        "oos_year": year,
                        "seed": 7,
                        "trade_date": f"{year}-01-{day + 3:02d}",
                        "split": "oos",
                        "top_k": 3,
                        "score_column": "score",
                        "universe_count": 10,
                        "universe_hash": f"hash-{year}-{day}",
                        "alpha_path_trade_value_v2_60d": 0.20 + offset,
                        "selected_hit_10pct_rate": 0.60 + offset,
                        "selected_loss_5pct_rate": 0.40 - offset,
                    }
                )
    frame = pd.DataFrame(rows)

    comparison = paired_topk_comparison(frame, block_length=2, replications=100, seed=7)
    primary = comparison.loc[comparison["metric"].eq("alpha_path_trade_value_v2_60d")].iloc[0]

    assert np.isclose(primary["pooled_mean_difference"], 0.10)
    assert np.isclose(primary["fold_equal_mean_difference"], 0.10)
    assert primary["positive_fold_count"] == 2
    assert np.isclose(primary["pooled_mbb_ci_low"], 0.10)
    assert np.isclose(primary["pooled_mbb_ci_high"], 0.10)
    assert primary["pooled_mbb_method"] == "continuous_ordered_moving_block_bootstrap"

    mismatched = frame.copy()
    mask = mismatched["profile"].eq("daily_only_summary_v2_ohlcva_aux_low")
    mismatched.loc[mask.idxmax(), "universe_hash"] = "different"
    with pytest.raises(ValueError, match="different daily universes"):
        paired_topk_comparison(mismatched, block_length=2, replications=10, seed=7)


def test_paired_comparisons_use_continuous_mbb_and_reject_non_finite_values() -> None:
    differences = np.asarray([0.4, 0.3, 0.2, -0.1, -0.2, -0.3], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    ic_rows: list[dict[str, Any]] = []
    for profile, multiplier in (("summary_v2_all_channels", 1.0), ("daily_only_summary_v2_ohlcva_aux_low", 0.0)):
        for index, difference in enumerate(differences):
            year = 2022 if index < 3 else 2023
            trade_date = f"{year}-01-{index % 3 + 3:02d}"
            rows.append(
                {
                    "profile": profile,
                    "oos_year": year,
                    "seed": 7,
                    "trade_date": trade_date,
                    "split": "oos",
                    "top_k": 3,
                    "score_column": "score",
                    "universe_count": 10,
                    "universe_hash": f"hash-{year}-{index}",
                    "alpha_path_trade_value_v2_60d": multiplier * difference,
                    "selected_hit_10pct_rate": 0.5 + multiplier * difference,
                    "selected_loss_5pct_rate": 0.5 - multiplier * difference,
                }
            )
            ic_rows.append(
                {
                    "profile": profile,
                    "oos_year": year,
                    "seed": 7,
                    "trade_date": trade_date,
                    "split": "oos",
                    "count": 10,
                    "rank_ic": multiplier * difference,
                }
            )
    comparison = paired_topk_comparison(pd.DataFrame(rows), block_length=2, replications=250, seed=7)
    primary = comparison.loc[comparison["metric"].eq("alpha_path_trade_value_v2_60d")].iloc[0]
    expected = moving_block_bootstrap_mean(differences, block_length=2, replications=250, seed=7 + 3 * 1009)
    assert np.isclose(primary["pooled_mbb_ci_low"], expected["ci_low"])
    assert np.isclose(primary["pooled_mbb_ci_high"], expected["ci_high"])
    assert "grouped_pooled_mbb_ci_low" in comparison.columns

    ic = paired_ic_comparison(pd.DataFrame(ic_rows), block_length=2, replications=250, seed=7)
    expected_ic = moving_block_bootstrap_mean(differences, block_length=2, replications=250, seed=7)
    assert ic["moving_block_bootstrap"] == expected_ic
    assert ic["moving_block_bootstrap_method"] == "continuous_ordered_moving_block_bootstrap"

    bad_topk = pd.DataFrame(rows)
    bad_topk.loc[0, "alpha_path_trade_value_v2_60d"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        paired_topk_comparison(bad_topk, block_length=2, replications=10, seed=7)
    bad_ic = pd.DataFrame(ic_rows)
    bad_ic.loc[0, "rank_ic"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        paired_ic_comparison(bad_ic, block_length=2, replications=10, seed=7)


def test_run_study_rejects_partial_profiles_and_unsafe_fold_overwrite(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported profiles"):
        run_walkforward_study(
            profiles=("not_a_profile",),
            study_root=tmp_path / "partial",
        )
    with pytest.raises(ValueError, match="overwrite-folds requires --rerun-completed"):
        run_walkforward_study(
            profiles=tuple(PROFILE_COMMANDS),
            study_root=tmp_path / "overwrite",
            overwrite_folds=True,
            rerun_completed=False,
        )


def test_summary_uses_manifest_store_root_and_rejects_stale_fold_hash(tmp_path: Path) -> None:
    study_root, store_root, view_path = _build_completed_tiny_study(tmp_path)
    summary = summarize_walkforward_study(
        study_root=study_root,
        bootstrap_replications=50,
        block_length=2,
        bootstrap_seed=7,
    )
    assert summary["status"] == "completed"
    assert summary["run_tag"] == STUDY_RUN_TAG
    assert summary["store_root"] == str(store_root.resolve())
    assert summary["primary_result"]["pooled_mbb_method"] == "continuous_ordered_moving_block_bootstrap"
    assert summary["evidence_verdict"]["promotion_allowed"] is False
    assert summary["evidence_verdict"]["active_execution_artifact_changed"] is False

    view = json.loads(view_path.read_text(encoding="utf-8"))
    view["fold_training_contract"]["sha256"] = "0" * 64
    _write_json(view_path, view)
    with pytest.raises(ValueError, match="fold_training_contract"):
        summarize_walkforward_study(
            study_root=study_root,
            bootstrap_replications=10,
            block_length=2,
            bootstrap_seed=7,
        )


def test_explicit_legacy_binding_validates_then_records_reconstruction(tmp_path: Path) -> None:
    study_root, store_root, view_path = _build_completed_tiny_study(tmp_path)
    _strip_tiny_study_to_legacy(study_root, view_path)
    manifest_path = study_root / "study_manifest.json"

    result = bind_legacy_walkforward_run_contracts(
        study_root=study_root,
        store_root=store_root,
        train_start_year=2021,
        device="cpu",
    )
    assert result["status"] == "completed"
    assert result["bound_fold_count"] == 1
    assert result["bound_run_count"] == len(PROFILE_COMMANDS)
    rebound_view = json.loads(view_path.read_text(encoding="utf-8"))
    binding = rebound_view["fold_training_contract_binding"]
    assert binding["method"] == "post_run_reconstruction_v1"
    assert len(binding["pre_correction_full_view_sha256"]) == 64
    assert len(binding["sample_index_sha256"]) == 64
    rebound_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert rebound_manifest["store_root"] == str(store_root.resolve())
    for entry in rebound_manifest["entries"].values():
        assert entry["fold_training_contract"]["sha256"] == result["fold_training_contract_sha256"]["2022"]
        run_summary = json.loads(Path(entry["summary_path"]).read_text(encoding="utf-8"))
        assert run_summary["fold_training_contract_binding"]["resolved_training_config_source"] == "final_checkpoint_config"
        assert run_summary["resolved_training_config"] == entry["resolved_profile_config"]

    manifest_before_recheck = manifest_path.read_bytes()
    second = bind_legacy_walkforward_run_contracts(
        study_root=study_root,
        store_root=store_root,
        train_start_year=2021,
        device="cpu",
    )
    assert second["changed_file_count"] == 0
    assert manifest_path.read_bytes() == manifest_before_recheck

    summary = summarize_walkforward_study(
        study_root=study_root,
        bootstrap_replications=20,
        block_length=2,
        bootstrap_seed=7,
    )
    assert summary["status"] == "completed"


def test_legacy_binding_rejects_checkpoint_config_mismatch_before_writes(tmp_path: Path) -> None:
    study_root, store_root, view_path = _build_completed_tiny_study(tmp_path)
    _strip_tiny_study_to_legacy(study_root, view_path)
    manifest_path = study_root / "study_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_entry = next(iter(manifest["entries"].values()))
    summary = json.loads(Path(first_entry["summary_path"]).read_text(encoding="utf-8"))
    checkpoint_path = Path(summary["best_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["config"]["learning_rate"] = 123.0
    torch.save(checkpoint, checkpoint_path)

    view_before = view_path.read_bytes()
    manifest_before = manifest_path.read_bytes()
    summaries_before = {
        entry["summary_path"]: Path(entry["summary_path"]).read_bytes()
        for entry in manifest["entries"].values()
    }
    with pytest.raises(ValueError, match="final checkpoint config"):
        bind_legacy_walkforward_run_contracts(
            study_root=study_root,
            store_root=store_root,
            train_start_year=2021,
            device="cpu",
        )
    assert view_path.read_bytes() == view_before
    assert manifest_path.read_bytes() == manifest_before
    assert {
        entry["summary_path"]: Path(entry["summary_path"]).read_bytes()
        for entry in manifest["entries"].values()
    } == summaries_before


def test_legacy_binding_resumes_after_partial_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    study_root, store_root, view_path = _build_completed_tiny_study(tmp_path)
    _strip_tiny_study_to_legacy(study_root, view_path)
    original_write_json = walkforward._write_json
    write_count = 0

    def _interrupt_third_write(path: str | Path, payload: dict[str, Any]) -> Path:
        nonlocal write_count
        write_count += 1
        if write_count == 3:
            raise RuntimeError("injected migration interruption")
        return original_write_json(path, payload)

    monkeypatch.setattr(walkforward, "_write_json", _interrupt_third_write)
    with pytest.raises(RuntimeError, match="injected migration interruption"):
        bind_legacy_walkforward_run_contracts(
            study_root=study_root,
            store_root=store_root,
            train_start_year=2021,
            device="cpu",
        )
    partial_view = json.loads(view_path.read_text(encoding="utf-8"))
    assert partial_view["fold_training_contract_binding"]["method"] == "post_run_reconstruction_v1"
    partial_manifest = json.loads((study_root / "study_manifest.json").read_text(encoding="utf-8"))
    partial_summaries = [
        json.loads(Path(entry["summary_path"]).read_text(encoding="utf-8"))
        for entry in partial_manifest["entries"].values()
    ]
    assert sum("resolved_training_config" in summary for summary in partial_summaries) == 1

    monkeypatch.setattr(walkforward, "_write_json", original_write_json)
    resumed = bind_legacy_walkforward_run_contracts(
        study_root=study_root,
        store_root=store_root,
        train_start_year=2021,
        device="cpu",
    )
    assert resumed["status"] == "completed"
    assert resumed["changed_file_count"] == len(PROFILE_COMMANDS)
    completed_manifest = json.loads((study_root / "study_manifest.json").read_text(encoding="utf-8"))
    for entry in completed_manifest["entries"].values():
        completed_summary = json.loads(Path(entry["summary_path"]).read_text(encoding="utf-8"))
        assert completed_summary["resolved_training_config"] == entry["resolved_profile_config"]


def test_bound_fold_metadata_reconciliation_never_rewrites_live_sample(tmp_path: Path) -> None:
    study_root, store_root, view_path = _build_completed_tiny_study(tmp_path)
    _strip_tiny_study_to_legacy(study_root, view_path)
    bind_legacy_walkforward_run_contracts(
        study_root=study_root,
        store_root=store_root,
        train_start_year=2021,
        device="cpu",
    )
    bound_view = json.loads(view_path.read_text(encoding="utf-8"))
    sample_path = Path(bound_view["sample_index_path"])
    sample_before = sample_path.read_bytes()
    contract_before = bound_view["fold_training_contract"]
    binding_before = bound_view["fold_training_contract_binding"]

    result = reconcile_bound_walkforward_fold_metadata(
        study_root=study_root,
        staging_root=tmp_path / "staging_store",
    )
    assert result["status"] == "completed"
    assert result["changed_view_count"] == 1
    reconciled_view = json.loads(view_path.read_text(encoding="utf-8"))
    assert Path(reconciled_view["sample_index_path"]) == sample_path
    assert sample_path.read_bytes() == sample_before
    assert reconciled_view["fold_training_contract"] == contract_before
    assert reconciled_view["fold_training_contract_binding"] == binding_before
    assert reconciled_view["metadata_reconciliation"]["method"] == "staged_metadata_only_v1"
    assert verify_purged_walkforward_view(view_path)["status"] == "ok"
    study_manifest = json.loads((study_root / "study_manifest.json").read_text(encoding="utf-8"))
    assert study_manifest["fold_metadata_reconciliation"]["status"] == "completed"

    second = reconcile_bound_walkforward_fold_metadata(
        study_root=study_root,
        staging_root=tmp_path / "staging_store",
    )
    assert second["changed_view_count"] == 0
    assert second["idempotent_recheck"] is True
    assert sample_path.read_bytes() == sample_before


def test_bound_fold_metadata_reconciliation_mismatch_leaves_live_artifacts_unchanged(tmp_path: Path) -> None:
    study_root, store_root, view_path = _build_completed_tiny_study(tmp_path)
    _strip_tiny_study_to_legacy(study_root, view_path)
    bind_legacy_walkforward_run_contracts(
        study_root=study_root,
        store_root=store_root,
        train_start_year=2021,
        device="cpu",
    )
    manifest_path = study_root / "study_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_view = json.loads(Path(manifest["source_view"]).read_text(encoding="utf-8"))
    source_index_path = Path(source_view["sample_index_path"])
    source_index = pd.read_parquet(source_index_path)
    source_index.loc[0, "split"] = "changed-after-training"
    source_index.to_parquet(source_index_path, index=False)

    bound_view = json.loads(view_path.read_text(encoding="utf-8"))
    live_sample_path = Path(bound_view["sample_index_path"])
    view_before = view_path.read_bytes()
    sample_before = live_sample_path.read_bytes()
    manifest_before = manifest_path.read_bytes()
    with pytest.raises(ValueError, match="sample frame differs"):
        reconcile_bound_walkforward_fold_metadata(
            study_root=study_root,
            staging_root=tmp_path / "mismatch_staging_store",
        )
    assert view_path.read_bytes() == view_before
    assert live_sample_path.read_bytes() == sample_before
    assert manifest_path.read_bytes() == manifest_before
    assert not (study_root / "fold_metadata_reconciliation.json").exists()
