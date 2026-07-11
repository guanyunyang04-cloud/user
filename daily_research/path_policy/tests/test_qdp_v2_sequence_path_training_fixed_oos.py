from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy import qdp_v2_sequence_path_training as training


def _write_float32(path: Path, values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float32)
    array.tofile(path)
    return {"path": str(path.resolve()), "shape": list(array.shape)}


def _build_tiny_pack(tmp_path: Path, *, fixed_oos: bool) -> Path:
    date_count = 6
    symbol_count = 4
    lookback_days = 2
    forward_days = 2
    symbols = [f"00000{idx + 1}.SZ" for idx in range(symbol_count)]

    daily_raw = np.linspace(
        0.0,
        1.0,
        num=date_count * symbol_count,
        dtype=np.float32,
    ).reshape(date_count, symbol_count, 1)
    daily_state = (daily_raw * 0.5 + 0.1).astype(np.float32)
    future_path = np.zeros((date_count, symbol_count, forward_days, 4), dtype=np.float32)
    for symbol_idx in range(symbol_count):
        close_1 = 0.01 * (symbol_idx + 1)
        close_2 = 0.02 * (symbol_idx + 1)
        future_path[:, symbol_idx, 0, :] = [0.0, close_1 + 0.01, -0.01, close_1]
        future_path[:, symbol_idx, 1, :] = [close_1, close_2 + 0.01, -0.005, close_2]

    rows: list[dict[str, Any]] = []
    split_dates = (
        [("train", 1, "2021-12-30"), ("oos", 4, "2022-01-04")]
        if fixed_oos
        else [
            ("train", 1, "2021-12-30"),
            ("validation", 2, "2022-01-04"),
            ("test", 3, "2023-01-03"),
        ]
    )
    for split, date_idx, trade_date in split_dates:
        for symbol_idx, symbol in enumerate(symbols):
            rows.append(
                {
                    "split": split,
                    "date_idx": date_idx,
                    "symbol_idx": symbol_idx,
                    "trade_date": trade_date,
                    "symbol": symbol,
                }
            )
    sample_index_path = tmp_path / "sample_index.parquet"
    pd.DataFrame(rows).to_parquet(sample_index_path, index=False)

    manifest = {
        "artifact_type": "tiny_sequence_path_pack",
        "lookback_days": lookback_days,
        "forward_days": forward_days,
        "symbol_values": symbols,
        "sample_index_path": str(sample_index_path.resolve()),
        "feature_channels": {
            "daily_raw": {
                **_write_float32(tmp_path / "daily_raw.float32.dat", daily_raw),
                "columns": ["daily_raw_feature"],
            },
            "daily_state": {
                **_write_float32(tmp_path / "daily_state.float32.dat", daily_state),
                "columns": ["daily_state_feature"],
            },
        },
        "label_arrays": {
            "future_ohlc_path": {
                **_write_float32(tmp_path / "future_ohlc_path.float32.dat", future_path),
                "fields": ["open", "high", "low", "close"],
                "price_anchor": "next_open",
            }
        },
        "normalization": {
            "daily_raw": {"mean": [0.0], "std": [1.0]},
            "daily_state": {"mean": [0.0], "std": [1.0]},
        },
    }
    if fixed_oos:
        manifest["fold_training_contract"] = {
            "schema_version": 1,
            "algorithm": "sha256",
            "sha256": "a" * 64,
            "sample_index_sha256": "b" * 64,
            "normalization_sha256": "c" * 64,
            "payload": {"forward_days": forward_days, "lookback_days": lookback_days},
        }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _config(
    tmp_path: Path,
    manifest_path: Path,
    *,
    evaluation_mode: str,
    early_stopping_patience: int = 0,
) -> training.TrainConfig:
    return training.TrainConfig(
        pack_manifest=manifest_path,
        output_root=tmp_path / "output",
        run_tag=f"tiny_{evaluation_mode}",
        epochs=1,
        batch_size=4,
        model_type="gru_path_value",
        hidden_dim=4,
        layers=1,
        dropout=0.0,
        symbol_embedding_dim=0,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        path_loss_weight=0.45,
        path_loss_profile=training.PATH_LOSS_PROFILE_DEFAULT,
        summary_loss_weight=0.20,
        richer_loss_weight=0.0,
        price_delta_loss_weight=0.0,
        va_level_loss_weight=0.0,
        va_delta_loss_weight=0.0,
        value_loss_weight=0.20,
        rank_loss_weight=0.15,
        residual_score_weight=0.25,
        residual_penalty_weight=0.01,
        summary_loss_profile=training.SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
        input_channel_profile=training.INPUT_CHANNEL_PROFILE_DAILY_ONLY,
        direct_value_horizon=0,
        rank_max_per_side=2,
        device="cpu",
        amp=False,
        seed=7,
        top_k=(1, 3),
        max_samples_per_split=0,
        prediction_mode="none",
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=0.0,
        evaluation_mode=evaluation_mode,
    )


def _install_fake_evaluation(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def fake_predict_split(**kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        split = str(kwargs["split"])
        calls.append(split)
        trade_date = "2022-01-04" if split != "test" else "2023-01-03"
        ic = pd.DataFrame([{"trade_date": trade_date, "rank_ic": 0.1, "count": 4}])
        topk = pd.DataFrame([{"top_k": 1, "day_count": 1}])
        daily_topk = pd.DataFrame([{"trade_date": trade_date, "top_k": 1}])
        candidates = pd.DataFrame(
            [{"trade_date": trade_date, "score_rank": 1, "symbol": "000001.SZ", "score": 0.2}]
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
            "va_level_mae": np.nan,
            "va_delta_mae": np.nan,
            "prediction_csv": "",
            "score_diagnostics": {},
        }
        return ic, topk, daily_topk, candidates, metrics

    def fake_write_report(output_dir: Path, **_: Any) -> str:
        report_path = output_dir / "sequence_path_training_report.md"
        report_path.write_text("tiny report\n", encoding="utf-8")
        return str(report_path.resolve())

    monkeypatch.setattr(training, "_predict_split", fake_predict_split)
    monkeypatch.setattr(training, "_write_report", fake_write_report)
    monkeypatch.setattr(training, "_find_latest_baseline_summary", lambda _: {})
    monkeypatch.setattr(training, "_trim_working_set", lambda: None)
    return calls


def test_fixed_oos_skips_training_evaluation_and_evaluates_oos_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _build_tiny_pack(tmp_path, fixed_oos=True)
    calls = _install_fake_evaluation(monkeypatch)

    config = _config(tmp_path, manifest_path, evaluation_mode="fixed_oos")
    summary = training.train_sequence_path_model(config)

    assert calls == ["oos"]
    assert {row["split"] for row in summary["split_metrics"]} == {"oos"}
    assert "validation_predictions_csv" not in summary["outputs"]
    assert "test_predictions_csv" not in summary["outputs"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert summary["fold_training_contract"] == manifest["fold_training_contract"]
    persisted = json.loads(
        (Path(summary["output_dir"]) / "sequence_path_training_summary.json").read_text(encoding="utf-8")
    )
    assert persisted["fold_training_contract"] == manifest["fold_training_contract"]
    expected_config = {
        key: value
        for key, value in config.__dict__.items()
        if key not in {"pack_manifest", "output_root", "run_tag"}
    }
    expected_config["top_k"] = list(expected_config["top_k"])
    assert summary["resolved_training_config"] == expected_config
    assert persisted["resolved_training_config"] == expected_config


def test_fixed_oos_rejects_early_stopping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _build_tiny_pack(tmp_path, fixed_oos=True)
    calls = _install_fake_evaluation(monkeypatch)

    with pytest.raises(ValueError, match=r"(?i)early[_ -]?stopping"):
        training.train_sequence_path_model(
            _config(
                tmp_path,
                manifest_path,
                evaluation_mode="fixed_oos",
                early_stopping_patience=1,
            )
        )

    assert calls == []


def test_fixed_oos_rejects_training_labels_that_reach_oos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _build_tiny_pack(tmp_path, fixed_oos=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_path = Path(manifest["sample_index_path"])
    sample = pd.read_parquet(sample_path)
    # Equality is leakage: the last train label lands on the first OOS feature date.
    sample.loc[sample["split"].eq("oos"), "date_idx"] = 3
    sample.to_parquet(sample_path, index=False)
    calls = _install_fake_evaluation(monkeypatch)

    with pytest.raises(ValueError, match="label_overlap_count"):
        training.train_sequence_path_model(
            _config(tmp_path, manifest_path, evaluation_mode="fixed_oos")
        )

    assert calls == []


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing_date_idx", "missing required columns"),
        ("nullable_date_idx", "finite integers"),
    ],
)
def test_fixed_oos_rejects_bad_loaded_index_before_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    message: str,
) -> None:
    manifest_path = _build_tiny_pack(tmp_path, fixed_oos=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_path = Path(manifest["sample_index_path"])
    sample = pd.read_parquet(sample_path)
    if failure == "missing_date_idx":
        sample = sample.drop(columns=["date_idx"])
    else:
        sample["date_idx"] = sample["date_idx"].astype("Float64")
        sample.loc[sample["split"].eq("oos"), "date_idx"] = pd.NA
    sample.to_parquet(sample_path, index=False)
    calls = _install_fake_evaluation(monkeypatch)

    with pytest.raises(ValueError, match=message):
        training.train_sequence_path_model(
            _config(tmp_path, manifest_path, evaluation_mode="fixed_oos")
        )

    assert calls == []


@pytest.mark.parametrize(
    ("split", "frame", "message"),
    [
        ("train", pd.DataFrame({"wrong_column": [1]}), "missing required date_idx"),
        ("train", pd.DataFrame({"date_idx": [1.5]}), "finite integers"),
        (
            "oos",
            pd.DataFrame({"date_idx": pd.Series([pd.NA], dtype="Int64")}),
            "finite integers",
        ),
        ("oos", pd.DataFrame({"date_idx": [-1]}), "non-negative"),
        ("oos", pd.DataFrame({"date_idx": [np.iinfo(np.int32).max + 1]}), "int32 range"),
    ],
)
def test_fixed_oos_rejects_invalid_date_index_schema(
    split: str,
    frame: pd.DataFrame,
    message: str,
) -> None:
    datasets = {
        "train": SimpleNamespace(sample_index=pd.DataFrame({"date_idx": [1]}), forward_days=2),
        "oos": SimpleNamespace(sample_index=pd.DataFrame({"date_idx": [4]}), forward_days=2),
    }
    datasets[split] = SimpleNamespace(sample_index=frame, forward_days=2)

    with pytest.raises(ValueError, match=message):
        training._validate_fixed_oos_split_contract(
            manifest={},
            train_ds=datasets["train"],
            oos_ds=datasets["oos"],
        )


def test_standard_mode_keeps_validation_and_test_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _build_tiny_pack(tmp_path, fixed_oos=False)
    calls = _install_fake_evaluation(monkeypatch)

    summary = training.train_sequence_path_model(
        _config(tmp_path, manifest_path, evaluation_mode="standard")
    )

    assert calls.count("validation") >= 1
    assert calls.count("test") == 1
    assert {row["split"] for row in summary["split_metrics"]} == {"validation", "test"}
