from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from daily_research.baseline.advanced_ml_runtime import HistoryWindow
from daily_research.continuous_policy.state_builder import PreparedPolicyInputs
from daily_research.path_policy import tq_baostock_lineage_audit as audit


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _frames(symbols: list[str], dates: pd.DatetimeIndex, *, open_shift: int = 0, scale: float = 1.0) -> dict[str, pd.DataFrame]:
    base = pd.DataFrame(
        {symbol: [10.0 + idx + pos * 0.1 for idx in range(len(dates))] for pos, symbol in enumerate(symbols)},
        index=dates,
    )
    open_frame = base.shift(open_shift) if open_shift else base.copy()
    open_frame = open_frame.bfill().ffill()
    return {
        "Open": open_frame * scale,
        "High": (base + 0.3) * scale,
        "Low": (base - 0.3) * scale,
        "Close": (base + 0.1) * scale,
        "Volume": pd.DataFrame(1000.0, index=dates, columns=symbols),
        "Amount": pd.DataFrame(10000.0, index=dates, columns=symbols),
    }


def _prepared(frames: dict[str, pd.DataFrame], symbols: list[str], dates: pd.DatetimeIndex) -> PreparedPolicyInputs:
    return PreparedPolicyInputs(
        universe=tuple(symbols),
        pool_name="unit",
        benchmark="000300.SH",
        data_source="unit",
        csv_folder="",
        start_date=dates.min().strftime("%Y-%m-%d"),
        end_date=dates.max().strftime("%Y-%m-%d"),
        requested_start_date=dates.min().strftime("%Y%m%d"),
        history_window=HistoryWindow(mode="train", requested_start_date="", effective_start_date="", end_date="", required_trading_days=len(dates)),
        raw_cache_meta={},
        prepared_cache_meta={},
        close=frames["Close"],
        open_=frames["Open"],
        high=frames["High"],
        low=frames["Low"],
        volume=frames["Volume"],
        amount=frames["Amount"],
        benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
        benchmark_open=pd.Series(4000.0, index=dates, name="000300.SH"),
        score_none=pd.DataFrame(0.0, index=dates, columns=symbols),
        score_v2=pd.DataFrame(0.0, index=dates, columns=symbols),
        score_blend=pd.DataFrame(0.0, index=dates, columns=symbols),
        feature_frames={},
        market_features={},
        membership_frame=pd.DataFrame(True, index=dates, columns=symbols),
        rolling_pool_summary={},
        alpha_prior_summary={},
        derived_frames={},
    )


class FailingAdapter:
    def get_market_data(self, **_: object) -> dict[str, pd.DataFrame]:
        raise RuntimeError("unit unavailable")


def test_recover_legacy156_manifest_reports_feature_diff(tmp_path: Path) -> None:
    current = tmp_path / "current" / "forecast_dataset_manifest.json"
    legacy = tmp_path / "legacy" / "forecast_dataset_manifest.json"
    _write_json(current, {"feature_store_shape": [1699, 500, 116], "feature_columns": [f"f{i}" for i in range(116)], "feature_group_counts": {"state": 1}})
    _write_json(
        legacy,
        {
            "feature_store_shape": [1699, 2430, 156],
            "feature_columns": [f"f{i}" for i in range(100)] + [f"old{i}" for i in range(56)],
            "feature_group_counts": {"state": 2},
        },
    )

    payload = audit.recover_legacy156_features(current_manifest_path=current, search_roots=(tmp_path / "legacy",))

    assert payload["status"] == "legacy156_recovered"
    assert payload["legacy156_profile_allowed"] is True
    assert payload["shared_feature_count"] == 100
    assert payload["old_only_feature_count"] == 56
    assert payload["new_only_feature_count"] == 16


def test_recover_legacy156_ignores_loose_text_without_columns(tmp_path: Path) -> None:
    current = tmp_path / "current" / "forecast_dataset_manifest.json"
    _write_json(current, {"feature_store_shape": [1699, 500, 116], "feature_columns": [f"f{i}" for i in range(116)]})
    loose = tmp_path / "ref.md"
    loose.write_text("Old Stage 2.8 had 156 features.", encoding="utf-8")

    payload = audit.recover_legacy156_features(current_manifest_path=current, search_roots=(tmp_path,))

    assert payload["status"] == "legacy156_not_recovered"
    assert payload["legacy156_profile_allowed"] is False


def test_compare_ohlcv_equivalent_for_identical_frames() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    symbols = ["600000.SH", "000001.SZ"]
    frames = _frames(symbols, dates)

    _, summary = audit.compare_ohlcv_frames(baostock_frames=frames, tq_frames_by_dividend={"none": frames})

    assert summary["status"] == "tq_baostock_ohlcv_equivalent"
    assert "ohlcv_close_enough_for_price_labels" in summary["dividend_summaries"]["none"]["flags"]


def test_compare_ohlcv_open_shift_flags_source_shift() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    symbols = ["600000.SH", "000001.SZ"]
    bao = _frames(symbols, dates)
    tq = _frames(symbols, dates, open_shift=1)

    _, summary = audit.compare_ohlcv_frames(baostock_frames=bao, tq_frames_by_dividend={"none": tq})

    assert summary["status"] == "material_data_source_shift"
    assert "open_source_shift" in summary["dividend_summaries"]["none"]["flags"]


def test_front_closer_than_none_flags_adjustment_mismatch() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    symbols = ["600000.SH", "000001.SZ"]
    bao = _frames(symbols, dates)
    tq_none = _frames(symbols, dates, scale=1.2)
    tq_front = _frames(symbols, dates)

    _, summary = audit.compare_ohlcv_frames(baostock_frames=bao, tq_frames_by_dividend={"none": tq_none, "front": tq_front})

    assert "likely_adjustment_mismatch" in summary["dividend_summaries"]["front"]["flags"]


def test_next_open_label_diff_detects_open_shift() -> None:
    dates = pd.date_range("2024-01-01", periods=45, freq="B")
    symbols = ["600000.SH", "000001.SZ", "000002.SZ"]
    bao = _frames(symbols, dates)
    tq = _frames(symbols, dates, open_shift=1)
    bao["Open"] = pd.DataFrame(10.0, index=dates, columns=symbols)
    tq["Open"] = pd.DataFrame(
        {symbol: [10.0 + idx * (0.05 + pos * 0.01) for idx in range(len(dates))] for pos, symbol in enumerate(symbols)},
        index=dates,
    )
    prepared = _prepared(bao, symbols, dates)

    _, summary = audit.compare_next_open_labels(
        prepared=prepared,
        baostock_frames=bao,
        tq_frames=tq,
        start_date="2024-01-01",
        end_date="2024-02-29",
        horizons=(1, 2, 3, 5, 8, 10, 15, 20, 30),
    )

    assert summary["status"] in {"label_shift_minor_but_relevant", "material_label_shift"}
    assert summary["max_hit_label_flip_rate"] > 0.0


def test_run_audit_records_tq_blocker_without_active_artifact(tmp_path: Path, monkeypatch) -> None:
    dates = pd.date_range("2024-01-01", periods=35, freq="B")
    symbols = ["600000.SH", "000001.SZ"]
    frames = _frames(symbols, dates)
    prepared = _prepared(frames, symbols, dates)
    active = tmp_path / "active_execution_strategy.json"
    active.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(audit, "ACTIVE_EXECUTION_ARTIFACT", active)
    monkeypatch.setattr(audit, "CURRENT_MANIFEST_PATH", tmp_path / "missing_manifest.json")
    monkeypatch.setattr(audit, "load_policy_inputs_from_lake", lambda **_: prepared)

    payload = audit.run_audit(output_root=tmp_path / "out", adapter=FailingAdapter())

    assert payload["status"] == "completed_with_tq_blocker"
    assert payload["tq_status"] == "blocked_tq_unavailable"
    assert active.read_text(encoding="utf-8") == "{}\n"
    assert (tmp_path / "out" / "legacy156_feature_recovery_report.json").exists()
