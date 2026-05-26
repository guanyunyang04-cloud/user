from __future__ import annotations

import json
import os
import sys
import warnings
import gc
from io import StringIO
from pathlib import Path
from typing import Any
from unittest import mock

import pandas as pd
import pytest


FORMAL_REFRESH_DOMAINS = [
    "market_daily",
    "trading_calendar",
    "universe_snapshot",
    "security_status",
    "limit_status",
    "valuation",
    "industry_concept",
    "money_flow_hotspot",
]

FORMAL_REQUIRED_DOMAINS = [
    "market_daily",
    "trading_calendar",
    "universe_snapshot",
    "security_status",
    "limit_status",
]


def test_trade_plan_summary_handles_missing_latest_artifact(tmp_path: Path) -> None:
    from daily_research.execution import app_service

    missing = tmp_path / "missing" / "latest_trade_plan.txt"
    payload = app_service.latest_trade_plan_summary(latest_txt_path=missing)

    assert payload["exists"] is False
    assert payload["status"] == "missing"
    assert payload["summary"] == {}
    assert payload["actions"] == []
    assert payload["holdings"] == []
    assert payload["watchlist"] == []
    assert payload["txt_preview"] == []


def test_trade_plan_summary_prefers_structured_latest_run(tmp_path: Path) -> None:
    from daily_research.execution import app_service

    output_dir = tmp_path / "execution" / "output"
    run_dir = output_dir / "20260523"
    run_dir.mkdir(parents=True)
    (output_dir / "latest_trade_plan.txt").write_text("plan text\nline2\n", encoding="utf-8")
    (run_dir / "daily_trade_plan.txt").write_text("run text\n", encoding="utf-8")
    (run_dir / "plan_summary.json").write_text(
        json.dumps(
            {
                "signal_date": "2026-05-22",
                "cash_input": 1000.0,
                "model_freshness": {
                    "trained_at": "2026-05-20T18:00:00",
                    "artifact_latest_data_date": "2026-05-20",
                    "trading_day_lag": 2,
                    "latest_completed_trading_date": "2026-05-22",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "actions_today.csv").write_text("stock,action,target_weight\n600000.SH,buy,0.1\n", encoding="utf-8-sig")
    (run_dir / "holdings_snapshot.csv").write_text("stock,current_weight,target_weight\n600000.SH,0,0.1\n", encoding="utf-8-sig")
    (run_dir / "watchlist.csv").write_text("stock,score\n600001.SH,0.7\n", encoding="utf-8-sig")

    payload = app_service.latest_trade_plan_summary(output_dir=output_dir)

    assert payload["status"] == "ok"
    assert payload["summary"]["signal_date"] == "2026-05-22"
    assert payload["actions"][0]["stock"] == "600000.SH"
    assert payload["holdings"][0]["stock"] == "600000.SH"
    assert payload["watchlist"][0]["stock"] == "600001.SH"
    assert payload["model_info"]["trading_day_lag"] == 2
    assert payload["txt_preview"] == ["plan text", "line2"]
    assert payload["artifact_paths"]["run_dir"].endswith("20260523")


def test_trade_plan_summary_normalizes_external_model_info(tmp_path: Path) -> None:
    from daily_research.execution import app_service

    output_dir = tmp_path / "execution" / "output"
    run_dir = output_dir / "20260523"
    run_dir.mkdir(parents=True)
    (output_dir / "latest_trade_plan.txt").write_text("plan text\n", encoding="utf-8")
    (run_dir / "plan_summary.json").write_text(
        json.dumps(
            {
                "signal_date": "2026-05-22",
                "production_model_train_end_date": "2026-04-03",
                "production_model_launch_cutoff_date": "2026-04-21",
                "production_model_status": "fresh",
                "production_model_trading_day_lag": 20,
                "candidate_target_weight_csv": "target.csv",
                "source_signal_date": "2026-05-22",
                "execution_signal_date": "2026-05-22",
                "signal_panel_status": "ok",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payload = app_service.latest_trade_plan_summary(output_dir=output_dir)

    assert payload["model_info"]["train_end_date"] == "2026-04-03"
    assert payload["model_info"]["launch_cutoff_date"] == "2026-04-21"
    assert payload["model_info"]["latest_completed_trading_date"] == "2026-05-22"
    assert payload["model_info"]["trading_day_lag"] == 20
    assert payload["model_info"]["status"] == "fresh"
    assert payload["model_info"]["source_signal_date"] == "2026-05-22"
    assert payload["model_info"]["execution_signal_date"] == "2026-05-22"
    assert payload["model_info"]["signal_panel_status"] == "ok"


def test_run_trade_plan_injects_fresh_signal_panel_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from daily_research.baseline import generate_daily_trade_plan
    from daily_research.execution import run_trade_plan

    exec_dir = tmp_path / "execution"
    exec_dir.mkdir()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(run_trade_plan, "bootstrap_execution_paths", lambda _file: exec_dir)
    monkeypatch.setattr(run_trade_plan, "ensure_default_pool_argument", lambda **_kwargs: None)
    monkeypatch.setattr(run_trade_plan, "ensure_execution_strategy_defaults", lambda: None)
    monkeypatch.setattr(
        run_trade_plan.sys,
        "argv",
        ["run_trade_plan.py", "--legacy-ml", "--model-artifact", str(tmp_path / "model.joblib")],
    )
    monkeypatch.setattr(generate_daily_trade_plan, "main", lambda: captured.setdefault("argv", list(run_trade_plan.sys.argv)))

    run_trade_plan.main()

    assert "--require-fresh-signal-panel" in captured["argv"]


def test_external_target_weight_universe_uses_latest_panel_stocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from daily_research.execution import entrypoint_utils, liquidity_universe

    panel = tmp_path / "target.csv"
    panel.write_text(
        "date,stock,target_weight\n"
        "2026-05-21,000001.SZ,1.0\n"
        "2026-05-22,600864.SH,0.5\n"
        "2026-05-22,002866.SZ,0.5\n",
        encoding="utf-8",
    )
    universe_dir = tmp_path / "universe"
    monkeypatch.setattr(liquidity_universe, "get_universe_dir", lambda: universe_dir)
    monkeypatch.setattr(
        entrypoint_utils.sys,
        "argv",
        ["run_trade_plan.py", "--external-target-weight-csv", str(panel)],
    )

    assert entrypoint_utils.ensure_external_target_weight_universe_argument() is True

    stocks_file = Path(entrypoint_utils.sys.argv[entrypoint_utils.sys.argv.index("--stocks-file") + 1])
    assert stocks_file.read_text(encoding="utf-8").splitlines() == ["002866.SZ", "600864.SH"]


def test_models_payload_includes_core_model_roles() -> None:
    from daily_research.execution import app_service

    payload = app_service.models_summary()
    roles = {item["role"] for item in payload["models"]}

    assert payload["status"] == "ok"
    assert {"live", "production", "path_policy", "legacy"} <= roles
    assert "continuous_policy" not in roles
    assert not any(item["id"] == "continuous-policy-shadow" for item in payload["models"])
    assert any(item["is_live"] for item in payload["models"])


def test_execution_smoke_task_is_registered_as_safe_diagnostic() -> None:
    from daily_research.execution.app_tasks import get_task_spec

    spec = get_task_spec("execution-smoke")

    assert spec.category == "diagnostic"
    assert spec.safety_level == "safe"
    assert spec.timeout_seconds == 60
    assert spec.script_path.name == "run_execution_smoke.py"


def test_active_manifest_summary_keeps_trade_plan_signal_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from daily_research.execution import app_service

    manifest_path = tmp_path / "active_execution_strategy.json"
    manifest_path.write_text(
        json.dumps(
            {
                "trade_plan_target_weight_panel_csv": str(tmp_path / "target.csv"),
                "trade_plan_score_panel_csv": str(tmp_path / "score.csv"),
                "production_root": str(tmp_path / "production"),
                "production_manifest_json": str(tmp_path / "production" / "production_retrain_manifest.json"),
                "production_anchor_source_run_dir": str(tmp_path / "fullfit"),
                "production_anchor_sync_manifest": str(tmp_path / "production" / "production_anchor_sync_manifest.json"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_service, "ACTIVE_MANIFEST_PATH", manifest_path)

    payload = app_service.active_manifest_summary()

    assert payload["trade_plan_target_weight_panel_csv"].endswith("target.csv")
    assert payload["trade_plan_score_panel_csv"].endswith("score.csv")
    assert payload["production_root"].endswith("production")
    assert payload["production_manifest_json"].endswith("production_retrain_manifest.json")
    assert payload["production_anchor_source_run_dir"].endswith("fullfit")
    assert payload["production_anchor_sync_manifest"].endswith("production_anchor_sync_manifest.json")


def test_data_sources_payload_includes_lake_and_platform() -> None:
    from daily_research.execution import app_service

    payload = app_service.data_sources_summary()

    assert payload["status"] == "ok"
    assert "lake_root" in payload
    assert "datasets" in payload
    assert "data_platform" in payload


def test_data_sources_payload_exposes_default_refresh_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    from daily_research.execution import app_service

    monkeypatch.setattr(app_service, "get_latest_completed_trading_date", lambda: "2026-05-22", raising=False)

    payload = app_service.data_sources_summary()

    assert payload["data_platform"]["latest_completed_trading_date"] == "2026-05-22"
    assert payload["data_platform"]["recommended_domains"] == FORMAL_REFRESH_DOMAINS
    assert payload["data_platform"]["default_refresh"] == {
        "as_of_date": "2026-05-22",
        "universe": "all_a",
        "domains": FORMAL_REFRESH_DOMAINS,
        "required_domains": FORMAL_REQUIRED_DOMAINS,
        "provider_plan": "formal_free_v3",
    }
    assert payload["data_platform"]["default_refresh"]["required_domains"] == FORMAL_REQUIRED_DOMAINS
    assert payload["formal_provider_plan"] == "formal_free_v3"
    assert "domain_matrix" in payload
    assert "scheduler_status" not in payload
    assert "last_auto_refresh" not in payload


def test_data_sources_payload_marks_current_dataset_latest(monkeypatch: pytest.MonkeyPatch) -> None:
    from daily_research.execution import app_service

    monkeypatch.setattr(app_service, "get_latest_completed_trading_date", lambda: "2026-05-22", raising=False)
    monkeypatch.setattr(
        app_service,
        "_latest_policy_input_lake_dataset",
        lambda lake_root=None: {
            "dataset_id": "policy_input_bundle__current",
            "start_date": "2024-01-01",
            "end_date": "2026-05-22",
            "created_at": "2026-05-24T00:00:00",
            "status": "stored",
        },
    )
    monkeypatch.setattr(
        app_service,
        "active_manifest_summary",
        lambda: {
            "lake_dataset_id": "policy_input_bundle__current",
            "source_market_dataset_id": "policy_input_bundle__current",
        },
    )
    monkeypatch.setattr(app_service, "_latest_refresh_manifest_path", lambda: "", raising=False)

    payload = app_service.data_sources_summary()

    assert payload["current_dataset_status"] == "latest_complete"
    assert payload["is_current_dataset_latest"] is True
    assert payload["is_current_dataset_complete"] is True
    assert payload["next_refresh_action"] == "skip"
    assert "2026-05-22" in payload["refresh_explanation"]


def test_production_anchor_audit_detects_default_fullfit_mismatch(tmp_path: Path) -> None:
    from daily_research.execution import production_signal

    fullfit = tmp_path / "fullfit"
    production = tmp_path / "production"
    fullfit.mkdir()
    production.mkdir()
    (fullfit / "deep_alpha_model.pt").write_bytes(b"fullfit-model")
    (production / "deep_alpha_model.pt").write_bytes(b"formal-source-model")
    (fullfit / "metrics.json").write_text(
        json.dumps({"train_end": "2026-03-31", "stocks_file": ""}, ensure_ascii=False),
        encoding="utf-8",
    )
    (production / "metrics.json").write_text(
        json.dumps(
            {
                "train_end": "2025-03-17",
                "stocks_file": r"H:\new_tdx64\PYPlugins\user\daily_research\execution\universe\liquid500_latest.txt",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    audit = production_signal.audit_production_anchor(
        production_root=production,
        fullfit_run_dir=fullfit,
        project_root=tmp_path / "daily_research",
    )

    assert audit["status"] == "stale_or_mismatched"
    assert audit["model_hash_match"] is False
    assert audit["metrics_train_end_match"] is False
    assert audit["old_path_count"] == 1


def test_data_sources_payload_marks_signal_panels_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    from daily_research.execution import app_service

    monkeypatch.setattr(app_service, "get_latest_completed_trading_date", lambda: "2026-05-22", raising=False)
    monkeypatch.setattr(
        app_service,
        "_latest_policy_input_lake_dataset",
        lambda lake_root=None: {
            "dataset_id": "policy_input_bundle__current",
            "start_date": "2024-01-01",
            "end_date": "2026-05-22",
            "created_at": "2026-05-24T00:00:00",
            "status": "stored",
        },
    )
    monkeypatch.setattr(
        app_service,
        "active_manifest_summary",
        lambda: {
            "lake_dataset_id": "policy_input_bundle__current",
            "source_market_dataset_id": "policy_input_bundle__current",
            "trade_plan_target_weight_panel_csv": "target.csv",
            "trade_plan_score_panel_csv": "score.csv",
        },
    )
    monkeypatch.setattr(app_service, "_latest_refresh_manifest_path", lambda: "", raising=False)
    monkeypatch.setattr(
        app_service.production_signal,
        "signal_panel_summary",
        lambda active, latest_completed_date: {
            "status": "stale",
            "target_panel_latest_date": "2026-05-19",
            "score_panel_latest_date": "2026-05-19",
            "latest_date": "2026-05-19",
            "required_date": "2026-05-22",
            "next_signal_action": "refresh",
        },
    )
    monkeypatch.setattr(
        app_service.production_signal,
        "audit_production_anchor",
        lambda **kwargs: {"status": "ok", "model_hash_match": True},
    )

    payload = app_service.data_sources_summary()

    assert payload["next_refresh_action"] == "skip"
    assert payload["next_signal_action"] == "refresh"
    assert payload["signal_panel_status"] == "stale"
    assert payload["signal_panel_latest_date"] == "2026-05-19"
    assert payload["production_anchor_status"] == "ok"


def test_data_refresh_api_runs_signal_refresh_when_data_latest_but_signal_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    captured: dict[str, Any] = {}

    def fake_launch_task_async(**kwargs):
        captured.update(kwargs)
        return {"job_id": "signal-job", "task_name": kwargs["task_name"], "status": "queued"}

    monkeypatch.setattr(
        web_server.app_service,
        "data_sources_summary",
        lambda: {
            "status": "ok",
            "active_dataset_id": "policy_input_bundle__current",
            "latest_policy_input_dataset_id": "policy_input_bundle__current",
            "latest_policy_input_dataset_end_date": "2026-05-22",
            "dataset_sync_status": "synced",
            "is_current_dataset_latest": True,
            "is_current_dataset_complete": True,
            "next_refresh_action": "skip",
            "next_signal_action": "refresh",
            "signal_panel_status": "stale",
            "refresh_explanation": "数据已最新，但生产信号面板需要刷新。",
            "data_platform": {
                "latest_completed_trading_date": "2026-05-22",
                "recommended_domains": FORMAL_REFRESH_DOMAINS,
                "default_refresh": {
                    "as_of_date": "2026-05-22",
                    "universe": "all_a",
                    "domains": FORMAL_REFRESH_DOMAINS,
                    "provider_plan": "formal_free_v3",
                },
            },
        },
    )
    monkeypatch.setattr(web_server.app_service, "launch_task_async", fake_launch_task_async)

    client = TestClient(web_server.create_app())
    response = client.post("/api/data-sources/refresh", json={})

    assert response.status_code == 200
    assert response.json()["task_name"] == "refresh-production-live-panels"
    assert captured["task_name"] == "refresh-production-live-panels"


def test_signal_refresh_clones_latest_sector_board_view_for_current_dataset(tmp_path: Path) -> None:
    from daily_research.data_lake import ResearchDataLake, load_policy_inputs_from_lake
    from daily_research.execution import production_signal

    dates = pd.to_datetime(["2026-05-20", "2026-05-21", "2026-05-22"])
    stocks = ["000001.SZ", "600000.SH"]
    close = pd.DataFrame([[10.0, 20.0], [10.1, 20.1], [10.2, 20.2]], index=dates, columns=stocks)
    market_frames = {
        "Open": close - 0.1,
        "High": close + 0.2,
        "Low": close - 0.2,
        "Close": close,
        "Volume": pd.DataFrame(1000.0, index=dates, columns=stocks),
        "Amount": pd.DataFrame(10000.0, index=dates, columns=stocks),
    }
    lake = ResearchDataLake(tmp_path / "lake")
    old_bundle = lake.save_market_data_bundle(
        spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "old"},
        market_frames=market_frames,
        benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
        benchmark_open=pd.Series(3995.0, index=dates, name="000300.SH"),
        membership_frame=pd.DataFrame(True, index=dates, columns=stocks),
        feature_frames={"score_none": close * 0.0, "score_v2": close * 0.0 + 0.1},
        source="old",
    )
    current_bundle = lake.save_market_data_bundle(
        spec={"pool_name": "learned_all_a", "benchmark": "000300.SH", "source": "current"},
        market_frames=market_frames,
        benchmark_close=pd.Series(4000.0, index=dates, name="000300.SH"),
        benchmark_open=pd.Series(3995.0, index=dates, name="000300.SH"),
        membership_frame=pd.DataFrame(True, index=dates, columns=stocks),
        feature_frames={"score_none": close * 0.0, "score_v2": close * 0.0 + 0.1},
        source="current",
    )
    lake.save_sector_board_view(
        spec={
            "source_market_dataset_id": old_bundle.dataset_id,
            "view_kind": "latest_static_snapshot",
            "view_name": "sector_board_latest_static",
            "snapshot_semantics": "latest_static_snapshot",
            "as_of_date": "2026-05-22",
        },
        industry_map_frame=pd.DataFrame(
            {
                "symbol": stocks,
                "industry": ["银行", "银行"],
                "source": ["synthetic", "synthetic"],
                "as_of_date": ["2026-05-22", "2026-05-22"],
            }
        ),
        board_membership_frame=pd.DataFrame(
            {
                "symbol": stocks,
                "board_kind": ["FG", "FG"],
                "board_name": ["高股息", "高股息"],
                "board_code": ["880002", "880002"],
                "source": ["synthetic", "synthetic"],
                "source_start_date": ["2020-01-01", "2020-01-01"],
                "source_update_date": ["2026-05-22", "2026-05-22"],
                "as_of_date": ["2026-05-22", "2026-05-22"],
            }
        ),
        board_summary_frame=pd.DataFrame({"board_kind": ["FG"], "board_name": ["高股息"], "board_code": ["880002"]}),
        source_cache={"cloned": False},
    )

    sector_view_id = production_signal._ensure_sector_board_view_for_dataset(
        data_lake_root=tmp_path / "lake",
        lake_dataset_id=current_bundle.dataset_id,
    )
    metadata = lake.describe_dataset(sector_view_id)
    prepared = load_policy_inputs_from_lake(
        lake=lake,
        dataset_id=current_bundle.dataset_id,
        sector_board_view_id=sector_view_id,
        start_date="2026-05-20",
        end_date="2026-05-22",
        benchmark="000300.SH",
    )

    assert metadata["source"] == current_bundle.dataset_id
    assert prepared.metadata_frames["industry_map"]["symbol"].tolist() == stocks
    assert prepared.metadata_frames["board_membership"]["symbol"].tolist() == stocks


def test_data_refresh_api_skips_when_current_dataset_is_latest(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    def fail_launch_task_async(**kwargs):
        raise AssertionError("latest complete dataset must not launch a real refresh task")

    monkeypatch.setattr(
        web_server.app_service,
        "data_sources_summary",
        lambda: {
            "status": "ok",
            "active_dataset_id": "policy_input_bundle__current",
            "latest_policy_input_dataset_id": "policy_input_bundle__current",
            "latest_policy_input_dataset_end_date": "2026-05-22",
            "dataset_sync_status": "synced",
            "is_current_dataset_latest": True,
            "is_current_dataset_complete": True,
            "next_refresh_action": "skip",
            "refresh_explanation": "当前 active dataset 已覆盖最新完成交易日 2026-05-22。",
            "data_platform": {
                "latest_completed_trading_date": "2026-05-22",
                "recommended_domains": FORMAL_REFRESH_DOMAINS,
                "default_refresh": {
                    "as_of_date": "2026-05-22",
                    "universe": "all_a",
                    "domains": FORMAL_REFRESH_DOMAINS,
                    "provider_plan": "formal_free_v3",
                },
            },
        },
    )
    monkeypatch.setattr(web_server.app_service, "launch_task_async", fail_launch_task_async)

    client = TestClient(web_server.create_app())
    response = client.post("/api/data-sources/refresh", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "skipped"
    assert payload["business_status"] == "synced"
    assert payload["dataset_id"] == "policy_input_bundle__current"
    assert "2026-05-22" in payload["summary_note"]


def test_data_refresh_skip_payload_includes_paper_reconcile(monkeypatch: pytest.MonkeyPatch) -> None:
    from daily_research.execution import app_service

    monkeypatch.setattr(
        app_service,
        "paper_account_reconcile",
        lambda as_of_date="": {"status": "ok", "as_of_date": as_of_date, "apply_result": {"pending_order_count": 0}},
    )

    payload = app_service.data_refresh_skip_payload(
        data_sources={
            "active_dataset_id": "policy_input_bundle__current",
            "refresh_explanation": "当前 active dataset 已覆盖最新完成交易日 2026-05-22。",
            "data_platform": {"latest_completed_trading_date": "2026-05-22"},
        },
        as_of_date="2026-05-22",
    )

    assert payload["paper_trading"]["status"] == "ok"
    assert payload["metadata"]["paper_trading"]["as_of_date"] == "2026-05-22"


def test_data_refresh_api_defaults_to_completed_date_and_formal_domains(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    captured: dict[str, list[str]] = {}

    def fake_launch_task_async(**kwargs):
        captured["passthrough_args"] = list(kwargs["passthrough_args"])
        return {"job_id": "job1", "task_name": kwargs["task_name"], "status": "queued"}

    monkeypatch.setattr(web_server.app_service, "get_latest_completed_trading_date", lambda: "2026-05-22", raising=False)
    monkeypatch.setattr(
        web_server.app_service,
        "data_sources_summary",
        lambda: {
            "status": "ok",
            "next_refresh_action": "refresh",
            "data_platform": {
                "latest_completed_trading_date": "2026-05-22",
                "default_refresh": {
                    "as_of_date": "2026-05-22",
                    "universe": "all_a",
                    "domains": FORMAL_REFRESH_DOMAINS,
                    "provider_plan": "formal_free_v3",
                },
            },
        },
    )
    monkeypatch.setattr(web_server.app_service, "launch_task_async", fake_launch_task_async)

    client = TestClient(web_server.create_app())
    response = client.post("/api/data-sources/refresh", json={})

    assert response.status_code == 200
    assert captured["passthrough_args"] == [
        "--as-of-date",
        "2026-05-22",
        "--universe",
        "all_a",
        "--provider-plan",
        "formal_free_v3",
        "--required-domains",
        ",".join(FORMAL_REQUIRED_DOMAINS),
        "--domains",
        ",".join(FORMAL_REFRESH_DOMAINS),
    ]
    assert "--timeout-seconds" not in captured["passthrough_args"]


def test_data_refresh_api_ignores_timeout_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    captured: dict[str, list[str]] = {}

    def fake_launch_task_async(**kwargs):
        captured["passthrough_args"] = list(kwargs["passthrough_args"])
        return {"job_id": "job1", "task_name": kwargs["task_name"], "status": "queued"}

    monkeypatch.setattr(web_server.app_service, "get_latest_completed_trading_date", lambda: "2026-05-22", raising=False)
    monkeypatch.setattr(
        web_server.app_service,
        "data_sources_summary",
        lambda: {
            "status": "ok",
            "next_refresh_action": "refresh",
            "data_platform": {
                "latest_completed_trading_date": "2026-05-22",
                "default_refresh": {
                    "as_of_date": "2026-05-22",
                    "universe": "all_a",
                    "domains": FORMAL_REFRESH_DOMAINS,
                    "provider_plan": "formal_free_v3",
                },
            },
        },
    )
    monkeypatch.setattr(web_server.app_service, "launch_task_async", fake_launch_task_async)

    client = TestClient(web_server.create_app())
    response = client.post("/api/data-sources/refresh", json={"advanced_args": "--timeout-seconds 1"})

    assert response.status_code == 200
    assert "--timeout-seconds" not in captured["passthrough_args"]
    assert "1" not in captured["passthrough_args"]


def test_provider_health_api_launches_background_progress_job(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    called: dict[str, Any] = {}

    def fake_launch_task_async(**kwargs):
        called.update(kwargs)
        return {
            "job_id": "provider-health-job",
            "task_name": kwargs["task_name"],
            "status": "queued",
            "command": ["python", "provider_health.py"],
        }

    monkeypatch.setattr(web_server.app_service, "launch_task_async", fake_launch_task_async)

    client = TestClient(web_server.create_app())
    response = client.post(
        "/api/data-sources/provider-health",
        json={"as_of_date": "2026-05-22", "domains": ["market_daily"], "provider_plan": "formal_free_v3"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["task_name"] == "provider-health-check"
    assert called["task_name"] == "provider-health-check"
    assert called["passthrough_args"] == [
        "--provider-plan",
        "formal_free_v3",
        "--as-of-date",
        "2026-05-22",
        "--domains",
        "market_daily",
        "--json",
    ]


def test_provider_health_sync_api_remains_available_for_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    called: dict[str, Any] = {}

    def fake_provider_health_summary(**kwargs):
        called.update(kwargs)
        return {
            "status": "ok",
            "provider_plan": "formal_free_v3",
            "summary": {"ok_domain_count": 2, "checked_domain_count": 2},
            "domain_matrix": [{"provider": "baostock", "domain": "market_daily"}],
            "providers": [],
        }

    monkeypatch.setattr(web_server.app_service, "provider_health_summary", fake_provider_health_summary)

    client = TestClient(web_server.create_app())
    response = client.post("/api/data-sources/provider-health/sync", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["domain_matrix"][0]["domain"] == "market_daily"
    assert called["provider_plan"] == "formal_free_v3"


def test_scheduler_api_is_read_only_and_patch_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    monkeypatch.setattr(
        web_server.scheduler_cli,
        "scheduler_status",
        lambda: {"status": "ok", "installed": True, "enabled": True, "time": "15:45", "task_name": "DailyResearchDailyPlan"},
    )

    client = TestClient(web_server.create_app())
    get_response = client.get("/api/data-sources/scheduler")
    patch_response = client.patch("/api/data-sources/scheduler", json={"enabled": False, "post_close_time": "15:45"})

    assert get_response.status_code == 200
    assert get_response.json()["task_name"] == "DailyResearchDailyPlan"
    assert patch_response.status_code == 405


def test_daily_run_api_returns_compact_status_and_does_not_expose_runtime_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    monkeypatch.setattr(
        web_server.daily_plan_runner,
        "daily_run_status",
        lambda: {
            "status": "blocked",
            "latest_run_date": "2026-05-26",
            "latest_verdict": {
                "status": "blocked",
                "blocker_code": "data_not_ready",
                "target_trading_date": "2026-05-26",
            },
            "scheduler": {"installed": True, "enabled": True},
        },
    )

    client = TestClient(web_server.create_app())
    response = client.get("/api/daily-run/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["latest_verdict"]["blocker_code"] == "data_not_ready"
    assert "recent_jobs" not in payload
    assert "current_job" not in payload
    assert "runtime_state" not in payload


def test_daily_run_api_launches_single_daily_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    captured: dict[str, Any] = {}

    def fake_launch(**kwargs):
        captured.update(kwargs)
        return {"job_id": "daily-plan-job", "task_name": kwargs["task_name"], "status": "queued"}

    monkeypatch.setattr(web_server.app_service, "launch_task_async", fake_launch)

    client = TestClient(web_server.create_app())
    response = client.post("/api/daily-run/run", json={"mode": "post-close"})

    assert response.status_code == 200
    assert response.json()["task_name"] == "daily-plan-runner"
    assert captured["task_name"] == "daily-plan-runner"
    assert "--mode" in captured["passthrough_args"]
    assert "post-close" in captured["passthrough_args"]


def test_data_readiness_api_reports_candidate_blocker(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    monkeypatch.setattr(
        web_server.data_readiness,
        "resolve_provider_ready_trading_date",
        lambda **kwargs: {
            "status": "blocked",
            "blocker_code": "data_not_ready",
            "candidate_date": "2026-05-26",
            "row_count": 0,
            "coverage_ratio": 0.0,
        },
    )

    client = TestClient(web_server.create_app())
    response = client.get("/api/data-readiness")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "blocked"
    assert payload["blocker_code"] == "data_not_ready"
    assert payload["candidate_date"] == "2026-05-26"


def test_web_scheduler_tick_interface_removed() -> None:
    from daily_research.execution import app_service

    assert not hasattr(app_service, "run_scheduler_tick")
    assert not hasattr(app_service, "update_scheduler_config")
    assert not hasattr(app_service, "scheduler_summary")


def test_trade_plan_summary_explains_empty_structured_plan(tmp_path: Path) -> None:
    from daily_research.execution import app_service

    output_dir = tmp_path / "execution" / "output"
    run_dir = output_dir / "20260522"
    run_dir.mkdir(parents=True)
    (output_dir / "latest_trade_plan.txt").write_text(
        "市场状态: trend_up_low_vol\n"
        "市场过滤: 关闭 | 当前市场状态仅展示，不拦截开仓\n"
        "external score context fallback: no usable rows for signal date\n",
        encoding="utf-8",
    )
    (run_dir / "daily_trade_plan.txt").write_text("run text\n", encoding="utf-8")
    (run_dir / "plan_summary.json").write_text(
        json.dumps(
            {
                "signal_date": "2026-05-22",
                "current_position_count": 0,
                "target_position_count": 0,
                "actionable_target_position_count": 0,
                "candidate_total_rows": 2249,
                "candidate_usable_rows": 471,
                "candidate_dropped_rows": 1778,
                "blocked_buy_candidate_count": 0,
                "regime_state": "trend_up_low_vol",
                "market_filter_text": "关闭 | 当前市场状态仅展示，不拦截开仓",
                "score_context_status": "fallback_no_signal_rows",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "actions_today.csv").write_text("stock,action,target_weight\n", encoding="utf-8-sig")
    (run_dir / "holdings_snapshot.csv").write_text("stock,current_weight,target_weight\n", encoding="utf-8-sig")
    (run_dir / "watchlist.csv").write_text("stock,score\n", encoding="utf-8-sig")

    payload = app_service.latest_trade_plan_summary(output_dir=output_dir)

    assert payload["status"] == "ok"
    assert payload["actions"] == []
    assert payload["watchlist"] == []
    assert payload["diagnostics"]["regime_state"] == "trend_up_low_vol"
    assert payload["diagnostics"]["market_filter_text"] == "关闭 | 当前市场状态仅展示，不拦截开仓"
    assert payload["diagnostics"]["candidate_total_rows"] == 2249
    assert payload["diagnostics"]["candidate_usable_rows"] == 471
    assert payload["diagnostics"]["candidate_dropped_rows"] == 1778
    assert "无当前持仓" in payload["diagnostics"]["empty_plan_reason"]
    assert "无可执行目标" in payload["diagnostics"]["empty_plan_reason"]
    assert "score panel" in payload["diagnostics"]["empty_plan_reason"]


def test_latest_policy_input_dataset_prefers_newest_end_date_then_creation() -> None:
    from daily_research.execution import app_service

    frame = pd.DataFrame(
        [
            {
                "dataset_id": "policy_input_bundle__old_created_later",
                "dataset_kind": "policy_input_bundle",
                "end_date": "2026-05-21",
                "created_at": "2026-05-24T10:00:00",
                "source": "data_platform_refresh",
                "status": "stored",
            },
            {
                "dataset_id": "policy_input_bundle__latest",
                "dataset_kind": "policy_input_bundle",
                "end_date": "2026-05-22",
                "created_at": "2026-05-24T00:18:28",
                "source": "data_platform_refresh",
                "status": "stored",
            },
            {
                "dataset_id": "not_policy_bundle",
                "dataset_kind": "gold",
                "end_date": "2026-05-23",
                "created_at": "2026-05-24T11:00:00",
                "source": "research",
                "status": "stored",
            },
        ]
    )

    selected = app_service._select_latest_policy_input_lake_dataset(frame)

    assert selected["dataset_id"] == "policy_input_bundle__latest"
    assert selected["end_date"] == "2026-05-22"


def test_sync_active_manifest_to_latest_lake_dataset_writes_execution_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from daily_research.execution import app_service

    active_manifest = tmp_path / "active_execution_strategy.json"
    active_manifest.write_text(
        json.dumps(
            {
                "strategy_name": "active",
                "data_source": "lake",
                "lake_dataset_id": "policy_input_bundle__old",
                "source_market_dataset_id": "policy_input_bundle__old",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_service, "ACTIVE_MANIFEST_PATH", active_manifest)
    monkeypatch.setattr(
        app_service,
        "_latest_policy_input_lake_dataset",
        lambda lake_root=None: {
            "dataset_id": "policy_input_bundle__latest",
            "end_date": "2026-05-22",
            "created_at": "2026-05-24T00:18:28",
            "source": "data_platform_refresh",
            "status": "stored",
        },
    )
    monkeypatch.setattr(app_service, "append_event", lambda *args, **kwargs: None)

    result = app_service.sync_active_manifest_to_latest_lake_dataset(
        reason="data-platform-refresh",
        refresh_manifest_path=str(tmp_path / "refresh_manifest.json"),
    )
    payload = json.loads(active_manifest.read_text(encoding="utf-8"))

    assert result["updated"] is True
    assert result["dataset_id"] == "policy_input_bundle__latest"
    assert payload["data_source"] == "lake"
    assert payload["lake_dataset_id"] == "policy_input_bundle__latest"
    assert payload["source_market_dataset_id"] == "policy_input_bundle__latest"
    assert payload["lake_dataset_end_date"] == "2026-05-22"
    assert payload["latest_data_refresh_manifest"].endswith("refresh_manifest.json")


def test_data_refresh_job_success_syncs_active_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daily_research.execution import app_service

    script = tmp_path / "refresh_success.py"
    script.write_text("print('refresh ok')\n", encoding="utf-8")
    refresh_manifest_path = tmp_path / "refresh_run" / "refresh_manifest.json"
    refresh_manifest_path.parent.mkdir()
    refresh_manifest_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "registered_market_dataset_id": "policy_input_bundle__latest",
                "policy_input_dataset_id": "policy_input_bundle__latest",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    signal_manifest_path = tmp_path / "signal" / "live_panel_refresh_manifest.json"

    class FakeSpec:
        description = "refresh data"

    def fake_create_job_record(**kwargs):
        job_dir = tmp_path / "job"
        job_dir.mkdir(exist_ok=True)

        class Paths:
            job_id = "refresh_job"
            stdout_path = job_dir / "stdout.log"
            stderr_path = job_dir / "stderr.log"
            metadata_path = job_dir / "metadata.json"

        return Paths()

    def fake_mark_job_finished(job_paths, **kwargs):
        return {
            "job_id": job_paths.job_id,
            "status": kwargs["status"],
            "exit_code": kwargs["exit_code"],
            "summary_note": kwargs["summary_note"],
        }

    captured: dict[str, object] = {}

    def fake_sync(**kwargs):
        captured.update(kwargs)
        return {"updated": True, "dataset_id": "policy_input_bundle__latest", "end_date": "2026-05-22"}

    monkeypatch.setattr(app_service, "get_task_spec", lambda _: FakeSpec())
    monkeypatch.setattr(app_service, "build_task_command", lambda **_: [sys.executable, str(script)])
    monkeypatch.setattr(app_service, "create_job_record", fake_create_job_record)
    monkeypatch.setattr(app_service, "mark_job_started", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "mark_job_finished", fake_mark_job_finished)
    monkeypatch.setattr(app_service, "update_job_metadata", lambda job_paths, **kwargs: {"active_manifest_update": kwargs["active_manifest_update"]})
    monkeypatch.setattr(app_service, "append_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "sync_active_manifest_to_latest_lake_dataset", fake_sync)
    monkeypatch.setattr(app_service, "_latest_refresh_manifest_path", lambda: str(refresh_manifest_path.resolve()))
    monkeypatch.setattr(
        app_service,
        "_signal_refresh_metadata",
        lambda **_: {
            "business_status": "ok",
            "runner_status": "ok",
            "artifact_status": "ok",
            "signal_refresh_manifest_path": str(signal_manifest_path.resolve()),
            "panel_latest_date": "2026-05-22",
            "artifact_paths": {"live_panel_refresh_manifest": str(signal_manifest_path.resolve())},
            "evidence_paths": {"live_panel_refresh_manifest": str(signal_manifest_path.resolve())},
        },
    )

    class FakeLock:
        payload = {}

        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(app_service, "ExecutionAppLock", FakeLock)

    result = app_service.run_task_sync(task_name="data-platform-refresh", passthrough_args=[], echo_output=False)

    assert result["status"] == "succeeded"
    assert captured["reason"] == "data-platform-refresh"
    assert "policy_input_bundle__latest" in result["metadata"]["summary_note"]
    assert result["metadata"]["active_manifest_update"]["dataset_id"] == "policy_input_bundle__latest"


def test_data_refresh_reconciles_successful_manifest_after_runner_output_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from daily_research.execution import app_service

    script = tmp_path / "refresh_manifest_then_output_error.py"
    refresh_manifest_path = tmp_path / "refresh_run" / "refresh_manifest.json"
    refresh_manifest_text = str(refresh_manifest_path).replace("\\", "/")
    script.write_text(
        "from pathlib import Path\n"
        "import json\n"
        f"manifest_path = Path({refresh_manifest_text!r})\n"
        "manifest_path.parent.mkdir(exist_ok=True)\n"
        "manifest_path.write_text(json.dumps({\n"
        "    'status': 'ok',\n"
        "    'refresh_run_id': 'refresh_ok',\n"
        "    'registered_market_dataset_id': 'policy_input_bundle__latest',\n"
        "    'policy_input_dataset_id': 'policy_input_bundle__latest',\n"
        "}, ensure_ascii=False), encoding='utf-8')\n"
        "print('business refresh completed')\n",
        encoding="utf-8",
    )

    class FakeSpec:
        description = "refresh data"

    def fake_create_job_record(**kwargs):
        job_dir = tmp_path / "job"
        job_dir.mkdir(exist_ok=True)

        class Paths:
            job_id = "refresh_job_reconciled"
            stdout_path = job_dir / "stdout.log"
            stderr_path = job_dir / "stderr.log"
            metadata_path = job_dir / "metadata.json"

        return Paths()

    finished: dict[str, object] = {}

    def fake_mark_job_finished(job_paths, **kwargs):
        finished.update(kwargs)
        return {
            "job_id": job_paths.job_id,
            "status": kwargs["status"],
            "exit_code": kwargs["exit_code"],
            "summary_note": kwargs["summary_note"],
            "business_status": kwargs.get("business_status", ""),
            "runner_status": kwargs.get("runner_status", ""),
            "runner_warnings": kwargs.get("runner_warnings", []),
            "artifact_status": kwargs.get("artifact_status", ""),
            "artifact_paths": kwargs.get("artifact_paths", {}),
        }

    captured: dict[str, object] = {}

    def fake_sync(**kwargs):
        captured.update(kwargs)
        return {"updated": True, "dataset_id": "policy_input_bundle__latest"}

    class BrokenFlush:
        def write(self, text: str) -> int:
            return len(text)

        def flush(self) -> None:
            raise OSError(22, "Invalid argument")

    monkeypatch.setattr(app_service, "get_task_spec", lambda _: FakeSpec())
    monkeypatch.setattr(app_service, "build_task_command", lambda **_: [sys.executable, str(script)])
    monkeypatch.setattr(app_service, "create_job_record", fake_create_job_record)
    monkeypatch.setattr(app_service, "mark_job_started", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "mark_job_finished", fake_mark_job_finished)
    monkeypatch.setattr(app_service, "update_job_metadata", lambda job_paths, **kwargs: dict(kwargs))
    monkeypatch.setattr(app_service, "append_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "sync_active_manifest_to_latest_lake_dataset", fake_sync)
    monkeypatch.setattr(app_service, "_latest_refresh_manifest_path", lambda: str(refresh_manifest_path.resolve()))
    monkeypatch.setattr(
        app_service,
        "_signal_refresh_metadata",
        lambda **_: {
            "business_status": "ok",
            "runner_status": "ok",
            "artifact_status": "ok",
            "signal_refresh_manifest_path": str((tmp_path / "signal" / "live_panel_refresh_manifest.json").resolve()),
            "panel_latest_date": "2026-05-22",
            "artifact_paths": {
                "live_panel_refresh_manifest": str((tmp_path / "signal" / "live_panel_refresh_manifest.json").resolve())
            },
            "evidence_paths": {
                "live_panel_refresh_manifest": str((tmp_path / "signal" / "live_panel_refresh_manifest.json").resolve())
            },
        },
    )
    monkeypatch.setattr(app_service.sys, "stdout", BrokenFlush())

    class FakeLock:
        payload = {}

        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(app_service, "ExecutionAppLock", FakeLock)

    result = app_service.run_task_sync(
        task_name="data-platform-refresh",
        passthrough_args=[],
        echo_output=True,
    )

    assert result["status"] == "succeeded"
    assert result["metadata"]["business_status"] == "ok"
    assert result["metadata"]["runner_status"] == "warning"
    assert result["metadata"]["runner_warnings"]
    assert result["metadata"]["artifact_status"] == "ok"
    assert result["metadata"]["artifact_paths"]["refresh_manifest"].endswith("refresh_manifest.json")
    assert captured["refresh_manifest_path"].endswith("refresh_manifest.json")
    assert finished["exit_code"] == 0


def test_data_refresh_post_process_persists_evidence_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daily_research.execution import app_service

    refresh_manifest_path = tmp_path / "refresh_run" / "refresh_manifest.json"
    refresh_manifest_path.parent.mkdir()
    refresh_manifest_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "registered_market_dataset_id": "policy_input_bundle__latest",
                "policy_input_dataset_id": "policy_input_bundle__latest",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class Paths:
        job_id = "refresh_job"
        metadata_path = tmp_path / "metadata.json"

    persisted: dict[str, object] = {}

    def fake_update(job_paths, **kwargs):
        persisted.update(kwargs)
        return dict(persisted)

    monkeypatch.setattr(app_service, "_latest_refresh_manifest_path", lambda: str(refresh_manifest_path.resolve()))
    monkeypatch.setattr(app_service, "update_job_metadata", fake_update)
    monkeypatch.setattr(
        app_service,
        "sync_active_manifest_to_latest_lake_dataset",
        lambda **kwargs: {"updated": True, "dataset_id": "policy_input_bundle__latest"},
    )
    monkeypatch.setattr(
        app_service,
        "_signal_refresh_metadata",
        lambda **_: {
            "business_status": "ok",
            "runner_status": "ok",
            "artifact_status": "ok",
            "signal_refresh_manifest_path": str((tmp_path / "signal" / "live_panel_refresh_manifest.json").resolve()),
            "panel_latest_date": "2026-05-22",
            "artifact_paths": {
                "live_panel_refresh_manifest": str((tmp_path / "signal" / "live_panel_refresh_manifest.json").resolve())
            },
            "evidence_paths": {
                "live_panel_refresh_manifest": str((tmp_path / "signal" / "live_panel_refresh_manifest.json").resolve())
            },
        },
    )

    _, metadata = app_service._post_process_successful_task(
        job_paths=Paths(),
        task_name="data-platform-refresh",
        summary_note="refresh data",
    )

    assert metadata["business_status"] == "ok"
    assert persisted["business_status"] == "ok"
    assert persisted["artifact_status"] == "ok"
    assert persisted["artifact_paths"]["refresh_manifest"].endswith("refresh_manifest.json")
    assert persisted["active_manifest_update"]["dataset_id"] == "policy_input_bundle__latest"


def test_data_refresh_custom_lake_root_skips_active_manifest_sync(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daily_research.execution import app_service

    custom_lake_root = tmp_path / "custom_lake"
    refresh_manifest_path = custom_lake_root / "data_platform" / "runs" / "smoke_run" / "refresh_manifest.json"
    refresh_manifest_path.parent.mkdir(parents=True)
    refresh_manifest_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "registered_market_dataset_id": "policy_input_bundle__smoke",
                "policy_input_dataset_id": "policy_input_bundle__smoke",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class Paths:
        job_id = "custom_lake_refresh"
        metadata_path = tmp_path / "metadata.json"

    def fail_sync(**kwargs):
        raise AssertionError(f"custom data lake smoke must not sync active manifest: {kwargs}")

    persisted: dict[str, object] = {}

    def fake_update(job_paths, **kwargs):
        persisted.update(kwargs)
        return dict(persisted)

    monkeypatch.setattr(app_service, "update_job_metadata", fake_update)
    monkeypatch.setattr(app_service, "sync_active_manifest_to_latest_lake_dataset", fail_sync)

    note, metadata = app_service._post_process_successful_task(
        job_paths=Paths(),
        task_name="data-platform-refresh",
        summary_note="refresh data",
        passthrough_args=[
            "--data-lake-root",
            str(custom_lake_root),
            "--run-id",
            "smoke_run",
        ],
    )

    assert metadata["business_status"] == "ok"
    assert metadata["active_manifest_update"]["status"] == "skipped"
    assert metadata["active_manifest_update"]["reason"] == "custom_data_lake_root"
    assert metadata["artifact_paths"]["refresh_manifest"] == str(refresh_manifest_path.resolve())
    assert "active manifest unchanged" in note


def test_data_refresh_custom_lake_root_without_run_id_does_not_use_latest_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from daily_research.execution import app_service

    default_manifest_path = tmp_path / "default_lake" / "data_platform" / "runs" / "old" / "refresh_manifest.json"
    default_manifest_path.parent.mkdir(parents=True)
    default_manifest_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "registered_market_dataset_id": "policy_input_bundle__old",
                "policy_input_dataset_id": "policy_input_bundle__old",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class Paths:
        job_id = "custom_lake_missing_run_id"
        metadata_path = tmp_path / "metadata.json"

    monkeypatch.setattr(app_service, "_latest_refresh_manifest_path", lambda: str(default_manifest_path.resolve()))
    monkeypatch.setattr(app_service, "update_job_metadata", lambda job_paths, **kwargs: dict(kwargs))
    monkeypatch.setattr(
        app_service,
        "sync_active_manifest_to_latest_lake_dataset",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(f"custom lake without run-id must not sync: {kwargs}")),
    )

    note, metadata = app_service._post_process_successful_task(
        job_paths=Paths(),
        task_name="data-platform-refresh",
        summary_note="refresh data",
        passthrough_args=[
            "--data-lake-root",
            str(tmp_path / "custom_lake_without_run_id"),
        ],
    )

    assert "missing" in note
    assert metadata["business_status"] == "missing"
    assert metadata["artifact_status"] == "missing"
    assert metadata["artifact_paths"]["refresh_manifest"] == ""
    assert "active_manifest_update" not in metadata


def test_data_refresh_signal_refresh_failure_marks_job_failed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daily_research.execution import app_service

    script = tmp_path / "refresh_manifest_ok.py"
    refresh_manifest_path = tmp_path / "refresh_run" / "refresh_manifest.json"
    refresh_manifest_text = str(refresh_manifest_path).replace("\\", "/")
    script.write_text(
        "from pathlib import Path\n"
        "import json\n"
        f"manifest_path = Path({refresh_manifest_text!r})\n"
        "manifest_path.parent.mkdir(exist_ok=True)\n"
        "manifest_path.write_text(json.dumps({\n"
        "    'status': 'ok',\n"
        "    'registered_market_dataset_id': 'policy_input_bundle__latest',\n"
        "    'policy_input_dataset_id': 'policy_input_bundle__latest',\n"
        "}, ensure_ascii=False), encoding='utf-8')\n",
        encoding="utf-8",
    )

    class FakeSpec:
        description = "refresh data"

    def fake_create_job_record(**kwargs):
        job_dir = tmp_path / "job"
        job_dir.mkdir(exist_ok=True)

        class Paths:
            job_id = "refresh_signal_failed"
            stdout_path = job_dir / "stdout.log"
            stderr_path = job_dir / "stderr.log"
            metadata_path = job_dir / "metadata.json"

        return Paths()

    monkeypatch.setattr(app_service, "get_task_spec", lambda _: FakeSpec())
    monkeypatch.setattr(app_service, "build_task_command", lambda **_: [sys.executable, str(script)])
    monkeypatch.setattr(app_service, "create_job_record", fake_create_job_record)
    monkeypatch.setattr(app_service, "mark_job_started", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "append_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "_latest_refresh_manifest_path", lambda: str(refresh_manifest_path.resolve()))
    monkeypatch.setattr(
        app_service,
        "sync_active_manifest_to_latest_lake_dataset",
        lambda **kwargs: {"updated": True, "dataset_id": "policy_input_bundle__latest", "end_date": "2026-05-22"},
    )
    monkeypatch.setattr(
        app_service,
        "_signal_refresh_metadata",
        lambda **_: {
            "business_status": "signal_failed",
            "runner_status": "ok",
            "artifact_status": "failed",
            "signal_refresh_error": "score panel export failed",
            "artifact_paths": {},
            "evidence_paths": {},
        },
    )

    class FakeLock:
        payload = {}

        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(app_service, "ExecutionAppLock", FakeLock)

    result = app_service.run_task_sync(task_name="data-platform-refresh", passthrough_args=[], echo_output=False)

    assert result["status"] == "failed"
    assert result["metadata"]["business_status"] == "signal_failed"
    assert result["metadata"]["signal_refresh_result"]["signal_refresh_error"] == "score panel export failed"


def test_blocked_refresh_manifest_does_not_sync_active_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daily_research.execution import app_service

    refresh_manifest_path = tmp_path / "refresh_run" / "refresh_manifest.json"
    refresh_manifest_path.parent.mkdir()
    refresh_manifest_path.write_text(
        json.dumps(
            {
                "status": "blocked",
                "blockers": ["coverage_below_threshold"],
                "registered_market_dataset_id": "",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class Paths:
        job_id = "refresh_blocked"
        metadata_path = tmp_path / "metadata.json"

    persisted: dict[str, object] = {}

    monkeypatch.setattr(app_service, "_latest_refresh_manifest_path", lambda: str(refresh_manifest_path.resolve()))
    monkeypatch.setattr(app_service, "update_job_metadata", lambda job_paths, **kwargs: persisted.update(kwargs) or dict(persisted))
    monkeypatch.setattr(
        app_service,
        "sync_active_manifest_to_latest_lake_dataset",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("blocked refresh must not sync active manifest")),
    )

    note, metadata = app_service._post_process_successful_task(
        job_paths=Paths(),
        task_name="data-platform-refresh",
        summary_note="refresh data",
    )

    assert "blocked" in note
    assert metadata["business_status"] == "blocked"
    assert "active_manifest_update" not in metadata
    assert persisted["artifact_status"] == "blocked"


def test_blocked_refresh_manifest_marks_job_blocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daily_research.execution import app_service

    script = tmp_path / "refresh_blocked.py"
    refresh_manifest_path = tmp_path / "refresh_run" / "refresh_manifest.json"
    refresh_manifest_text = str(refresh_manifest_path).replace("\\", "/")
    script.write_text(
        "from pathlib import Path\n"
        "import json\n"
        f"manifest_path = Path({refresh_manifest_text!r})\n"
        "manifest_path.parent.mkdir(exist_ok=True)\n"
        "manifest_path.write_text(json.dumps({\n"
        "    'status': 'blocked',\n"
        "    'blockers': ['coverage_below_threshold'],\n"
        "    'registered_market_dataset_id': '',\n"
        "}, ensure_ascii=False), encoding='utf-8')\n"
        "print('refresh blocked by coverage')\n",
        encoding="utf-8",
    )

    class FakeSpec:
        description = "refresh data"

    def fake_create_job_record(**kwargs):
        job_dir = tmp_path / "job"
        job_dir.mkdir(exist_ok=True)

        class Paths:
            job_id = "refresh_blocked_job"
            stdout_path = job_dir / "stdout.log"
            stderr_path = job_dir / "stderr.log"
            metadata_path = job_dir / "metadata.json"

        return Paths()

    monkeypatch.setattr(app_service, "get_task_spec", lambda _: FakeSpec())
    monkeypatch.setattr(app_service, "build_task_command", lambda **_: [sys.executable, str(script)])
    monkeypatch.setattr(app_service, "create_job_record", fake_create_job_record)
    monkeypatch.setattr(app_service, "mark_job_started", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "append_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "_latest_refresh_manifest_path", lambda: str(refresh_manifest_path.resolve()))
    monkeypatch.setattr(
        app_service,
        "sync_active_manifest_to_latest_lake_dataset",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("blocked refresh must not sync active manifest")),
    )

    class FakeLock:
        payload = {}

        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(app_service, "ExecutionAppLock", FakeLock)

    result = app_service.run_task_sync(task_name="data-platform-refresh", passthrough_args=[], echo_output=False)

    assert result["status"] == "blocked"
    assert result["exit_code"] == 0
    assert result["metadata"]["business_status"] == "blocked"
    assert result["metadata"]["runner_status"] == "ok"
    assert result["metadata"]["artifact_status"] == "blocked"


def test_blocked_refresh_manifest_with_nonzero_exit_is_business_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from daily_research.execution import app_service

    script = tmp_path / "refresh_blocked_exit2.py"
    refresh_manifest_path = tmp_path / "refresh_run" / "refresh_manifest.json"
    refresh_manifest_text = str(refresh_manifest_path).replace("\\", "/")
    script.write_text(
        "from pathlib import Path\n"
        "import json\n"
        "import sys\n"
        f"manifest_path = Path({refresh_manifest_text!r})\n"
        "manifest_path.parent.mkdir(exist_ok=True)\n"
        "manifest_path.write_text(json.dumps({\n"
        "    'status': 'blocked',\n"
        "    'blockers': ['severe_source_conflict'],\n"
        "    'registered_market_dataset_id': '',\n"
        "}, ensure_ascii=False), encoding='utf-8')\n"
        "print('blocked: ' + str(manifest_path), flush=True)\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )

    class FakeSpec:
        description = "refresh data"
        timeout_seconds = 0

    def fake_create_job_record(**kwargs):
        job_dir = tmp_path / "job"
        job_dir.mkdir(exist_ok=True)

        class Paths:
            job_id = "refresh_blocked_exit2_job"
            stdout_path = job_dir / "stdout.log"
            stderr_path = job_dir / "stderr.log"
            metadata_path = job_dir / "metadata.json"

        return Paths()

    monkeypatch.setattr(app_service, "get_task_spec", lambda _: FakeSpec())
    monkeypatch.setattr(app_service, "build_task_command", lambda **_: [sys.executable, str(script)])
    monkeypatch.setattr(app_service, "create_job_record", fake_create_job_record)
    monkeypatch.setattr(app_service, "mark_job_started", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "append_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "_latest_refresh_manifest_path", lambda: str(refresh_manifest_path.resolve()))
    monkeypatch.setattr(
        app_service,
        "sync_active_manifest_to_latest_lake_dataset",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("blocked refresh must not sync active manifest")),
    )

    class FakeLock:
        payload = {}

        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(app_service, "ExecutionAppLock", FakeLock)

    result = app_service.run_task_sync(task_name="data-platform-refresh", passthrough_args=[], echo_output=False)

    assert result["status"] == "blocked"
    assert result["exit_code"] == 2
    assert result["metadata"]["business_status"] == "blocked"
    assert result["metadata"]["runner_status"] == "ok"
    assert result["metadata"]["artifact_status"] == "blocked"


def test_missing_refresh_manifest_marks_job_failed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daily_research.execution import app_service

    script = tmp_path / "refresh_without_manifest.py"
    script.write_text("print('refresh script exited without manifest')\n", encoding="utf-8")
    missing_manifest_path = tmp_path / "missing" / "refresh_manifest.json"

    class FakeSpec:
        description = "refresh data"

    def fake_create_job_record(**kwargs):
        job_dir = tmp_path / "job"
        job_dir.mkdir(exist_ok=True)

        class Paths:
            job_id = "refresh_missing_manifest"
            stdout_path = job_dir / "stdout.log"
            stderr_path = job_dir / "stderr.log"
            metadata_path = job_dir / "metadata.json"

        return Paths()

    monkeypatch.setattr(app_service, "get_task_spec", lambda _: FakeSpec())
    monkeypatch.setattr(app_service, "build_task_command", lambda **_: [sys.executable, str(script)])
    monkeypatch.setattr(app_service, "create_job_record", fake_create_job_record)
    monkeypatch.setattr(app_service, "mark_job_started", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "append_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "_latest_refresh_manifest_path", lambda: str(missing_manifest_path.resolve()))
    monkeypatch.setattr(
        app_service,
        "sync_active_manifest_to_latest_lake_dataset",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("missing refresh manifest must not sync active manifest")),
    )

    class FakeLock:
        payload = {}

        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(app_service, "ExecutionAppLock", FakeLock)

    result = app_service.run_task_sync(task_name="data-platform-refresh", passthrough_args=[], echo_output=False)

    assert result["status"] == "failed"
    assert result["exit_code"] == 0
    assert result["metadata"]["business_status"] == "missing"
    assert result["metadata"]["runner_status"] == "ok"
    assert result["metadata"]["artifact_status"] == "missing"


def test_failed_data_refresh_does_not_reconcile_from_old_latest_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from daily_research.execution import app_service

    script = tmp_path / "refresh_failed_without_manifest.py"
    script.write_text(
        "import sys\n"
        "print('refresh started')\n"
        "print('refresh failed before manifest', file=sys.stderr)\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    old_manifest_path = tmp_path / "old_refresh" / "refresh_manifest.json"
    old_manifest_path.parent.mkdir()
    old_manifest_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "registered_market_dataset_id": "policy_input_bundle__old",
                "policy_input_dataset_id": "policy_input_bundle__old",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class FakeSpec:
        description = "refresh data"
        timeout_seconds = 0

    def fake_create_job_record(**kwargs):
        job_dir = tmp_path / "job"
        job_dir.mkdir(exist_ok=True)

        class Paths:
            job_id = "refresh_failed_without_manifest"
            stdout_path = job_dir / "stdout.log"
            stderr_path = job_dir / "stderr.log"
            metadata_path = job_dir / "metadata.json"

        return Paths()

    monkeypatch.setattr(app_service, "get_task_spec", lambda _: FakeSpec())
    monkeypatch.setattr(app_service, "build_task_command", lambda **_: [sys.executable, str(script)])
    monkeypatch.setattr(app_service, "create_job_record", fake_create_job_record)
    monkeypatch.setattr(app_service, "mark_job_started", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "append_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "_latest_refresh_manifest_path", lambda: str(old_manifest_path.resolve()))
    monkeypatch.setattr(
        app_service,
        "sync_active_manifest_to_latest_lake_dataset",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(f"failed refresh must not sync from old manifest: {kwargs}")),
    )

    class FakeLock:
        payload = {}

        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(app_service, "ExecutionAppLock", FakeLock)

    result = app_service.run_task_sync(task_name="data-platform-refresh", passthrough_args=[], echo_output=False)

    assert result["status"] == "failed"
    assert result["exit_code"] == 1
    assert result["metadata"]["business_status"] == "failed"
    assert result["metadata"]["artifact_status"] == ""
    assert "policy_input_bundle__old" not in json.dumps(result["metadata"], ensure_ascii=False)


def test_tee_text_io_compresses_repeated_carriage_progress() -> None:
    from daily_research.execution.app_service import _TeeTextIO

    handle = StringIO()
    heartbeats: list[str] = []
    tee = _TeeTextIO(handle, heartbeat=heartbeats.append)

    tee.write("stage 1\r")
    tee.write("stage 1\r")
    tee.write("stage 2\r")
    tee.write("stage 2\r")
    tee.write("done\n")
    tee.flush()

    assert handle.getvalue().splitlines() == ["stage 1", "stage 2", "done"]
    assert heartbeats == ["stage 1", "stage 2", "done"]


def test_task_launch_strips_timeout_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daily_research.execution import app_service

    captured: dict[str, list[str]] = {}

    def fake_build_task_command(**kwargs):
        captured["passthrough_args"] = list(kwargs["passthrough_args"])
        return [sys.executable, str(tmp_path / "noop.py"), *captured["passthrough_args"]]

    def fake_create_job_record(**kwargs):
        job_dir = tmp_path / "job"
        job_dir.mkdir(exist_ok=True)

        class Paths:
            job_id = "job_without_timeout"
            stdout_path = job_dir / "stdout.log"
            stderr_path = job_dir / "stderr.log"
            metadata_path = job_dir / "metadata.json"

        return Paths()

    monkeypatch.setattr(app_service, "build_task_command", fake_build_task_command)
    monkeypatch.setattr(app_service, "create_job_record", fake_create_job_record)
    monkeypatch.setattr(app_service, "append_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "LOCK_PATH", tmp_path / "execution_app.lock")
    monkeypatch.setattr(app_service, "STATE_PATH", tmp_path / "runtime_state.json")
    monkeypatch.setattr(app_service, "RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(
        app_service,
        "_spawn_detached_job_worker",
        lambda **kwargs: {
            "worker_launcher_pid": 12345,
            "worker_command_argv": [sys.executable, "-m", "daily_research.execution.job_worker", "--job-id", "job_without_timeout"],
            "worker_stdout_log": str(tmp_path / "worker_stdout.log"),
            "worker_stderr_log": str(tmp_path / "worker_stderr.log"),
        },
        raising=False,
    )

    app_service.launch_task_async(
        task_name="data-platform-refresh",
        passthrough_args=["--as-of-date", "2026-05-22", "--timeout-seconds", "1", "--universe", "all_a"],
    )

    assert captured["passthrough_args"] == ["--as-of-date", "2026-05-22", "--universe", "all_a"]


def test_launch_task_async_uses_detached_worker_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daily_research.execution import app_service

    script = tmp_path / "noop.py"
    script.write_text("print('noop')\n", encoding="utf-8")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    detached_metadata_path = job_dir / "metadata.json"
    detached_metadata_path.write_text("{}", encoding="utf-8")

    class Paths:
        job_id = "detached_job"
        job_root = job_dir
        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"
        metadata_path = detached_metadata_path

    class FakeProcess:
        pid = 24680

    captured: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = dict(kwargs)
        return FakeProcess()

    updates: dict[str, object] = {}

    monkeypatch.setattr(app_service, "build_task_command", lambda **_: [sys.executable, str(script)])
    monkeypatch.setattr(app_service, "create_job_record", lambda **_: Paths())
    monkeypatch.setattr(app_service, "update_job_metadata", lambda job_paths, **kwargs: updates.update(kwargs) or dict(updates))
    monkeypatch.setattr(app_service, "append_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "LOCK_PATH", tmp_path / "execution_app.lock")
    monkeypatch.setattr(app_service, "STATE_PATH", tmp_path / "runtime_state.json")
    monkeypatch.setattr(app_service, "RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(app_service.subprocess, "Popen", fake_popen)

    result = app_service.launch_task_async(task_name="data-platform-refresh", passthrough_args=[])

    command = captured["command"]
    assert command[:3] == [sys.executable, "-m", "daily_research.execution.job_worker"]
    assert "--job-id" in command
    assert command[command.index("--job-id") + 1] == "detached_job"
    assert captured["kwargs"]["cwd"] == str(app_service.WORKSPACE_ROOT)
    assert captured["kwargs"]["env"]["PYTHONUNBUFFERED"] == "1"
    assert updates["async_runner_mode"] == "detached_worker"
    assert updates["worker_launcher_pid"] == 24680
    assert result["worker_launcher_pid"] == 24680


def test_detached_job_worker_exits_zero_for_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from daily_research.execution import job_worker

    monkeypatch.setattr(
        job_worker.app_service,
        "run_recorded_job_sync",
        lambda **_: {"status": "succeeded", "exit_code": 0},
    )
    monkeypatch.setattr(job_worker, "append_event", lambda *args, **kwargs: None)

    assert job_worker.main(["--job-id", "ok_job"]) == 0


def test_execution_lock_allows_same_job_to_claim_launch_pending(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from daily_research.execution import app_runtime

    runtime_root = tmp_path / "runtime"
    jobs_root = runtime_root / "jobs"
    jobs_root.mkdir(parents=True)
    lock_path = runtime_root / "execution_app.lock"
    state_path = runtime_root / "runtime_state.json"
    lock_path.write_text(
        json.dumps(
            {
                "job_id": "pending_job",
                "task_name": "execution-smoke",
                "status": "launch_pending",
                "acquired_at": "2026-05-26T01:51:16+08:00",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    state_path.write_text(json.dumps({"lock": {"job_id": "pending_job"}, "current_job": {"job_id": "pending_job"}}), encoding="utf-8")

    monkeypatch.setattr(app_runtime, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(app_runtime, "JOBS_ROOT", jobs_root)
    monkeypatch.setattr(app_runtime, "LOCK_PATH", lock_path)
    monkeypatch.setattr(app_runtime, "STATE_PATH", state_path)
    monkeypatch.setattr(app_runtime, "EVENTS_PATH", runtime_root / "events.jsonl")

    with app_runtime.ExecutionAppLock(job_id="pending_job", task_name="execution-smoke") as lock:
        assert lock.payload["job_id"] == "pending_job"
        assert lock.payload["pid"] == os.getpid()


def test_launch_task_async_rejects_existing_active_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daily_research.execution import app_service

    monkeypatch.setattr(app_service, "_has_active_runtime_job", lambda: True)

    with pytest.raises(app_service.ExecutionAppLockError):
        app_service.launch_task_async(task_name="execution-smoke", passthrough_args=[])


def test_run_task_sync_executes_task_in_worker_subprocess(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daily_research.execution import app_service

    script = tmp_path / "inline_task.py"
    script.write_text(
        "import os\nimport sys\n"
        "print('inline task argv=' + '|'.join(sys.argv[1:]))\n"
        "print('worker pid=' + str(os.getpid()))\n",
        encoding="utf-8",
    )

    class FakeSpec:
        description = "inline task"
        timeout_seconds = 0

    def fake_create_job_record(**kwargs):
        job_dir = tmp_path / "job"
        job_dir.mkdir(exist_ok=True)
        metadata_file = job_dir / "metadata.json"
        metadata_file.write_text("{}", encoding="utf-8")

        class Paths:
            job_id = "inline_job"
            stdout_path = job_dir / "stdout.log"
            stderr_path = job_dir / "stderr.log"
            metadata_path = metadata_file

        return Paths()

    monkeypatch.setattr(app_service, "get_task_spec", lambda _: FakeSpec())
    monkeypatch.setattr(app_service, "build_task_command", lambda **_: [sys.executable, str(script), "--flag", "x"])
    monkeypatch.setattr(app_service, "create_job_record", fake_create_job_record)
    monkeypatch.setattr(app_service, "mark_job_started", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_service, "mark_job_finished", lambda job_paths, **kwargs: {"status": kwargs["status"], "exit_code": kwargs["exit_code"]})
    monkeypatch.setattr(app_service, "append_event", lambda *args, **kwargs: None)

    class FakeLock:
        payload = {}

        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(app_service, "ExecutionAppLock", FakeLock)

    result = app_service.run_task_sync(task_name="inline", passthrough_args=["--flag", "x"], echo_output=False)
    metadata = json.loads((tmp_path / "job" / "metadata.json").read_text(encoding="utf-8"))

    assert result["status"] == "succeeded"
    stdout_text = (tmp_path / "job" / "stdout.log").read_text(encoding="utf-8")
    assert "inline task argv=--flag|x" in stdout_text
    assert metadata["runner_mode"] == "subprocess"
    assert metadata["server_pid"] == os.getpid()
    assert int(metadata["worker_pid"]) != os.getpid()


def test_worker_pid_is_reflected_in_runtime_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daily_research.execution import app_service

    lock_path = tmp_path / "execution_app.lock"
    state_path = tmp_path / "runtime_state.json"
    state_path.write_text(
        json.dumps(
            {
                "lock": {"job_id": "inline_job", "task_name": "inline", "pid": os.getpid()},
                "current_job": {"job_id": "inline_job", "task_name": "inline", "status": "running"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    lock_path.write_text(
        json.dumps({"job_id": "inline_job", "task_name": "inline", "pid": os.getpid()}, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(app_service, "LOCK_PATH", lock_path)
    monkeypatch.setattr(app_service, "STATE_PATH", state_path)
    monkeypatch.setattr(app_service, "EVENTS_PATH", tmp_path / "events.jsonl")
    monkeypatch.setattr(app_service, "RUNTIME_ROOT", tmp_path)

    app_service._patch_runtime_worker_payload(job_id="inline_job", worker_pid=12345)

    lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert lock_payload["pid"] == 12345
    assert lock_payload["server_pid"] == os.getpid()
    assert lock_payload["worker_pid"] == 12345
    assert state_payload["lock"]["worker_pid"] == 12345
    assert state_payload["current_job"]["worker_pid"] == 12345


def test_job_detail_marks_missing_subprocess_worker_blocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daily_research.execution import app_service

    runtime_root = tmp_path / "runtime"
    jobs_root = runtime_root / "jobs"
    job_id = "lost_subprocess_job"
    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True)
    stdout_path = job_dir / "stdout.log"
    stderr_path = job_dir / "stderr.log"
    stdout_path.write_text("stage snapshot\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    (job_dir / "metadata.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "task_name": "data-platform-refresh",
                "status": "running",
                "started_at": "2026-05-26T01:01:17+08:00",
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "runner_mode": "subprocess",
                "server_pid": 111111,
                "worker_pid": 999999,
                "artifact_paths": {},
                "evidence_paths": {},
                "runner_warnings": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    lock_path = runtime_root / "execution_app.lock"
    state_path = runtime_root / "runtime_state.json"
    lock_path.write_text(json.dumps({"job_id": job_id, "task_name": "data-platform-refresh", "worker_pid": 999999}), encoding="utf-8")
    state_path.write_text(
        json.dumps(
            {
                "lock": {"job_id": job_id, "task_name": "data-platform-refresh", "worker_pid": 999999},
                "current_job": {"job_id": job_id, "task_name": "data-platform-refresh", "status": "running"},
                "recent_jobs": [{"job_id": job_id, "task_name": "data-platform-refresh", "status": "running"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(app_service, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(app_service, "JOBS_ROOT", jobs_root)
    monkeypatch.setattr(app_service, "LOCK_PATH", lock_path)
    monkeypatch.setattr(app_service, "STATE_PATH", state_path)
    monkeypatch.setattr(app_service, "EVENTS_PATH", runtime_root / "events.jsonl")
    monkeypatch.setattr(app_service, "_pid_is_running", lambda pid: False, raising=False)

    payload = app_service.build_job_detail_payload(job_id, lines=10)

    assert payload["status"] == "blocked"
    assert payload["runner_status"] == "blocked"
    assert payload["business_status"] == "blocked"
    assert payload["can_resume"] is True
    assert any("subprocess_worker_missing" in warning for warning in payload["runner_warnings"])
    assert payload["stdout_tail"] == ["stage snapshot"]
    assert not lock_path.exists()
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert state_payload["lock"] == {}
    assert state_payload["current_job"] == {}


def test_job_detail_does_not_mark_missing_child_blocked_while_detached_worker_is_alive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from daily_research.execution import app_service

    runtime_root = tmp_path / "runtime"
    jobs_root = runtime_root / "jobs"
    job_id = "post_processing_detached_job"
    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True)
    stdout_path = job_dir / "stdout.log"
    stderr_path = job_dir / "stderr.log"
    stdout_path.write_text("task child completed\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    (job_dir / "metadata.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "task_name": "data-platform-refresh",
                "status": "running",
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "async_runner_mode": "detached_worker",
                "worker_launcher_pid": 222222,
                "runner_mode": "subprocess",
                "worker_pid": 333333,
                "runner_warnings": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    lock_path = runtime_root / "execution_app.lock"
    state_path = runtime_root / "runtime_state.json"
    lock_path.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "task_name": "data-platform-refresh",
                "worker_launcher_pid": 222222,
                "worker_pid": 333333,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    state_path.write_text(
        json.dumps(
            {
                "lock": {"job_id": job_id, "task_name": "data-platform-refresh"},
                "current_job": {"job_id": job_id, "task_name": "data-platform-refresh", "status": "running"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(app_service, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(app_service, "JOBS_ROOT", jobs_root)
    monkeypatch.setattr(app_service, "LOCK_PATH", lock_path)
    monkeypatch.setattr(app_service, "STATE_PATH", state_path)
    monkeypatch.setattr(app_service, "EVENTS_PATH", runtime_root / "events.jsonl")
    monkeypatch.setattr(app_service, "_pid_is_running", lambda pid: int(pid) == 222222, raising=False)

    payload = app_service.build_job_detail_payload(job_id, lines=10)

    assert payload["status"] == "running"
    assert payload["runner_warnings"] == []
    assert lock_path.exists()


def test_subprocess_runner_times_out_and_marks_warning(tmp_path: Path) -> None:
    from daily_research.execution import app_service

    script = tmp_path / "sleep_task.py"
    script.write_text(
        "import time\nprint('before sleep', flush=True)\ntime.sleep(5)\nprint('after sleep', flush=True)\n",
        encoding="utf-8",
    )
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text("{}", encoding="utf-8")

    class Paths:
        job_id = "timeout_job"
        metadata_path = metadata_file

    warnings_seen: list[str] = []
    with (tmp_path / "stdout.log").open("w", encoding="utf-8") as stdout_handle:
        with (tmp_path / "stderr.log").open("w", encoding="utf-8") as stderr_handle:
            exit_code = app_service._run_command_in_subprocess(
                command=[sys.executable, str(script)],
                stdout_handle=stdout_handle,
                stderr_handle=stderr_handle,
                console_stdout=None,
                console_stderr=None,
                job_paths=Paths(),
                runner_warnings=warnings_seen,
                timeout_seconds=1,
            )

    assert exit_code == app_service.RUNNER_TIMEOUT_EXIT_CODE
    assert any("runner_timeout" in item for item in warnings_seen)
    assert "before sleep" in (tmp_path / "stdout.log").read_text(encoding="utf-8")
    assert "after sleep" not in (tmp_path / "stdout.log").read_text(encoding="utf-8")


def test_subprocess_runner_streams_unflushed_python_output_before_exit(tmp_path: Path) -> None:
    from daily_research.execution import app_service

    script = tmp_path / "unflushed_sleep_task.py"
    script.write_text(
        "import time\nprint('unflushed before sleep')\ntime.sleep(5)\nprint('unflushed after sleep')\n",
        encoding="utf-8",
    )
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text("{}", encoding="utf-8")

    class Paths:
        job_id = "unflushed_timeout_job"
        metadata_path = metadata_file

    warnings_seen: list[str] = []
    with (tmp_path / "stdout.log").open("w", encoding="utf-8") as stdout_handle:
        with (tmp_path / "stderr.log").open("w", encoding="utf-8") as stderr_handle:
            exit_code = app_service._run_command_in_subprocess(
                command=[sys.executable, str(script)],
                stdout_handle=stdout_handle,
                stderr_handle=stderr_handle,
                console_stdout=None,
                console_stderr=None,
                job_paths=Paths(),
                runner_warnings=warnings_seen,
                timeout_seconds=1,
            )

    stdout_text = (tmp_path / "stdout.log").read_text(encoding="utf-8")
    assert exit_code == app_service.RUNNER_TIMEOUT_EXIT_CODE
    assert "unflushed before sleep" in stdout_text
    assert "unflushed after sleep" not in stdout_text


def test_panel_row_count_closes_file_handle(tmp_path: Path) -> None:
    from daily_research.execution import production_signal

    path = tmp_path / "panel.csv"
    path.write_text("date,score\n2026-05-22,1\n2026-05-25,2\n", encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        assert production_signal.panel_row_count(path) == 2
        gc.collect()


def test_status_payload_omits_continuous_policy() -> None:
    from daily_research.execution import app_service

    payload = app_service.build_status_payload(history_limit=1)

    assert "continuous_policy" not in payload


def test_status_payload_clears_stale_lock_for_finished_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daily_research.execution import app_service

    runtime_root = tmp_path / "runtime"
    jobs_root = runtime_root / "jobs"
    job_dir = jobs_root / "finished_job"
    job_dir.mkdir(parents=True)
    (job_dir / "metadata.json").write_text(
        json.dumps({"job_id": "finished_job", "status": "failed", "task_name": "data-platform-refresh"}, ensure_ascii=False),
        encoding="utf-8",
    )
    lock_path = runtime_root / "execution_app.lock"
    state_path = runtime_root / "runtime_state.json"
    lock_path.write_text(json.dumps({"job_id": "finished_job", "task_name": "data-platform-refresh"}), encoding="utf-8")
    state_path.write_text(
        json.dumps(
            {
                "lock": {"job_id": "finished_job", "task_name": "data-platform-refresh"},
                "current_job": {"job_id": "finished_job", "status": "running"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(app_service, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(app_service, "JOBS_ROOT", jobs_root)
    monkeypatch.setattr(app_service, "LOCK_PATH", lock_path)
    monkeypatch.setattr(app_service, "STATE_PATH", state_path)
    monkeypatch.setattr(app_service, "EVENTS_PATH", runtime_root / "events.jsonl")
    monkeypatch.setattr(app_service, "list_active_thread_job_ids", lambda: [])
    monkeypatch.setattr(app_service, "paper_account_summary", lambda: {"status": "ok"})
    monkeypatch.setattr(app_service, "positions_summary", lambda: {"exists": True, "headers_ok": True, "row_count": 0})
    monkeypatch.setattr(app_service, "active_manifest_summary", lambda: {})
    monkeypatch.setattr(app_service, "latest_trade_plan_summary", lambda **_: {"status": "missing"})
    payload = app_service.build_status_payload(history_limit=1)

    assert payload["lock"] == {}
    assert payload["current_job"] == {}
    assert not lock_path.exists()


def test_continuous_policy_api_is_removed_from_execution_console() -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    client = TestClient(web_server.create_app())

    assert client.get("/api/continuous-policy").status_code == 404
    assert client.get("/api/models/continuous-policy-shadow").status_code == 404
    assert client.get("/continuous-policy").status_code == 404


def test_runtime_json_atomic_writes_use_unique_temp_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daily_research.execution import app_runtime

    state_path = tmp_path / "runtime_state.json"
    temp_paths: list[str] = []

    def fake_replace(self: Path, target: Path) -> None:
        temp_paths.append(self.name)
        if len(temp_paths) == 1:
            app_runtime.write_json_file(state_path, {"writer": "nested"})
        target.write_text(self.read_text(encoding="utf-8"), encoding="utf-8")
        self.unlink()

    monkeypatch.setattr(Path, "replace", fake_replace)

    app_runtime.write_json_file(state_path, {"writer": "outer"})

    assert len(temp_paths) == 2
    assert temp_paths[0] != temp_paths[1]


def test_read_file_increment_reads_only_new_lines_and_survives_truncation(tmp_path: Path) -> None:
    from daily_research.execution import app_runtime

    log_path = tmp_path / "stdout.log"
    log_path.write_text("line 1\nline 2\n", encoding="utf-8")

    first = app_runtime.read_file_increment(log_path, offset=0)
    assert first["lines"] == ["line 1", "line 2"]
    assert first["offset"] > 0

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("line 3\n")

    second = app_runtime.read_file_increment(log_path, offset=int(first["offset"]))
    assert second["lines"] == ["line 3"]

    log_path.write_text("fresh\n", encoding="utf-8")
    truncated = app_runtime.read_file_increment(log_path, offset=int(second["offset"]))
    assert truncated["truncated"] is True
    assert truncated["lines"] == ["fresh"]


def test_job_stream_endpoint_emits_snapshot_incremental_stdout_and_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    stdout_path.write_text("hello stream\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    metadata = {
        "job_id": "stream-job",
        "task_name": "data-platform-refresh",
        "status": "succeeded",
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "progress": {"mode": "determinate", "stage": "done", "completed_steps": 1, "total_steps": 1, "percent": 100},
    }

    monkeypatch.setattr(web_server.app_service, "_reconcile_running_job_metadata", lambda job_id: metadata)
    monkeypatch.setattr(web_server.app_service, "load_job_metadata", lambda job_id: metadata)

    client = TestClient(web_server.create_app())
    response = client.get("/api/jobs/stream-job/stream")

    assert response.status_code == 200
    body = response.text
    assert "event: snapshot" in body
    assert '"status": "succeeded"' in body
    assert "event: stdout" in body
    assert "hello stream" in body
    assert "event: done" in body


def test_provider_health_cli_updates_job_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    from daily_research.data_platform import provider_health

    updates: list[dict[str, object]] = []

    class FakeProvider:
        name = "fake"

        def fetch_domain(self, request):
            import pandas as pd

            return type(
                "Result",
                (),
                {"data": pd.DataFrame({"symbol": ["000001.SZ"]}), "coverage_report": {}, "error_report": []},
            )()

    monkeypatch.setenv("EXECUTION_APP_JOB_ID", "provider-health-job")
    monkeypatch.setattr(provider_health, "build_default_providers", lambda provider_plan: [FakeProvider()])
    monkeypatch.setattr(provider_health, "_update_execution_job_progress", lambda **kwargs: updates.append(kwargs))

    payload = provider_health.run_provider_health(
        provider_health.ProviderHealthConfig(
            provider_plan="formal_free_v3",
            as_of_date="2026-05-22",
            domains=("market_daily", "trading_calendar"),
            symbols=("000001.SZ",),
        )
    )

    assert payload["status"] == "ok"
    assert updates
    assert updates[-1]["completed_steps"] == 2
    assert updates[-1]["total_steps"] == 2
    assert updates[-1]["current_domain"] == "trading_calendar"


def test_provider_health_progress_updates_before_each_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    from daily_research.data_platform import provider_health

    updates: list[dict[str, object]] = []

    class SlowProvider:
        name = "slow_provider"

        def fetch_domain(self, request):
            import pandas as pd

            assert updates
            assert updates[-1]["stage"] == "Checking slow_provider / market_daily"
            return type(
                "Result",
                (),
                {"data": pd.DataFrame({"symbol": ["000001.SZ"]}), "coverage_report": {}, "error_report": []},
            )()

    monkeypatch.setenv("EXECUTION_APP_JOB_ID", "provider-health-progress-job")
    monkeypatch.setattr(provider_health, "build_default_providers", lambda provider_plan: [SlowProvider()])
    monkeypatch.setattr(provider_health, "_update_execution_job_progress", lambda **kwargs: updates.append(kwargs))

    payload = provider_health.run_provider_health(
        provider_health.ProviderHealthConfig(
            provider_plan="formal_free_v3",
            as_of_date="2026-05-22",
            domains=("market_daily",),
            symbols=("000001.SZ",),
        )
    )

    assert payload["status"] == "ok"
    assert updates[0]["completed_steps"] == 0
    assert updates[0]["total_steps"] == 1
    assert updates[0]["current_item"] == "slow_provider / market_daily"
    assert updates[-1]["stage"] == "Provider health completed"


def test_ui_paths_include_react_dist() -> None:
    from daily_research.execution.web_paths import ui_paths

    payload = ui_paths()

    assert payload["react_dist"].name == "dist"
    assert payload["react_dist"].parent.name == "webapp"


def test_trade_plan_generate_preserves_windows_paths_with_spaces(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    captured: dict[str, list[str]] = {}

    def fake_launch_task_async(**kwargs):
        captured["passthrough_args"] = list(kwargs["passthrough_args"])
        return {"job_id": "job1", "task_name": kwargs["task_name"], "status": "queued"}

    monkeypatch.setattr(web_server.app_service, "launch_task_async", fake_launch_task_async)

    client = TestClient(web_server.create_app())
    response = client.post(
        "/api/trade-plan/generate",
        json={
            "candidate_profile": "active_execution_strategy",
            "positions_file": "H:/quant project/current positions.csv",
            "cash": "12345.67",
        },
    )

    assert response.status_code == 200
    assert captured["passthrough_args"] == [
        "--candidate-profile",
        "active_execution_strategy",
        "--positions-file",
        "H:/quant project/current positions.csv",
        "--cash",
        "12345.67",
    ]


def test_model_train_rejects_non_production_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    called = {"launch": False}

    def fake_launch_task_async(**kwargs):
        called["launch"] = True
        return {"job_id": "job1", "task_name": kwargs["task_name"], "status": "queued"}

    monkeypatch.setattr(web_server.app_service, "launch_task_async", fake_launch_task_async)

    client = TestClient(web_server.create_app())
    response = client.post("/api/models/path-policy-research/train", json={"dataset_mode": "latest"})

    assert response.status_code == 400
    assert called["launch"] is False


def test_production_model_train_preserves_explicit_window(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    captured: dict[str, list[str]] = {}

    def fake_launch_task_async(**kwargs):
        captured["passthrough_args"] = list(kwargs["passthrough_args"])
        return {"job_id": "job1", "task_name": kwargs["task_name"], "status": "queued"}

    monkeypatch.setattr(web_server.app_service, "launch_task_async", fake_launch_task_async)

    client = TestClient(web_server.create_app())
    response = client.post(
        "/api/models/production-full-fit/train",
        json={"dataset_mode": "custom", "start_date": "20250101", "end_date": "20260522"},
    )

    assert response.status_code == 200
    assert captured["passthrough_args"] == ["--start-date", "20250101", "--end-date", "20260522"]


def test_react_build_is_primary_ui_when_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    react_dist = tmp_path / "dist"
    react_dist.mkdir()
    (react_dist / "index.html").write_text('<div id="root">react shell</div>', encoding="utf-8")
    monkeypatch.setitem(web_server.UI_PATHS, "react_dist", react_dist)

    client = TestClient(web_server.create_app())

    assert "react shell" in client.get("/").text
    assert "react shell" in client.get("/models").text


def test_trade_plan_profile_defaults_do_not_auto_retrain(monkeypatch: pytest.MonkeyPatch) -> None:
    from daily_research.execution import research_candidate_profiles as profiles

    called = {"auto": False}

    def fake_auto(*args, **kwargs) -> None:
        called["auto"] = True

    monkeypatch.setattr(profiles, "_maybe_auto_retrain_production", fake_auto)
    monkeypatch.setattr(profiles, "_panel_is_fresh", lambda *args, **kwargs: True)

    profiles.apply_profile_defaults("active_execution_strategy", mode="trade_plan", ensure_live_panels=True)

    assert called["auto"] is False


def test_trade_plan_profile_defaults_do_not_refresh_live_panels_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from daily_research.execution import research_candidate_profiles as profiles

    called = {"refresh": False}

    def fake_refresh(*args, **kwargs) -> None:
        called["refresh"] = True
        raise AssertionError("trade-plan defaults must not refresh live panels implicitly")

    monkeypatch.setattr(profiles, "_ensure_live_panels", fake_refresh)

    profiles.apply_profile_defaults("active_execution_strategy", mode="trade_plan")

    assert called["refresh"] is False


def test_active_profile_uses_current_lake_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from daily_research.execution import research_candidate_profiles as profiles

    project_daily_research = tmp_path / "daily_research"
    production_dir = project_daily_research / "output" / "short_expert_policy_v5b_execalign_production_default"
    source_dir = project_daily_research / "output" / "short_alpha_policy_v5_family_formal_review_20260412_r1" / "runs" / "short_expert_policy_v5b"
    source_dir.mkdir(parents=True)
    production_dir.mkdir(parents=True)
    (source_dir / "execution_aligned_daily_live_target_weight_panel.csv").write_text(
        "date,stock,target_weight\n",
        encoding="utf-8",
    )
    (source_dir / "execution_aligned_daily_live_score_panel.csv").write_text(
        "date,stock,score\n",
        encoding="utf-8",
    )
    (production_dir / "execution_aligned_daily_live_target_weight_panel.csv").write_text(
        "date,stock,target_weight\n",
        encoding="utf-8",
    )
    (production_dir / "execution_aligned_daily_live_score_panel.csv").write_text(
        "date,stock,score\n",
        encoding="utf-8",
    )
    (production_dir / "production_retrain_manifest.json").write_text("{}", encoding="utf-8")
    (production_dir / "daily_live_score_reference.json").write_text("{}", encoding="utf-8")

    manifest = {
        "source_target_weight_panel_csv": str(source_dir / "execution_aligned_daily_live_target_weight_panel.csv"),
        "source_score_panel_csv": str(source_dir / "execution_aligned_daily_live_score_panel.csv"),
        "trade_plan_target_weight_panel_csv": str(
            production_dir / "execution_aligned_daily_live_target_weight_panel.csv"
        ),
        "trade_plan_score_panel_csv": str(production_dir / "execution_aligned_daily_live_score_panel.csv"),
        "production_manifest_json": str(production_dir / "production_retrain_manifest.json"),
        "score_reference_metadata_json": str(production_dir / "daily_live_score_reference.json"),
        "candidate_label": "active",
        "liquidity_pool_name": "liquid500",
        "data_source": "lake",
        "lake_dataset_id": "policy_input_bundle__x",
        "data_lake_root": str(project_daily_research / "output" / "research_data_lake"),
    }
    monkeypatch.setattr(profiles, "_DAILY_RESEARCH_ROOT", project_daily_research)
    monkeypatch.setattr(profiles, "load_strategy_manifest", lambda: manifest)

    profile = profiles.get_profile("active_execution_strategy")

    assert profile.trade_plan_target_weight_panel_csv == str(
        (production_dir / "execution_aligned_daily_live_target_weight_panel.csv").resolve()
    )
    assert profile.trade_plan_score_panel_csv == str(
        (production_dir / "execution_aligned_daily_live_score_panel.csv").resolve()
    )
    assert profile.trade_plan_model_manifest_json == str((production_dir / "production_retrain_manifest.json").resolve())
    assert profile.score_reference_metadata_json == str((production_dir / "daily_live_score_reference.json").resolve())
    assert profile.data_source == "lake"
    assert profile.lake_dataset_id == "policy_input_bundle__x"
    assert profile.data_lake_root == str(project_daily_research / "output" / "research_data_lake")


def test_model_freshness_lag_is_informational_not_blocking() -> None:
    from daily_research.baseline.generate_daily_trade_plan import _assess_model_freshness

    calendar = pd.DatetimeIndex(pd.to_datetime(["2026-05-18", "2026-05-19", "2026-05-20", "2026-05-21"]))
    payload = _assess_model_freshness(
        artifact_meta={"latest_data_date": "2026-05-18", "trained_at": "2026-05-18T20:00:00"},
        latest_signal_date=pd.Timestamp("2026-05-21"),
        trading_dates=calendar,
        warn_trading_days=1,
        max_trading_days=2,
    )

    assert payload["trading_day_lag"] == 3
    assert payload["status"] == "lagged"
    assert payload["status_text"] == "lagged"
    assert payload["should_block"] is False
    assert payload["warnings"] == []


def test_trade_plan_txt_renders_model_dates_as_information(tmp_path: Path) -> None:
    from daily_research.baseline.generate_daily_trade_plan import _write_trade_plan_txt

    path = tmp_path / "plan.txt"
    _write_trade_plan_txt(
        path,
        summary={
            "signal_date": "2026-05-22",
            "execution_date": "2026-05-25",
            "cash_input": 100000.0,
            "total_equity": 100000.0,
            "estimated_cash_after_plan": 100000.0,
            "current_position_count": 0,
            "blocked_buy_candidate_count": 0,
        },
        regime_state_row=pd.Series({"quadrant": "trend_up_low_vol", "regime_on": True}),
        action_df=pd.DataFrame(),
        hold_df=pd.DataFrame(),
        watch_df=pd.DataFrame(),
        model_info={
            "mode": "research_candidate_target_weight_csv",
            "market_regime_filter_enabled": False,
            "production_model_train_end_date": "2026-04-03",
            "production_model_launch_cutoff_date": "2026-04-21",
            "latest_completed_trading_date": "2026-05-22",
            "production_model_retrain_trading_day_lag": 20,
            "production_model_retrain_policy": "Retrain Every 21 Trading Days",
            "production_model_retrain_status": "fresh",
            "production_model_retrain_window": "2025-03-18 -> 2026-03-27",
            "freshness_status": "lagged",
            "freshness_label": "候选信号新鲜度",
            "trading_day_lag": 3,
        },
    )

    text = path.read_text(encoding="utf-8")

    assert "底层模型训练样本截止: 2026-04-03" in text
    assert "底层模型最近一次上线截止: 2026-04-21" in text
    assert "最新完成交易日: 2026-05-22" in text
    assert "底层模型距最新交易日: 20 个交易日" in text
    assert "候选信号距最新交易日: 3 个交易日" in text
    assert "lagged" not in text
    assert "fresh" not in text
    assert "候选信号间隔" not in text
    assert "重训策略" not in text
    assert "重训时效" not in text
    assert "模型新鲜度" not in text
    assert "候选信号新鲜度" not in text
    assert "提醒阈值" not in text
    assert "阻断" not in text


def _write_paper_account_snapshot(path: Path) -> None:
    path.write_text(
        "record_type,stock,shares,cost_price,available_cash\n"
        "account,,,,100000\n"
        "position,600000.SH,100,10.5,\n",
        encoding="utf-8",
    )


def test_paper_account_api_initializes_from_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    snapshot = tmp_path / "current_positions.csv"
    _write_paper_account_snapshot(snapshot)
    monkeypatch.setattr(web_server.app_service, "PAPER_ACCOUNT_DB_PATH", tmp_path / "paper.sqlite3", raising=False)
    monkeypatch.setattr(web_server.app_service, "POSITIONS_PATH", snapshot)

    client = TestClient(web_server.create_app())
    response = client.get("/api/paper-account")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["source"] == "paper_ledger"
    assert payload["available_cash"] == 100000.0
    assert payload["positions"][0]["stock"] == "600000.SH"
    assert payload["pending_order_count"] == 0


def test_paper_account_cash_flow_and_performance_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import paper_trading, web_server

    snapshot = tmp_path / "current_positions.csv"
    snapshot.write_text("record_type,stock,shares,cost_price,available_cash\naccount,,,,0\n", encoding="utf-8")
    db_path = tmp_path / "paper.sqlite3"
    monkeypatch.setattr(web_server.app_service, "PAPER_ACCOUNT_DB_PATH", db_path, raising=False)
    monkeypatch.setattr(web_server.app_service, "POSITIONS_PATH", snapshot)
    paper_trading.ensure_ledger(db_path=db_path, snapshot_path=snapshot)

    client = TestClient(web_server.create_app())
    deposit = client.post("/api/paper-account/cash-flow", json={"flow_type": "deposit", "amount": 100000, "reason": "initial"})
    assert deposit.status_code == 200
    paper_trading.write_daily_equity(db_path=db_path, price_lookup={"2026-05-22": {}}, as_of_date="2026-05-22")
    top_up = client.post("/api/paper-account/cash-flow", json={"flow_type": "deposit", "amount": 50000, "reason": "top up"})
    assert top_up.status_code == 200
    paper_trading.write_daily_equity(db_path=db_path, price_lookup={"2026-05-25": {}}, as_of_date="2026-05-25")

    response = client.get("/api/paper-account/performance?start_date=2026-05-22&end_date=2026-05-25")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["total_return"] == 0.0
    assert payload["net_cash_flow"] == 50000.0


def test_legacy_account_api_reads_paper_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    snapshot = tmp_path / "current_positions.csv"
    _write_paper_account_snapshot(snapshot)
    monkeypatch.setattr(web_server.app_service, "PAPER_ACCOUNT_DB_PATH", tmp_path / "paper.sqlite3", raising=False)
    monkeypatch.setattr(web_server.app_service, "POSITIONS_PATH", snapshot)

    client = TestClient(web_server.create_app())
    response = client.get("/api/account")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "paper_ledger"
    assert payload["available_cash"] == 100000.0


def test_trade_plan_post_process_registers_paper_orders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from daily_research.execution import app_service

    output_dir = tmp_path / "execution" / "output"
    run_dir = output_dir / "20260522"
    run_dir.mkdir(parents=True)
    (output_dir / "latest_trade_plan.txt").write_text("plan\n", encoding="utf-8")
    (run_dir / "plan_summary.json").write_text(
        json.dumps(
            {
                "signal_date": "2026-05-22",
                "execution_date": "2026-05-25",
                "candidate_label": "paper-test",
                "transaction_cost_bps": 3.0,
                "slippage_bps": 7.0,
                "sell_tax_bps": 10.0,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "actions_today.csv").write_text(
        "stock,action,shares,price,target_weight\n002866.SZ,买入,100,22,0.1\n",
        encoding="utf-8-sig",
    )
    snapshot = tmp_path / "current_positions.csv"
    snapshot.write_text("record_type,stock,shares,cost_price,available_cash\naccount,,,,100000\n", encoding="utf-8")
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(app_service, "EXECUTION_DIR", tmp_path / "execution")
    monkeypatch.setattr(app_service, "PAPER_ACCOUNT_DB_PATH", tmp_path / "paper.sqlite3", raising=False)
    monkeypatch.setattr(app_service, "POSITIONS_PATH", snapshot)

    note, metadata = app_service._post_process_successful_task(
        job_paths=SimpleNamespace(metadata_path=metadata_path),
        task_name="trade-plan",
        summary_note="trade plan done",
    )
    account = app_service.paper_account_summary()

    assert "paper_order_status=registered" in note
    assert metadata["paper_trading"]["status"] == "registered"
    assert account["pending_order_count"] == 1
