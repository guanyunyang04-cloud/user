from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


FORMAL_REFRESH_DOMAINS = [
    "market_daily",
    "trading_calendar",
    "universe_snapshot",
    "security_status",
    "limit_status",
    "industry_concept",
    "valuation",
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


def test_models_payload_includes_core_model_roles() -> None:
    from daily_research.execution import app_service

    payload = app_service.models_summary()
    roles = {item["role"] for item in payload["models"]}

    assert payload["status"] == "ok"
    assert {"live", "production", "path_policy", "legacy"} <= roles
    assert "continuous_policy" not in roles
    assert not any(item["id"] == "continuous-policy-shadow" for item in payload["models"])
    assert any(item["is_live"] for item in payload["models"])


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
        "provider_plan": "default_free",
    }


def test_data_refresh_api_defaults_to_completed_date_and_formal_domains(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    captured: dict[str, list[str]] = {}

    def fake_launch_task_async(**kwargs):
        captured["passthrough_args"] = list(kwargs["passthrough_args"])
        return {"job_id": "job1", "task_name": kwargs["task_name"], "status": "queued"}

    monkeypatch.setattr(web_server.app_service, "get_latest_completed_trading_date", lambda: "2026-05-22", raising=False)
    monkeypatch.setattr(web_server.app_service, "launch_task_async", fake_launch_task_async)

    client = TestClient(web_server.create_app())
    response = client.post("/api/data-sources/refresh", json={})

    assert response.status_code == 200
    assert captured["passthrough_args"] == [
        "--as-of-date",
        "2026-05-22",
        "--universe",
        "all_a",
        "--domains",
        ",".join(FORMAL_REFRESH_DOMAINS),
    ]


def test_status_payload_omits_continuous_policy() -> None:
    from daily_research.execution import app_service

    payload = app_service.build_status_payload(history_limit=1)

    assert "continuous_policy" not in payload


def test_continuous_policy_api_is_removed_from_execution_console() -> None:
    from fastapi.testclient import TestClient

    from daily_research.execution import web_server

    client = TestClient(web_server.create_app())

    assert client.get("/api/continuous-policy").status_code == 404
    assert client.get("/api/models/continuous-policy-shadow").status_code == 404
    assert client.get("/continuous-policy").status_code == 404


def test_ui_paths_include_react_dist() -> None:
    from daily_research.execution.web_service import ui_paths

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
