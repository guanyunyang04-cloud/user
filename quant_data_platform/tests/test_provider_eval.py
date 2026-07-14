from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from quant_data_platform.provider_eval import (
    ProbeResult,
    ProviderEvalConfig,
    build_cross_source_ohlcv_diff,
    build_field_coverage,
    build_latency_summary,
    choose_unit_factor,
    ensure_provider_dependency,
    run_provider_eval,
)


def _market_frame(provider: str = "fake") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "trade_date": ["2024-06-03", "2024-06-04"],
            "open": [10.0, 10.2],
            "high": [10.5, 10.4],
            "low": [9.9, 10.1],
            "close": [10.3, 10.2],
            "volume": [1000.0, 1100.0],
            "amount": [10000.0, 11220.0],
            "source": [provider, provider],
            "adjusted_flag": ["none", "none"],
            "eval_provider": [provider, provider],
            "endpoint": ["market_daily_none", "market_daily_none"],
        }
    )


def test_provider_eval_fake_provider_success(tmp_path: Path) -> None:
    def fake_probe(config: ProviderEvalConfig, provider: str, cache_dir: Path):
        del config, cache_dir
        frame = _market_frame(provider)
        result = ProbeResult(
            provider=provider,
            endpoint="market_daily_none",
            domain="market_daily",
            symbol_count=1,
            start_date="2024-06-03",
            end_date="2024-06-04",
            adjusted_flag="none",
            ok=True,
            elapsed_sec=0.01,
            rows=len(frame),
            columns=tuple(frame.columns),
            field_finite_rates={"close": 1.0},
            normalized_sample=frame.head(1).to_dict(orient="records"),
        )
        return [result], [frame]

    report = run_provider_eval(
        ProviderEvalConfig(
            providers=("fake",),
            symbols=("000001.SZ",),
            windows=(("2024-06-03", "2024-06-04"),),
            run_tag="unit_fake",
            output_root=tmp_path,
        ),
        probe_overrides={"fake": fake_probe},
    )

    run_dir = tmp_path / "unit_fake"
    assert report["status"] == "ok"
    assert (run_dir / "provider_eval_report.json").exists()
    assert (run_dir / "endpoint_results.csv").exists()
    assert (run_dir / "normalized_market_daily.csv").exists()
    assert (run_dir / "field_coverage.csv").exists()
    normalized = pd.read_csv(run_dir / "normalized_market_daily.csv")
    assert len(normalized) == 2
    assert set(normalized["eval_provider"]) == {"fake"}


def test_provider_eval_dependency_missing() -> None:
    status = ensure_provider_dependency(
        "definitely_missing_provider_eval_module",
        install_missing=False,
        python_executable="python",
    )
    assert status["import_ok"] is False
    assert status["status"] == "missing"


def test_provider_eval_install_dependency_plan_for_mootdx(monkeypatch) -> None:
    calls = []

    def fake_import(name: str):
        if name == "mootdx":
            raise ModuleNotFoundError("missing mootdx")
        raise AssertionError(name)

    def fake_runner(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args=args[0], returncode=1, stdout="", stderr="install failed")

    monkeypatch.setattr("quant_data_platform.provider_eval.importlib.import_module", fake_import)
    status = ensure_provider_dependency("mootdx", install_missing=True, python_executable="py", runner=fake_runner)

    assert calls
    assert status["install_ok"] is False
    assert status["status"] == "install_failed"
    assert "mootdx[all]" in status["install_command"]


def test_provider_eval_latency_summary() -> None:
    rows = pd.DataFrame(
        [
            {"provider": "a", "endpoint": "x", "ok": True, "elapsed_sec": 1.0},
            {"provider": "a", "endpoint": "x", "ok": False, "elapsed_sec": 3.0},
            {"provider": "b", "endpoint": "y", "ok": True, "elapsed_sec": 2.0},
        ]
    )
    summary = build_latency_summary(rows)
    a_row = summary.loc[summary["provider"].eq("a")].iloc[0]
    assert a_row["call_count"] == 2
    assert a_row["ok_count"] == 1
    assert a_row["p50_latency"] == 2.0


def test_provider_eval_field_coverage() -> None:
    frame = _market_frame()
    frame.loc[1, "amount"] = None
    coverage = build_field_coverage(frame)
    amount = coverage.loc[coverage["field"].eq("amount")].iloc[0]
    assert amount["rows"] == 2
    assert amount["finite_rows"] == 1
    assert amount["finite_rate"] == 0.5


def test_provider_eval_unit_factor_detection() -> None:
    factor = choose_unit_factor([1.0, 2.0, 3.0], [10000.0, 20000.0, 30000.0])
    assert factor == 10000.0


def test_provider_eval_cross_source_diff() -> None:
    qdp = _market_frame("current_qdp")
    other = _market_frame("akshare")
    other["volume"] = other["volume"] / 100.0
    frame = pd.concat([qdp, other], ignore_index=True)
    diff = build_cross_source_ohlcv_diff(frame)
    assert not diff.empty
    volume = diff.loc[diff["field"].eq("volume")].iloc[0]
    assert volume["suggested_unit_factor"] == 100.0


def test_provider_eval_proxy_retry_records_clean_mode(tmp_path: Path) -> None:
    attempts = []

    def fake_probe(config: ProviderEvalConfig, provider: str, cache_dir: Path):
        from quant_data_platform.provider_eval import _cached_or_call

        def call(network_mode: str):
            attempts.append(network_mode)
            if network_mode == "current_env":
                return (
                    ProbeResult(
                        provider=provider,
                        endpoint="x",
                        ok=False,
                        error_type="ProxyError",
                        error_message="proxy blocked",
                    ),
                    pd.DataFrame(),
                )
            frame = _market_frame(provider)
            return (
                ProbeResult(
                    provider=provider,
                    endpoint="x",
                    ok=True,
                    network_mode=network_mode,
                    rows=len(frame),
                    elapsed_sec=0.01,
                ),
                frame,
            )

        results, frame = _cached_or_call(config, cache_dir=cache_dir, provider=provider, endpoint="x", params={}, call=call)
        return results, [frame]

    run_provider_eval(
        ProviderEvalConfig(providers=("fake",), run_tag="proxy_retry", output_root=tmp_path, use_cache=False),
        probe_overrides={"fake": fake_probe},
    )

    endpoint = pd.read_csv(tmp_path / "proxy_retry" / "endpoint_results.csv")
    assert attempts == ["current_env", "clean_proxy_env"]
    assert list(endpoint["network_mode"]) == ["current_env", "clean_proxy_env"]
    assert bool(endpoint.loc[0, "ok"]) is False
    assert bool(endpoint.loc[1, "ok"]) is True


def test_provider_eval_does_not_touch_registry_or_canonical(tmp_path: Path) -> None:
    registry = tmp_path / "registry" / "root_manifest.json"
    canonical = tmp_path / "canonical" / "canonical_manifest.json"
    registry.parent.mkdir(parents=True)
    canonical.parent.mkdir(parents=True)
    registry.write_text(json.dumps({"canonical_dataset_id": "unchanged"}), encoding="utf-8")
    canonical.write_text(json.dumps({"canonical_dataset_id": "unchanged"}), encoding="utf-8")

    def fake_probe(config: ProviderEvalConfig, provider: str, cache_dir: Path):
        del config, provider, cache_dir
        return [ProbeResult(provider="fake", endpoint="dependency", ok=True)], []

    before_registry = registry.read_text(encoding="utf-8")
    before_canonical = canonical.read_text(encoding="utf-8")
    run_provider_eval(
        ProviderEvalConfig(
            providers=("fake",),
            run_tag="readonly",
            output_root=tmp_path / "provider_eval",
            qdp_root_manifest=registry,
            canonical_manifest=canonical,
        ),
        probe_overrides={"fake": fake_probe},
    )

    assert registry.read_text(encoding="utf-8") == before_registry
    assert canonical.read_text(encoding="utf-8") == before_canonical


def test_qdp_cli_provider_eval_is_archived(monkeypatch, tmp_path: Path, capsys) -> None:
    from quant_data_platform import cli as qdp_cli

    captured = {}

    def fake_run_provider_eval(config: ProviderEvalConfig):
        captured["config"] = config
        return {"status": "ok", "summary": {"endpoint_count": 0}}

    monkeypatch.setattr("quant_data_platform.provider_eval.run_provider_eval", fake_run_provider_eval)
    rc = qdp_cli.main(
        [
            "--workspace-root",
            str(Path.cwd()),
            "provider-eval",
            "--providers",
            "current_qdp",
            "--symbols",
            "000001.SZ",
            "--windows",
            "2024-06-03:2024-06-07",
            "--run-tag",
            "unit_cli",
            "--output-root",
            str(tmp_path),
            "--json",
        ]
    )

    assert rc == 2
    assert captured == {}
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "archived"
    assert payload["command"] == "provider-eval"
