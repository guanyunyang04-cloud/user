from __future__ import annotations

from pathlib import Path

import pandas as pd

from traditional_quant_research.experiments import v2_external_size_source_scout as scout


def _capabilities(**overrides: dict[str, object]) -> pd.DataFrame:
    rows = []
    for package_name in scout.PACKAGE_NAMES:
        row = {
            "package_name": package_name,
            "installed": False,
            "version": "",
            "auth_hint_present": False,
            "import_error": "",
        }
        row.update(overrides.get(package_name, {}))
        rows.append(row)
    return pd.DataFrame(rows)


def test_candidates_include_tushare_as_best_pending_auth() -> None:
    capabilities = _capabilities(
        tushare={"installed": False, "version": "", "auth_hint_present": False},
        baostock={"installed": True, "version": "fixture", "auth_hint_present": False},
    )

    candidates = scout.build_external_size_source_candidates(capabilities)
    summary = scout.summarize_candidates(candidates, capabilities, run_id="fixture")

    tushare = candidates.loc[candidates["source"] == "tushare_daily_basic"].iloc[0]
    assert summary["best_candidate"] == "tushare_daily_basic"
    assert bool(tushare["usable_for_v2_2"]) is False
    assert tushare["auth_state"] == "package_missing"
    assert "total_mv" in tushare["market_cap_fields"]
    assert "circ_mv" in tushare["float_cap_fields"]
    assert "Tushare" in tushare["recommendation"]
    assert summary["v2_2_ready"] is False


def test_baostock_is_explicitly_not_usable_for_size() -> None:
    capabilities = _capabilities(baostock={"installed": True, "version": "1.0.0"})

    candidates = scout.build_external_size_source_candidates(capabilities)
    baostock = candidates.loc[candidates["source"] == "baostock_v2_1"].iloc[0]

    assert bool(baostock["installed"]) is True
    assert bool(baostock["usable_for_v2_2"]) is False
    assert "lacks" in baostock["blocker"]
    assert "Baostock" in baostock["recommendation"]


def test_public_sources_are_captured_but_not_pit_ready_by_default() -> None:
    capabilities = _capabilities(
        akshare={"installed": True, "version": "fixture"},
        efinance={"installed": True, "version": "fixture"},
    )

    candidates = scout.build_external_size_source_candidates(capabilities)
    public = candidates.loc[candidates["source"].isin(["akshare_public_endpoints", "efinance_public_endpoints"])]

    assert public["installed"].eq(True).all()
    assert public["usable_for_v2_2"].eq(False).all()
    assert public["auth_required"].eq(False).all()
    assert public["blocker"].str.contains("PIT|Historical", regex=True).all()


def test_auth_hint_does_not_promote_without_live_probe() -> None:
    capabilities = _capabilities(
        tushare={"installed": True, "version": "1.2.3", "auth_hint_present": True},
    )

    candidates = scout.build_external_size_source_candidates(capabilities)
    summary = scout.summarize_candidates(candidates, capabilities, run_id="fixture")
    tushare = candidates.loc[candidates["source"] == "tushare_daily_basic"].iloc[0]

    assert tushare["auth_state"] == "auth_hint_present"
    assert bool(tushare["local_live_probe_supported"]) is True
    assert bool(tushare["usable_for_v2_2"]) is False
    assert summary["pending_auth_sources"] == ["tushare_daily_basic"]


def test_installed_tushare_without_token_is_pending_auth() -> None:
    capabilities = _capabilities(
        tushare={"installed": True, "version": "1.4.29", "auth_hint_present": False},
    )

    candidates = scout.build_external_size_source_candidates(capabilities)
    summary = scout.summarize_candidates(candidates, capabilities, run_id="fixture")
    tushare = candidates.loc[candidates["source"] == "tushare_daily_basic"].iloc[0]

    assert tushare["auth_state"] == "auth_missing_or_unknown"
    assert summary["pending_auth_sources"] == ["tushare_daily_basic"]
    assert summary["next_actions"][0].startswith("Configure TUSHARE_TOKEN")


def test_run_scout_writes_artifacts_and_research_log(tmp_path: Path) -> None:
    capabilities = _capabilities(
        akshare={"installed": True, "version": "fixture"},
        efinance={"installed": True, "version": "fixture"},
        baostock={"installed": True, "version": "fixture"},
        pandas={"installed": True, "version": "fixture"},
        pyarrow={"installed": True, "version": "fixture"},
    )

    result = scout.run_v2_external_size_source_scout(
        output_dir=tmp_path / "output",
        package_capabilities=capabilities,
        write_research_log=True,
        research_log_path=tmp_path / "research_log.md",
    )

    run_dir = Path(result["run_dir"])
    assert result["best_candidate"] == "tushare_daily_basic"
    assert result["v2_2_ready"] is False
    assert (run_dir / "local_package_capabilities.csv").exists()
    assert (run_dir / "external_size_source_candidates.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()
    assert (tmp_path / "research_log.md").exists()

    candidates = pd.read_csv(run_dir / "external_size_source_candidates.csv")
    assert set(candidates["source"]) == {
        "tushare_daily_basic",
        "joinquant_get_fundamentals_valuation",
        "rqdata_fundamental_or_factor",
        "akshare_public_endpoints",
        "efinance_public_endpoints",
        "baostock_v2_1",
    }
