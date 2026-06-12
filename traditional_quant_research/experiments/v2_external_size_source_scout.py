"""Scout external PIT market-cap / float-cap sources for the v2.2 dataset."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/v2_external_size_source_scout")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/brain/references/research_log/2026-06-03_v2_external_size_source_scout.md")

PACKAGE_NAMES = (
    "tushare",
    "jqdatasdk",
    "rqdatac",
    "akshare",
    "efinance",
    "baostock",
    "pandas",
    "pyarrow",
)

DOCS_URLS: Mapping[str, str] = {
    "tushare_daily_basic": "https://tushare.pro/document/2?doc_id=32",
    "joinquant_get_fundamentals_valuation": "https://www.joinquant.com/help/data/stock?f=home%24m%3Dfooter",
    "rqdata_fundamental_or_factor": "https://www.ricequant.com/doc/rqdata/python/",
    "akshare_public_endpoints": "https://akshare.akfamily.xyz/data/stock/stock.html",
    "efinance_public_endpoints": "https://efinance.readthedocs.io/en/latest/index.html",
    "baostock_v2_1": "https://www.baostock.com/baostock/index.php/Python_API%E6%96%87%E6%A1%A3",
}

EXPECTED_COLUMNS = [
    "source",
    "priority_rank",
    "package_name",
    "installed",
    "auth_required",
    "auth_state",
    "pit_daily_capable_claim",
    "market_cap_fields",
    "float_cap_fields",
    "share_base_fields",
    "coverage_claim",
    "cost_or_access",
    "local_live_probe_supported",
    "local_live_probe_status",
    "usable_for_v2_2",
    "blocker",
    "recommendation",
    "evidence_level",
    "docs_url",
]


def inspect_local_packages(
    package_names: Sequence[str] = PACKAGE_NAMES,
    *,
    env: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Inspect import availability, version and relevant local auth hints."""

    env_map = os.environ if env is None else env
    rows: list[dict[str, Any]] = []
    token_by_package = {
        "tushare": _env_any(env_map, ("TUSHARE_TOKEN", "TS_TOKEN")),
        "jqdatasdk": _env_any(env_map, ("JQDATA_USERNAME", "JQDATA_PASSWORD")),
        "rqdatac": _env_any(env_map, ("RQDATAC_USERNAME", "RQDATAC_PASSWORD", "RQDATA_USERNAME", "RQDATA_PASSWORD")),
    }
    for package_name in package_names:
        installed, import_error = _is_package_installed(package_name)
        rows.append(
            {
                "package_name": package_name,
                "installed": installed,
                "version": _package_version(package_name) if installed else "",
                "auth_hint_present": bool(token_by_package.get(package_name, False)),
                "import_error": import_error,
            }
        )
    return pd.DataFrame(rows, columns=["package_name", "installed", "version", "auth_hint_present", "import_error"])


def build_external_size_source_candidates(package_capabilities: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the external source decision matrix for PIT size/float-size fields."""

    capabilities = inspect_local_packages() if package_capabilities is None else package_capabilities.copy()
    package_lookup = _package_lookup(capabilities)
    rows = [
        _candidate(
            source="tushare_daily_basic",
            priority_rank=1,
            package_name="tushare",
            package_lookup=package_lookup,
            auth_required=True,
            pit_daily_capable_claim=(
                "daily_basic is trade-date addressable and documents total/float market value plus share-base fields."
            ),
            market_cap_fields="total_mv",
            float_cap_fields="circ_mv",
            share_base_fields="total_share,float_share,free_share",
            coverage_claim="A-share daily basic indicators by trade_date or ts_code.",
            cost_or_access="Requires Tushare Pro account/token and point quota.",
            usable_rule="auth_live_probe",
            blocker_if_not_usable="Tushare package/token/live daily_basic probe has not been locally validated.",
            recommendation=(
                "First choice for v2.2: install Tushare, configure token, live-probe daily_basic for target dates, "
                "then cache total_mv/circ_mv/total_share/float_share/free_share by date/code."
            ),
            evidence_level="official_docs_pending_local_auth_probe",
        ),
        _candidate(
            source="joinquant_get_fundamentals_valuation",
            priority_rank=2,
            package_name="jqdatasdk",
            package_lookup=package_lookup,
            auth_required=True,
            pit_daily_capable_claim=(
                "get_fundamentals(..., date=YYYY-MM-DD) claims date-visible valuation data without future leakage."
            ),
            market_cap_fields="valuation.market_cap",
            float_cap_fields="valuation.circulating_market_cap",
            share_base_fields="",
            coverage_claim="A-share fundamentals/valuation via JQData subscription.",
            cost_or_access="Requires JoinQuant/JQData account and local credentials.",
            usable_rule="auth_live_probe",
            blocker_if_not_usable="JQData package/auth/live valuation probe has not been locally validated.",
            recommendation=(
                "Institutional fallback if a JoinQuant subscription exists; validate date semantics, symbol mapping, "
                "daily coverage and query limits before v2.2 ingestion."
            ),
            evidence_level="official_docs_pending_local_auth_probe",
        ),
        _candidate(
            source="rqdata_fundamental_or_factor",
            priority_rank=3,
            package_name="rqdatac",
            package_lookup=package_lookup,
            auth_required=True,
            pit_daily_capable_claim=(
                "RQData/rqdatac is a commercial quant data client likely to expose PIT fundamentals or factor data."
            ),
            market_cap_fields="market_cap,total_market_cap",
            float_cap_fields="float_market_cap,circulating_market_cap",
            share_base_fields="total_shares,float_shares",
            coverage_claim="A-share commercial data coverage; exact table names require account documentation.",
            cost_or_access="Requires Ricequant/RQData subscription and credentials.",
            usable_rule="auth_docs_and_live_probe",
            blocker_if_not_usable="No local rqdatac package/auth or table-level PIT field probe is available.",
            recommendation=(
                "Paid-data fallback; only promote after table-level docs and a live historical date probe are captured."
            ),
            evidence_level="package_known_pending_subscription_docs",
        ),
        _candidate(
            source="akshare_public_endpoints",
            priority_rank=4,
            package_name="akshare",
            package_lookup=package_lookup,
            auth_required=False,
            pit_daily_capable_claim=(
                "Public endpoints expose historical A-share OHLCV and some turnover/share fields, but PIT cap timing "
                "and revision behavior are not guaranteed by a paid data contract."
            ),
            market_cap_fields="not_promoted",
            float_cap_fields="not_promoted",
            share_base_fields="outstanding_share on selected history endpoints",
            coverage_claim="Public A-share endpoints from exchange/portal sources; endpoint coverage varies.",
            cost_or_access="Free/public; endpoint stability and timing must be audited.",
            usable_rule="not_pit_ready_by_default",
            blocker_if_not_usable="Public endpoint PIT timing, historical revision policy and full-market cap coverage are unproven.",
            recommendation=(
                "Use only as an auxiliary cross-check unless a specific endpoint proves date-bounded historical cap/share coverage."
            ),
            evidence_level="public_docs_not_pit_contract",
        ),
        _candidate(
            source="efinance_public_endpoints",
            priority_rank=5,
            package_name="efinance",
            package_lookup=package_lookup,
            auth_required=False,
            pit_daily_capable_claim=(
                "Realtime quote APIs expose current total/float market value; historical quote APIs are not yet proven "
                "to provide date-bounded PIT cap fields."
            ),
            market_cap_fields="realtime_total_market_value_only",
            float_cap_fields="realtime_float_market_value_only",
            share_base_fields="",
            coverage_claim="Public Eastmoney-derived quote/history interfaces.",
            cost_or_access="Free/public; endpoint stability and timing must be audited.",
            usable_rule="not_pit_ready_by_default",
            blocker_if_not_usable="Historical PIT total/float market-cap coverage is not proven.",
            recommendation="Do not use as v2.2 primary size source; keep for ad hoc current-quote checks only.",
            evidence_level="public_docs_realtime_not_pit",
        ),
        _candidate(
            source="baostock_v2_1",
            priority_rank=6,
            package_name="baostock",
            package_lookup=package_lookup,
            auth_required=False,
            pit_daily_capable_claim=(
                "Existing v2.1 audit proved Baostock daily history/basic fields do not expose true cap or share-base fields."
            ),
            market_cap_fields="not_available",
            float_cap_fields="not_available",
            share_base_fields="not_available",
            coverage_claim="Current project PIT universe/status/OHLCV/metrics source.",
            cost_or_access="Free login client; already integrated locally.",
            usable_rule="known_not_usable",
            blocker_if_not_usable="Baostock v2.1 lacks total_mv/circ_mv/market_cap/float_market_cap/share-base fields.",
            recommendation=(
                "Keep Baostock as PIT state and OHLCV source; do not attempt true size neutralization from Baostock alone."
            ),
            evidence_level="local_live_audit_complete",
        ),
    ]
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def summarize_candidates(candidates: pd.DataFrame, package_capabilities: pd.DataFrame, *, run_id: str) -> dict[str, Any]:
    """Summarize the source matrix into an actionable v2.2 decision record."""

    ordered = candidates.sort_values("priority_rank")
    usable = ordered.loc[ordered["usable_for_v2_2"]]
    pending_auth = ordered.loc[
        ordered["auth_required"]
        & ordered["installed"]
        & ordered["auth_state"].isin(["auth_hint_present", "auth_missing_or_unknown"])
    ]
    best = "tushare_daily_basic"
    best_row = ordered.loc[ordered["source"] == best].iloc[0].to_dict()
    best_installed = bool(best_row["installed"])
    installed_packages = (
        package_capabilities.loc[package_capabilities["installed"], "package_name"].astype(str).sort_values().tolist()
    )
    return {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_count": int(len(candidates)),
        "usable_now_count": int(len(usable)),
        "installed_packages": installed_packages,
        "best_candidate": best,
        "best_candidate_installed": best_installed,
        "best_candidate_auth_state": str(best_row["auth_state"]),
        "recommended_path": (
            "Use Tushare daily_basic as the first v2.2 validation target; if token/auth is unavailable, "
            "evaluate JoinQuant or RQData subscriptions; keep AkShare/efinance as auxiliary public checks only."
        ),
        "v2_2_ready": bool(len(usable) > 0),
        "primary_blocker": (
            "No locally authenticated and live-probed PIT daily market-cap/float-cap source is currently validated."
            if len(usable) == 0
            else ""
        ),
        "pending_auth_sources": pending_auth["source"].astype(str).tolist(),
        "not_pit_ready_public_sources": ordered.loc[
            ordered["source"].isin(["akshare_public_endpoints", "efinance_public_endpoints"]), "source"
        ]
        .astype(str)
        .tolist(),
        "baostock_usable_for_size": bool(
            ordered.loc[ordered["source"] == "baostock_v2_1", "usable_for_v2_2"].iloc[0]
        ),
        "next_actions": [
            (
                "Configure TUSHARE_TOKEN or TS_TOKEN and run a small historical daily_basic live probe for 600000.SH/000001.SZ."
                if best_installed
                else "Install/configure Tushare and run a small historical daily_basic live probe for 600000.SH/000001.SZ."
            ),
            "Verify symbol mapping, trade_date coverage, units and missingness against the existing v2.1 universe.",
            "If Tushare is unavailable, request JoinQuant or RQData credentials and repeat the same two-symbol probe.",
            "Until then, keep size controls proxy-only and do not promote candidate-frontier strategies.",
        ],
    }


def run_v2_external_size_source_scout(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    package_capabilities: pd.DataFrame | None = None,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    """Run the source scout and write reproducible artifacts."""

    run_id = f"v2_external_size_source_scout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    capabilities = inspect_local_packages() if package_capabilities is None else package_capabilities.copy()
    candidates = build_external_size_source_candidates(capabilities)
    summary = summarize_candidates(candidates, capabilities, run_id=run_id)
    summary_md = render_summary_markdown(summary, candidates, capabilities)

    capabilities.to_csv(run_dir / "local_package_capabilities.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(run_dir / "external_size_source_candidates.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    if write_research_log:
        path = Path(research_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(summary_md, encoding="utf-8")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "candidate_count": summary["candidate_count"],
        "usable_now_count": summary["usable_now_count"],
        "best_candidate": summary["best_candidate"],
        "v2_2_ready": summary["v2_2_ready"],
        "primary_blocker": summary["primary_blocker"],
    }


def render_summary_markdown(
    summary: Mapping[str, Any],
    candidates: pd.DataFrame,
    package_capabilities: pd.DataFrame,
) -> str:
    """Render a compact research-log style summary."""

    lines = [
        "# V2 External Size Source Scout",
        "",
        "## Summary",
        "",
        f"- `run_id`: `{summary['run_id']}`",
        f"- `best_candidate`: `{summary['best_candidate']}`",
        f"- `v2_2_ready`: `{summary['v2_2_ready']}`",
        f"- `usable_now_count`: `{summary['usable_now_count']}`",
        f"- `primary_blocker`: {summary['primary_blocker'] or 'none'}",
        "",
        "## Local Packages",
        "",
    ]
    for row in package_capabilities.to_dict("records"):
        lines.append(
            f"- `{row['package_name']}`: installed=`{bool(row['installed'])}`, "
            f"version=`{row.get('version', '')}`, auth_hint=`{bool(row.get('auth_hint_present', False))}`"
        )
    lines.extend(["", "## Candidate Matrix", ""])
    for row in candidates.sort_values("priority_rank").to_dict("records"):
        lines.append(
            f"- `{row['source']}`: installed=`{bool(row['installed'])}`, auth=`{row['auth_state']}`, "
            f"usable_for_v2_2=`{bool(row['usable_for_v2_2'])}`; blocker: {row['blocker']}"
        )
    lines.extend(["", "## Decision", ""])
    lines.append(str(summary["recommended_path"]))
    lines.extend(["", "## Next Actions", ""])
    for action in summary["next_actions"]:
        lines.append(f"- {action}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a data-source gate, not a strategy promotion. The current frontier remains "
            "`candidate-frontier/backtest_only` until true size/float-size data or a stricter proxy-only policy is validated.",
            "",
        ]
    )
    return "\n".join(lines)


def _candidate(
    *,
    source: str,
    priority_rank: int,
    package_name: str,
    package_lookup: Mapping[str, Mapping[str, Any]],
    auth_required: bool,
    pit_daily_capable_claim: str,
    market_cap_fields: str,
    float_cap_fields: str,
    share_base_fields: str,
    coverage_claim: str,
    cost_or_access: str,
    usable_rule: str,
    blocker_if_not_usable: str,
    recommendation: str,
    evidence_level: str,
) -> dict[str, Any]:
    package = package_lookup.get(package_name, {})
    installed = bool(package.get("installed", False))
    auth_hint = bool(package.get("auth_hint_present", False))
    auth_state = _auth_state(auth_required=auth_required, installed=installed, auth_hint=auth_hint)
    live_probe_supported = _live_probe_supported(installed=installed, auth_required=auth_required, auth_hint=auth_hint)
    usable = _usable_for_v2_2(
        usable_rule=usable_rule,
        installed=installed,
        auth_required=auth_required,
        auth_hint=auth_hint,
        live_probe_supported=live_probe_supported,
    )
    return {
        "source": source,
        "priority_rank": priority_rank,
        "package_name": package_name,
        "installed": installed,
        "auth_required": auth_required,
        "auth_state": auth_state,
        "pit_daily_capable_claim": pit_daily_capable_claim,
        "market_cap_fields": market_cap_fields,
        "float_cap_fields": float_cap_fields,
        "share_base_fields": share_base_fields,
        "coverage_claim": coverage_claim,
        "cost_or_access": cost_or_access,
        "local_live_probe_supported": live_probe_supported,
        "local_live_probe_status": "not_run",
        "usable_for_v2_2": usable,
        "blocker": "" if usable else blocker_if_not_usable,
        "recommendation": recommendation,
        "evidence_level": evidence_level,
        "docs_url": DOCS_URLS[source],
    }


def _usable_for_v2_2(
    *,
    usable_rule: str,
    installed: bool,
    auth_required: bool,
    auth_hint: bool,
    live_probe_supported: bool,
) -> bool:
    if usable_rule in {"known_not_usable", "not_pit_ready_by_default"}:
        return False
    if usable_rule in {"auth_live_probe", "auth_docs_and_live_probe"}:
        # A credential hint alone is not enough; a successful live probe must be added before promotion.
        return False
    if auth_required:
        return bool(installed and auth_hint and live_probe_supported)
    return bool(installed and live_probe_supported)


def _auth_state(*, auth_required: bool, installed: bool, auth_hint: bool) -> str:
    if not auth_required:
        return "not_required"
    if not installed:
        return "package_missing"
    if auth_hint:
        return "auth_hint_present"
    return "auth_missing_or_unknown"


def _live_probe_supported(*, installed: bool, auth_required: bool, auth_hint: bool) -> bool:
    if not installed:
        return False
    if auth_required and not auth_hint:
        return False
    return True


def _package_lookup(capabilities: pd.DataFrame) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    if capabilities.empty:
        return output
    for row in capabilities.to_dict("records"):
        output[str(row.get("package_name", ""))] = row
    return output


def _env_any(env: Mapping[str, str], names: Sequence[str]) -> bool:
    return any(bool(env.get(name)) for name in names)


def _is_package_installed(package_name: str) -> tuple[bool, str]:
    try:
        module = importlib.util.find_spec(package_name)
        return module is not None, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except Exception:  # noqa: BLE001
        try:
            module = importlib.import_module(package_name)
            return str(getattr(module, "__version__", ""))
        except Exception:  # noqa: BLE001
            return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", default=str(DEFAULT_RESEARCH_LOG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_v2_external_size_source_scout(
        output_dir=Path(args.output_dir),
        write_research_log=bool(args.write_research_log),
        research_log_path=Path(args.research_log_path),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
