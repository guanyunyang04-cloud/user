"""Audit whether the v2 PIT snapshot can support true industry/size controls."""

from __future__ import annotations

import argparse
import importlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from traditional_quant_research.dataset_v2 import DEFAULT_V2_SNAPSHOT_ROOT


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/v2_industry_size_source_audit")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-02_v2_industry_size_source_audit.md")

SNAPSHOT_TABLE_FILES: Mapping[str, str] = {
    "security_master": "security_master.parquet",
    "stock_industry": "stock_industry.parquet",
    "daily_metrics": "daily_metrics.parquet",
    "daily_universe": "daily_universe.parquet",
    "daily_bars": "daily_bars.parquet",
    "daily_status": "daily_status.parquet",
}

TRUE_INDUSTRY_FIELDS = (
    "industry",
    "industry_code",
    "industry_name",
    "industry_level1",
    "industry_level2",
    "industryclassification",
    "sw_industry",
    "cs_industry",
    "citic_industry",
    "baostock_industry",
)
TRUE_MARKET_CAP_FIELDS = (
    "market_cap",
    "marketcap",
    "float_market_cap",
    "free_float_market_cap",
    "total_market_cap",
    "total_mv",
    "float_mv",
    "circ_mv",
    "market_value",
    "total_market_value",
    "circulating_market_value",
    "totalMarketValue",
    "circulatingMarketValue",
)
SHARE_BASE_FIELDS = (
    "total_share",
    "totalShare",
    "float_share",
    "tradableShare",
    "free_float_share",
    "liqa_share",
    "liqaShare",
    "nonrestfloata",
)
TURNOVER_FIELDS = (
    "turn",
    "turnover",
    "turnover_rate",
    "turnrate",
)
SIZE_LIQUIDITY_PROXY_FIELDS = (
    "amount",
    "volume",
    "log_amount_mean_20d",
    "log_amount_mean_20d_z",
)
PIT_STATUS_FIELDS = (
    "ipo_date",
    "out_date",
    "security_type",
    "status",
    "is_st_on_date",
    "is_suspended_on_date",
    "is_tradeable",
    "tradestatus",
    "isst",
)

BAOSTOCK_METHODS_OF_INTEREST = (
    "query_stock_industry",
    "query_history_k_data_plus",
    "query_stock_basic",
    "query_profit_data",
    "query_operation_data",
    "query_growth_data",
    "query_balance_data",
    "query_cash_flow_data",
    "query_dupont_data",
    "query_performance_express_report",
)
BAOSTOCK_HISTORY_PROBE_FIELDS = (
    "turn",
    "turnover",
    "turnover_rate",
    "pctChg",
    "peTTM",
    "pbMRQ",
    "psTTM",
    "pcfNcfTTM",
    "totalShare",
    "liqaShare",
    "total_mv",
    "float_mv",
    "market_cap",
    "float_market_cap",
)
BAOSTOCK_STOCK_BASIC_PROBE_FIELDS = (
    "totalShare",
    "liqaShare",
    "tradableShare",
    "total_mv",
    "float_mv",
    "market_cap",
    "float_market_cap",
)


def resolve_snapshot_root(root: str | Path | None = None) -> Path:
    """Resolve an explicit snapshot root or the default latest v2 snapshot."""

    base = Path(root) if root is not None else DEFAULT_V2_SNAPSHOT_ROOT
    latest = base / "latest_manifest.json"
    if latest.exists():
        payload = json.loads(latest.read_text(encoding="utf-8-sig"))
        snapshot_path = payload.get("snapshot_path")
        if snapshot_path:
            return Path(snapshot_path)
    return base


def read_manifest(snapshot_root: Path) -> dict[str, Any]:
    path = snapshot_root / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_snapshot_schema(snapshot_root: Path) -> pd.DataFrame:
    """Return table, field and dtype rows without loading full parquet bodies."""

    rows: list[dict[str, Any]] = []
    for table, file_name in SNAPSHOT_TABLE_FILES.items():
        path = snapshot_root / file_name
        if not path.exists():
            rows.append({"table": table, "field": "", "dtype": "", "exists": False, "path": str(path)})
            continue
        for field, dtype in _read_parquet_schema(path):
            rows.append({"table": table, "field": field, "dtype": dtype, "exists": True, "path": str(path)})
    rows.extend(_tradeable_panel_schema_rows(rows))
    return pd.DataFrame(rows, columns=["table", "field", "dtype", "exists", "path"])


def build_source_field_audit(
    schema: pd.DataFrame,
    *,
    baostock_probe: pd.DataFrame | None = None,
    history_field_probe: pd.DataFrame | None = None,
    stock_basic_field_probe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assess current snapshot fields plus optional Baostock client capabilities."""

    table_fields = _table_field_map(schema)
    rows: list[dict[str, Any]] = []
    rows.extend(
        _field_group_rows(
            "true_industry",
            "Needed for real industry neutralization.",
            TRUE_INDUSTRY_FIELDS,
            table_fields,
            missing_note="No true industry field is present in the current v2 snapshot.",
        )
    )
    rows.extend(
        _field_group_rows(
            "true_market_cap",
            "Needed for real size or market-cap neutralization.",
            TRUE_MARKET_CAP_FIELDS,
            table_fields,
            missing_note="No true market-cap or float market-cap field is present in the current v2 snapshot.",
        )
    )
    rows.extend(
        _field_group_rows(
            "share_base",
            "Could help derive market cap if a vetted price/share formula is adopted.",
            SHARE_BASE_FIELDS,
            table_fields,
            missing_note="No share-base field is present in the current v2 snapshot.",
        )
    )
    rows.extend(
        _field_group_rows(
            "turnover",
            "Could strengthen liquidity and size proxy controls if sourced reliably.",
            TURNOVER_FIELDS,
            table_fields,
            missing_note="No turnover field is present in the current v2 snapshot.",
        )
    )
    rows.extend(
        _field_group_rows(
            "size_liquidity_proxy",
            "Proxy only; useful for liquidity/size diagnostics, not a true market-cap control.",
            SIZE_LIQUIDITY_PROXY_FIELDS,
            table_fields,
            missing_note="No size/liquidity proxy field is present.",
        )
    )
    rows.extend(
        _field_group_rows(
            "pit_status",
            "Confirms the PIT state fields needed for tradeable-universe construction.",
            PIT_STATUS_FIELDS,
            table_fields,
            missing_note="Some PIT state fields are missing.",
        )
    )
    if baostock_probe is not None and not baostock_probe.empty:
        for row in baostock_probe.to_dict("records"):
            rows.append(
                {
                    "domain": "baostock_client",
                    "requirement": str(row.get("requirement", "")),
                    "evidence_source": "baostock_client",
                    "field_or_method": str(row.get("field_or_method", "")),
                    "available": bool(row.get("available", False)),
                    "evidence": str(row.get("evidence", "")),
                    "notes": str(row.get("notes", "")),
                }
            )
    if history_field_probe is not None and not history_field_probe.empty:
        for row in history_field_probe.to_dict("records"):
            rows.append(
                {
                    "domain": "baostock_history_field",
                    "requirement": _history_field_requirement(str(row.get("field_or_method", ""))),
                    "evidence_source": "baostock_live_history_probe",
                    "field_or_method": str(row.get("field_or_method", "")),
                    "available": bool(row.get("available", False)),
                    "evidence": str(row.get("evidence", "")),
                    "notes": str(row.get("notes", "")),
                }
            )
    if stock_basic_field_probe is not None and not stock_basic_field_probe.empty:
        for row in stock_basic_field_probe.to_dict("records"):
            rows.append(
                {
                    "domain": "baostock_stock_basic_field",
                    "requirement": _stock_basic_field_requirement(str(row.get("field_or_method", ""))),
                    "evidence_source": "baostock_live_stock_basic_probe",
                    "field_or_method": str(row.get("field_or_method", "")),
                    "available": bool(row.get("available", False)),
                    "evidence": str(row.get("evidence", "")),
                    "notes": str(row.get("notes", "")),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["domain", "requirement", "evidence_source", "field_or_method", "available", "evidence", "notes"],
    )


def probe_baostock_client(client: Any | None = None) -> pd.DataFrame:
    """Inspect local Baostock client methods without logging in or issuing network requests."""

    rows: list[dict[str, Any]] = []
    import_error = ""
    if client is None:
        try:
            client = importlib.import_module("baostock")
        except Exception as exc:  # noqa: BLE001
            import_error = str(exc)
            client = None
    version = str(getattr(client, "__version__", "")) if client is not None else ""
    for method in BAOSTOCK_METHODS_OF_INTEREST:
        available = client is not None and hasattr(client, method)
        rows.append(
            {
                "requirement": _baostock_method_requirement(method),
                "field_or_method": method,
                "available": bool(available),
                "evidence": f"baostock_version={version}" if version else "",
                "notes": "" if available else (f"import_error={import_error}" if import_error else "method not found"),
            }
        )
    return pd.DataFrame(rows, columns=["requirement", "field_or_method", "available", "evidence", "notes"])


def probe_baostock_history_fields(
    *,
    code: str = "sh.600000",
    start_date: str = "2026-05-25",
    end_date: str = "2026-06-01",
    fields: Sequence[str] = BAOSTOCK_HISTORY_PROBE_FIELDS,
) -> pd.DataFrame:
    """Live-probe Baostock daily history fields one by one."""

    rows: list[dict[str, Any]] = []
    try:
        import baostock as bs  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        for field in fields:
            rows.append(
                {
                    "field_or_method": field,
                    "available": False,
                    "error_code": "",
                    "error_msg": "",
                    "returned_fields": "",
                    "row_count": 0,
                    "sample": "",
                    "evidence": "",
                    "notes": f"import_error={exc}",
                }
            )
        return pd.DataFrame(rows)

    login = bs.login()
    login_ok = str(login.error_code) == "0"
    try:
        if not login_ok:
            for field in fields:
                rows.append(
                    {
                        "field_or_method": field,
                        "available": False,
                        "error_code": str(login.error_code),
                        "error_msg": str(login.error_msg),
                        "returned_fields": "",
                        "row_count": 0,
                        "sample": "",
                        "evidence": "",
                        "notes": "login_failed",
                    }
                )
            return pd.DataFrame(rows)
        for field in fields:
            requested = ",".join(["date", "code", "close", field])
            result = bs.query_history_k_data_plus(
                code,
                requested,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3",
            )
            records: list[list[str]] = []
            while result.next():
                records.append(result.get_row_data())
            returned_fields = list(getattr(result, "fields", []))
            available = str(result.error_code) == "0" and field in returned_fields and bool(records)
            sample = records[0] if records else []
            rows.append(
                {
                    "field_or_method": field,
                    "available": bool(available),
                    "error_code": str(result.error_code),
                    "error_msg": str(result.error_msg),
                    "returned_fields": ",".join(returned_fields),
                    "row_count": int(len(records)),
                    "sample": json.dumps(sample, ensure_ascii=False),
                    "evidence": f"code={code};start={start_date};end={end_date};rows={len(records)}",
                    "notes": "" if available else str(result.error_msg),
                }
            )
    finally:
        try:
            bs.logout()
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "field_or_method": "logout",
                    "available": False,
                    "error_code": "",
                    "error_msg": str(exc),
                    "returned_fields": "",
                    "row_count": 0,
                    "sample": "",
                    "evidence": "",
                    "notes": "logout_error",
                }
            )
    return pd.DataFrame(rows)


def probe_baostock_stock_basic_fields(
    *,
    code: str = "sh.600000",
    fields: Sequence[str] = BAOSTOCK_STOCK_BASIC_PROBE_FIELDS,
) -> pd.DataFrame:
    """Live-probe Baostock stock-basic fields for share-base or cap candidates."""

    rows: list[dict[str, Any]] = []
    try:
        import baostock as bs  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        for field in fields:
            rows.append(
                {
                    "field_or_method": field,
                    "available": False,
                    "error_code": "",
                    "error_msg": "",
                    "returned_fields": "",
                    "row_count": 0,
                    "sample": "",
                    "evidence": "",
                    "notes": f"import_error={exc}",
                }
            )
        return pd.DataFrame(rows)

    login = bs.login()
    login_ok = str(login.error_code) == "0"
    try:
        if not login_ok:
            for field in fields:
                rows.append(
                    {
                        "field_or_method": field,
                        "available": False,
                        "error_code": str(login.error_code),
                        "error_msg": str(login.error_msg),
                        "returned_fields": "",
                        "row_count": 0,
                        "sample": "",
                        "evidence": "",
                        "notes": "login_failed",
                    }
                )
            return pd.DataFrame(rows)
        result = bs.query_stock_basic(code=code)
        records: list[list[str]] = []
        while result.next():
            records.append(result.get_row_data())
        returned_fields = list(getattr(result, "fields", []))
        returned_normalized = {_normalize_field(field): field for field in returned_fields}
        for field in fields:
            available = str(result.error_code) == "0" and _normalize_field(field) in returned_normalized and bool(records)
            rows.append(
                {
                    "field_or_method": field,
                    "available": bool(available),
                    "error_code": str(result.error_code),
                    "error_msg": str(result.error_msg),
                    "returned_fields": ",".join(returned_fields),
                    "row_count": int(len(records)),
                    "sample": json.dumps(records[0] if records else [], ensure_ascii=False),
                    "evidence": f"code={code};rows={len(records)};returned_fields={','.join(returned_fields)}",
                    "notes": "" if available else "field not returned by query_stock_basic",
                }
            )
    finally:
        try:
            bs.logout()
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "field_or_method": "logout",
                    "available": False,
                    "error_code": "",
                    "error_msg": str(exc),
                    "returned_fields": "",
                    "row_count": 0,
                    "sample": "",
                    "evidence": "",
                    "notes": "logout_error",
                }
            )
    return pd.DataFrame(rows)


def summarize_audit(
    *,
    snapshot_root: Path,
    manifest: Mapping[str, Any],
    schema: pd.DataFrame,
    audit: pd.DataFrame,
    run_id: str,
) -> dict[str, Any]:
    field_summary = {
        "true_industry_in_snapshot": _domain_any(audit, "true_industry"),
        "true_market_cap_in_snapshot": _domain_any(audit, "true_market_cap"),
        "share_base_in_snapshot": _domain_any(audit, "share_base"),
        "turnover_in_snapshot": _domain_any(audit, "turnover"),
        "size_liquidity_proxy_in_snapshot": _domain_any(audit, "size_liquidity_proxy"),
        "pit_status_fields_in_snapshot": _domain_any(audit, "pit_status"),
        "baostock_query_stock_industry_available": _client_available(audit, "query_stock_industry"),
        "baostock_history_query_available": _client_available(audit, "query_history_k_data_plus"),
        "baostock_history_turn_available": _history_field_available(audit, "turn"),
        "baostock_history_pctchg_available": _history_field_available(audit, "pctChg"),
        "baostock_history_valuation_fields_available": _history_fields_any(audit, ("peTTM", "pbMRQ", "psTTM", "pcfNcfTTM")),
        "baostock_history_market_cap_fields_available": _history_fields_any(audit, TRUE_MARKET_CAP_FIELDS),
        "baostock_history_share_base_fields_available": _history_fields_any(audit, SHARE_BASE_FIELDS),
        "baostock_stock_basic_market_cap_fields_available": _stock_basic_fields_any(audit, TRUE_MARKET_CAP_FIELDS),
        "baostock_stock_basic_share_base_fields_available": _stock_basic_fields_any(audit, SHARE_BASE_FIELDS),
    }
    recommendations = _recommendations(field_summary)
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "snapshot_root": str(snapshot_root),
        "snapshot_id": manifest.get("snapshot_id", ""),
        "dataset": manifest.get("dataset", ""),
        "schema_tables": sorted(schema.loc[schema["exists"], "table"].dropna().unique().tolist()),
        "field_summary": field_summary,
        "recommendations": recommendations,
        "candidate_count": 0,
        "status": "field_audit",
    }


def run_v2_industry_size_source_audit(
    *,
    root: str | Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    probe_baostock: bool = True,
    live_history_probe: bool = False,
    live_stock_basic_probe: bool = False,
    history_probe_code: str = "sh.600000",
    history_probe_start_date: str = "2026-05-25",
    history_probe_end_date: str = "2026-06-01",
    write_research_log: bool = False,
    research_log_path: Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    snapshot_root = resolve_snapshot_root(root)
    run_id = f"v2_industry_size_source_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(snapshot_root)
    schema = read_snapshot_schema(snapshot_root)
    baostock_probe = probe_baostock_client() if probe_baostock else pd.DataFrame()
    history_field_probe = (
        probe_baostock_history_fields(
            code=history_probe_code,
            start_date=history_probe_start_date,
            end_date=history_probe_end_date,
        )
        if live_history_probe
        else pd.DataFrame()
    )
    stock_basic_field_probe = (
        probe_baostock_stock_basic_fields(code=history_probe_code)
        if live_stock_basic_probe
        else pd.DataFrame()
    )
    audit = build_source_field_audit(
        schema,
        baostock_probe=baostock_probe,
        history_field_probe=history_field_probe,
        stock_basic_field_probe=stock_basic_field_probe,
    )
    summary = summarize_audit(
        snapshot_root=snapshot_root,
        manifest=manifest,
        schema=schema,
        audit=audit,
        run_id=run_id,
    )
    summary_md = render_summary_markdown(summary, audit)

    schema.to_csv(run_dir / "snapshot_schema.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(run_dir / "source_field_audit.csv", index=False, encoding="utf-8-sig")
    if not baostock_probe.empty:
        baostock_probe.to_csv(run_dir / "baostock_client_capabilities.csv", index=False, encoding="utf-8-sig")
    if not history_field_probe.empty:
        history_field_probe.to_csv(run_dir / "baostock_history_field_probe.csv", index=False, encoding="utf-8-sig")
    if not stock_basic_field_probe.empty:
        stock_basic_field_probe.to_csv(run_dir / "baostock_stock_basic_field_probe.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(summary_md, encoding="utf-8")
    if write_research_log:
        research_log_path.parent.mkdir(parents=True, exist_ok=True)
        research_log_path.write_text(summary_md, encoding="utf-8")
    return {**summary, "run_dir": str(run_dir)}


def render_summary_markdown(summary: Mapping[str, Any], audit: pd.DataFrame) -> str:
    field_summary = dict(summary.get("field_summary", {}))
    recommendations = list(summary.get("recommendations", []))
    lines = [
        "# v2 Industry/Size Source Audit",
        "",
        f"- run_id: `{summary.get('run_id', '')}`",
        f"- snapshot_id: `{summary.get('snapshot_id', '')}`",
        f"- snapshot_root: `{summary.get('snapshot_root', '')}`",
        f"- status: `{summary.get('status', '')}`",
        f"- candidate_count: `{summary.get('candidate_count', 0)}`",
        "",
        "## Field Summary",
        "",
    ]
    for key, value in field_summary.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Key Evidence", ""])
    for domain in (
        "true_industry",
        "true_market_cap",
        "share_base",
        "turnover",
        "size_liquidity_proxy",
        "baostock_client",
        "baostock_history_field",
        "baostock_stock_basic_field",
    ):
        subset = audit.loc[audit["domain"] == domain]
        if subset.empty:
            continue
        available = subset.loc[subset["available"], "field_or_method"].astype(str).tolist()
        label = domain.replace("_", " ")
        lines.append(f"- {label}: {', '.join(available) if available else 'not available'}")
    lines.extend(["", "## Recommendations", ""])
    for item in recommendations:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This audit is a source/schema gate. It does not promote any strategy candidate. "
            "It decides whether the current v2 snapshot can support true industry and size neutralization, "
            "or whether the dataset must be extended first.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_parquet_schema(path: Path) -> list[tuple[str, str]]:
    try:
        import pyarrow.parquet as pq

        schema = pq.read_schema(path)
        return [(name, str(schema.field(name).type)) for name in schema.names]
    except Exception:  # noqa: BLE001
        frame = pd.read_parquet(path)
        return [(str(name), str(dtype)) for name, dtype in frame.dtypes.items()]


def _tradeable_panel_schema_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_table: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if not row.get("exists"):
            continue
        by_table.setdefault(str(row.get("table", "")), []).append(row)
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for table in ("daily_universe", "daily_bars", "daily_metrics"):
        for row in by_table.get(table, []):
            field = str(row.get("field", ""))
            if not field or field in seen:
                continue
            seen.add(field)
            output.append(
                {
                    "table": "tradeable_panel",
                    "field": field,
                    "dtype": str(row.get("dtype", "")),
                    "exists": True,
                    "path": "virtual:daily_universe INNER JOIN daily_bars LEFT JOIN daily_metrics ON date,code",
                }
            )
    return output


def _table_field_map(schema: pd.DataFrame) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for row in schema.loc[schema["exists"]].to_dict("records"):
        table = str(row.get("table", ""))
        field = str(row.get("field", ""))
        if not field:
            continue
        output.setdefault(table, set()).add(_normalize_field(field))
    return output


def _field_group_rows(
    domain: str,
    requirement: str,
    candidates: Iterable[str],
    table_fields: Mapping[str, set[str]],
    *,
    missing_note: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in candidates:
        normalized = _normalize_field(field)
        tables = sorted(table for table, fields in table_fields.items() if normalized in fields)
        rows.append(
            {
                "domain": domain,
                "requirement": requirement,
                "evidence_source": ",".join(tables) if tables else "snapshot_schema",
                "field_or_method": field,
                "available": bool(tables),
                "evidence": f"present_in={','.join(tables)}" if tables else "",
                "notes": "" if tables else missing_note,
            }
        )
    return rows


def _normalize_field(field: str) -> str:
    return re.sub(r"[^a-z0-9]", "", field.strip().lower())


def _domain_any(audit: pd.DataFrame, domain: str) -> bool:
    subset = audit.loc[audit["domain"] == domain]
    return bool(subset["available"].any()) if not subset.empty else False


def _client_available(audit: pd.DataFrame, method: str) -> bool | None:
    subset = audit.loc[(audit["domain"] == "baostock_client") & (audit["field_or_method"] == method)]
    if subset.empty:
        return None
    return bool(subset.iloc[0]["available"])


def _history_field_available(audit: pd.DataFrame, field: str) -> bool | None:
    subset = audit.loc[(audit["domain"] == "baostock_history_field") & (audit["field_or_method"] == field)]
    if subset.empty:
        return None
    return bool(subset.iloc[0]["available"])


def _history_fields_any(audit: pd.DataFrame, fields: Sequence[str]) -> bool | None:
    normalized = {_normalize_field(field) for field in fields}
    subset = audit.loc[
        (audit["domain"] == "baostock_history_field")
        & (audit["field_or_method"].map(_normalize_field).isin(normalized))
    ]
    if subset.empty:
        return None
    return bool(subset["available"].any())


def _stock_basic_fields_any(audit: pd.DataFrame, fields: Sequence[str]) -> bool | None:
    normalized = {_normalize_field(field) for field in fields}
    subset = audit.loc[
        (audit["domain"] == "baostock_stock_basic_field")
        & (audit["field_or_method"].map(_normalize_field).isin(normalized))
    ]
    if subset.empty:
        return None
    return bool(subset["available"].any())


def _baostock_method_requirement(method: str) -> str:
    if method == "query_stock_industry":
        return "Potential source for true industry classification."
    if method == "query_history_k_data_plus":
        return "Potential source for additional daily fields if supported by Baostock."
    if method == "query_stock_basic":
        return "Current PIT security master source."
    return "Fundamental source candidate; requires separate field and timing audit before use."


def _history_field_requirement(field: str) -> str:
    if field == "turn":
        return "Daily turnover field candidate from Baostock history bars."
    if field in {"turnover", "turnover_rate"}:
        return "Common turnover aliases; useful only if Baostock accepts the field name."
    if field == "pctChg":
        return "Daily percent-change field candidate from Baostock history bars."
    if field in {"peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"}:
        return "Daily valuation field candidate from Baostock history bars."
    if _normalize_field(field) in {_normalize_field(item) for item in TRUE_MARKET_CAP_FIELDS}:
        return "Daily market-cap field candidate from Baostock history bars."
    if _normalize_field(field) in {_normalize_field(item) for item in SHARE_BASE_FIELDS}:
        return "Daily share-base field candidate from Baostock history bars."
    return "Additional Baostock history field candidate."


def _stock_basic_field_requirement(field: str) -> str:
    if _normalize_field(field) in {_normalize_field(item) for item in TRUE_MARKET_CAP_FIELDS}:
        return "Market-cap field candidate from Baostock stock basic."
    if _normalize_field(field) in {_normalize_field(item) for item in SHARE_BASE_FIELDS}:
        return "Share-base field candidate from Baostock stock basic."
    return "Additional Baostock stock-basic field candidate."


def _recommendations(field_summary: Mapping[str, Any]) -> list[str]:
    recommendations: list[str] = []
    if not field_summary.get("true_industry_in_snapshot"):
        if field_summary.get("baostock_query_stock_industry_available"):
            recommendations.append(
                "Extend v2 with a cached Baostock industry table before running true industry-neutral tests."
            )
        else:
            recommendations.append("Find an external point-in-time industry source before industry-neutral tests.")
    if not field_summary.get("true_market_cap_in_snapshot"):
        recommendations.append(
            "Do not claim market-cap neutralization from the current snapshot; add vetted market-cap/float-cap data or keep size controls explicitly proxy-only."
        )
    if field_summary.get("baostock_history_market_cap_fields_available") is False and field_summary.get("baostock_stock_basic_market_cap_fields_available") is False:
        recommendations.append(
            "Current Baostock probes did not expose direct market-cap fields; evaluate an external PIT market-cap/float-cap source."
        )
    if field_summary.get("baostock_history_share_base_fields_available") is False and field_summary.get("baostock_stock_basic_share_base_fields_available") is False:
        recommendations.append(
            "Current Baostock probes did not expose share-base fields; deriving true market cap from Baostock alone is not supported."
        )
    if field_summary.get("size_liquidity_proxy_in_snapshot"):
        recommendations.append(
            "Continue using amount/volume-derived fields only as liquidity/size proxies until true cap fields are added."
        )
    if not field_summary.get("turnover_in_snapshot"):
        if field_summary.get("baostock_history_turn_available"):
            recommendations.append("Baostock daily history supports `turn`; add a cached daily metrics table before using turnover controls.")
        else:
            recommendations.append("Audit whether Baostock daily bars can add turnover before using turnover-based liquidity controls.")
    if field_summary.get("baostock_history_valuation_fields_available"):
        recommendations.append("Baostock daily history supports valuation fields; verify PIT timing before using valuation factors.")
    recommendations.append("Keep frontier signals at candidate-frontier/backtest_only until true exposure controls are audited.")
    return recommendations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="Snapshot root or v2 root containing latest_manifest.json.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-probe-baostock", action="store_true", help="Skip import-only Baostock client capability probe.")
    parser.add_argument("--live-history-probe", action="store_true", help="Issue short live Baostock history field probes.")
    parser.add_argument("--live-stock-basic-probe", action="store_true", help="Issue a short live Baostock stock-basic field probe.")
    parser.add_argument("--history-probe-code", default="sh.600000")
    parser.add_argument("--history-probe-start-date", default="2026-05-25")
    parser.add_argument("--history-probe-end-date", default="2026-06-01")
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", default=str(DEFAULT_RESEARCH_LOG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_v2_industry_size_source_audit(
        root=args.root,
        output_dir=Path(args.output_dir),
        probe_baostock=not args.no_probe_baostock,
        live_history_probe=bool(args.live_history_probe),
        live_stock_basic_probe=bool(args.live_stock_basic_probe),
        history_probe_code=args.history_probe_code,
        history_probe_start_date=args.history_probe_start_date,
        history_probe_end_date=args.history_probe_end_date,
        write_research_log=bool(args.write_research_log),
        research_log_path=Path(args.research_log_path),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
