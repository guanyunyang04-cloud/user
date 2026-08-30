"""Research Event Update: announcement responsibilities."""

from __future__ import annotations

import math
import os
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

from quantlab.data.providers.cninfo import _fetch_cninfo_announcements

from .config import (
    ANNOUNCEMENT_COLUMNS,
    END_DATE,
    MAX_WORKERS,
    START_DATE,
    ResearchEventUpdateError,
)
from .context import (
    _identity_symbols,
    _next_open_date,
    _normalized_text,
    _open_dates,
    _read_state,
    _request_json,
    _runtime,
    _stable_hash,
    _text,
    _workspace,
    _write_parquet,
    _write_state,
)
from .report_download import (
    _download_symbol_files,
)


def _cninfo_announcement_path(workspace: Path, symbol: str) -> Path:
    return _runtime(workspace) / "raw" / "cninfo_announcements" / f"symbol={symbol.replace('.', '_')}.parquet"


def _eastmoney_announcement_path(workspace: Path, symbol: str) -> Path:
    return _runtime(workspace) / "raw" / "eastmoney_announcements" / f"symbol={symbol.replace('.', '_')}.parquet"


def _fetch_cninfo_symbol(symbol: str) -> pd.DataFrame:
    frame = _fetch_cninfo_announcements(
        symbol=symbol,
        start_date=START_DATE,
        end_date=END_DATE,
        page_size=30,
        max_pages=0,
    )
    if frame.empty:
        return pd.DataFrame({"_query_symbol": pd.Series(dtype="object")})
    frame["_query_symbol"] = symbol
    return frame


def _fetch_eastmoney_announcements(symbol: str) -> pd.DataFrame:
    code = str(symbol).split(".", 1)[0]
    url = "https://np-anotice-stock.eastmoney.com/api/security/ann"
    common = {
        "sr": "-1",
        "page_size": "100",
        "ann_type": "A",
        "client_source": "web",
        "f_node": "0",
        "s_node": "0",
        "stock_list": code,
        "begin_time": START_DATE,
        "end_time": END_DATE,
    }
    page = 1
    total_pages = 1
    rows: list[dict[str, Any]] = []
    with requests.Session() as session:
        while page <= total_pages:
            payload = _request_json(
                "GET",
                url,
                params={**common, "page_index": str(page)},
                session=session,
            )
            data = dict(payload.get("data", {}) or {})
            total_pages = math.ceil(int(data.get("total_hits", 0) or 0) / 100)
            for item in list(data.get("list", []) or []):
                record = dict(item)
                record["_query_symbol"] = symbol
                columns = list(record.pop("columns", []) or [])
                codes = list(record.pop("codes", []) or [])
                if columns:
                    record.update(dict(columns[0]))
                selected = next(
                    (dict(value) for value in codes if str(value.get("stock_code", "")) == code),
                    {},
                )
                record.update(selected)
                rows.append(record)
            page += 1
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame({"_query_symbol": pd.Series(dtype="object")})


def download_announcements(
    *,
    workspace_root: str | Path | None = None,
    max_workers: int = MAX_WORKERS,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    symbols = _identity_symbols(workspace)
    cninfo = _download_symbol_files(
        workspace=workspace,
        state_key="cninfo_announcements",
        symbols=symbols,
        path_for=_cninfo_announcement_path,
        fetch=_fetch_cninfo_symbol,
        max_workers=max_workers,
        allow_failures=True,
    )
    cninfo_failures = sorted(dict(cninfo.get("failures", {}) or {}))
    eastmoney = _download_symbol_files(
        workspace=workspace,
        state_key="eastmoney_announcements",
        symbols=cninfo_failures,
        path_for=_eastmoney_announcement_path,
        fetch=_fetch_eastmoney_announcements,
        max_workers=max_workers,
        allow_failures=True,
    )
    state = _read_state(workspace)
    source_policy = {
        "primary_provider": "cninfo",
        "fallback_provider": "eastmoney",
        "fallback_unit": "whole_symbol",
        "cninfo_failed_symbols": cninfo_failures,
        "cninfo_failed_symbol_count": len(cninfo_failures),
        "eastmoney_fallback_symbols": cninfo_failures,
        "eastmoney_fallback_symbol_count": len(cninfo_failures),
        "eastmoney_unresolved_symbols": sorted(dict(eastmoney.get("failures", {}) or {})),
        "stale_non_fallback_cache_excluded": True,
    }
    state["announcement_source_policy"] = source_policy
    _write_state(workspace, state)
    return {
        "cninfo": cninfo,
        "eastmoney": eastmoney,
        "source_policy": source_policy,
    }


def _announcement_raw_source_paths(
    workspace: Path,
    *,
    fallback_symbols: Sequence[str],
) -> dict[str, list[Path]]:
    """Select canonical announcement caches without admitting stale fallbacks."""
    cninfo_root = _runtime(workspace) / "raw" / "cninfo_announcements"
    return {
        "cninfo": sorted(cninfo_root.glob("*.parquet")),
        "eastmoney": [
            path
            for symbol in sorted(set(fallback_symbols))
            if (path := _eastmoney_announcement_path(workspace, symbol)).is_file()
        ],
    }


_CATEGORY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "risk_warning",
        ("风险提示", "退市", "立案", "处罚", "诉讼", "仲裁", "违约", "冻结", "警示"),
    ),
    (
        "performance",
        ("业绩", "年度报告", "年报", "季度报告", "季报", "半年度报告", "快报", "预告"),
    ),
    (
        "restructuring",
        ("重组", "收购", "并购", "重大资产", "发行股份购买"),
    ),
    (
        "holding_change",
        ("增持", "减持", "持股变动", "回购", "股权激励", "解除限售", "解禁"),
    ),
    (
        "financing",
        ("融资", "增发", "配股", "可转债", "公司债", "发行证券"),
    ),
    (
        "governance",
        ("董事会", "监事会", "股东大会", "独立董事", "高级管理人员"),
    ),
    (
        "operations",
        ("合同", "中标", "项目", "投资", "经营", "订单"),
    ),
)


def _announcement_category(title: Any, provider_category: Any = "") -> str:
    text = unicodedata.normalize("NFKC", _text(title))
    provider = _text(provider_category)
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(keyword in text or keyword in provider for keyword in keywords):
            return category
    return provider if provider and not provider.isdigit() else "other"


def _normalize_cninfo_announcements(
    raw: pd.DataFrame,
    *,
    open_dates: np.ndarray,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=ANNOUNCEMENT_COLUMNS)
    data = raw.copy()
    data["symbol"] = data.get("_query_symbol", data.get("symbol", "")).fillna("").astype(str)
    data["source_date"] = pd.to_datetime(data.get("source_date", data.get("trade_date")), errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    data["title"] = data.get("title", "").fillna("").astype(str)
    data["normalized_title"] = data["title"].map(_normalized_text)
    data["feature_available_date"] = _next_open_date(data["source_date"], open_dates)
    data["category"] = [
        _announcement_category(title, codes)
        for title, codes in zip(data["title"], data.get("announcement_type_codes", ""), strict=True)
    ]
    data["file_size_kb"] = pd.to_numeric(data.get("file_size_kb"), errors="coerce")
    data["_event_key"] = [
        _stable_hash(symbol, date, title)
        for symbol, date, title in zip(data["symbol"], data["source_date"], data["normalized_title"], strict=True)
    ]
    data["announcement_id"] = data["_event_key"].map(lambda value: f"ann_{value[:24]}")
    data["trade_date"] = data["source_date"]
    for column, default in {
        "publish_time": "",
        "announcement_type_codes": "",
        "cninfo_announcement_id": "",
        "eastmoney_art_code": "",
        "org_id": "",
        "url": "",
        "pdf_url": "",
        "file_size_kb": math.nan,
        "cninfo_present": True,
        "eastmoney_present": False,
        "source_disagreement": False,
        "source": "cninfo",
    }.items():
        if column not in data:
            data[column] = default
    valid = data["source_date"].between(START_DATE, END_DATE) & data["normalized_title"].ne("")
    return data.loc[valid, list(ANNOUNCEMENT_COLUMNS)].drop_duplicates().reset_index(drop=True)


def _normalize_eastmoney_announcements(
    raw: pd.DataFrame,
    *,
    open_dates: np.ndarray,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=ANNOUNCEMENT_COLUMNS)
    data = raw.copy()
    data["symbol"] = data.get("_query_symbol", "").fillna("").astype(str)
    data["source_date"] = pd.to_datetime(data.get("notice_date"), errors="coerce").dt.strftime("%Y-%m-%d")
    data["publish_time"] = pd.to_datetime(
        data.get("display_time", data.get("notice_date")), errors="coerce"
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    data["title"] = data.get("title", "").fillna("").astype(str)
    data["normalized_title"] = data["title"].map(_normalized_text)
    data["feature_available_date"] = _next_open_date(data["source_date"], open_dates)
    provider_category = data.get("column_name", pd.Series("", index=data.index))
    data["category"] = [
        _announcement_category(title, category)
        for title, category in zip(data["title"], provider_category, strict=True)
    ]
    data["_event_key"] = [
        _stable_hash(symbol, date, title)
        for symbol, date, title in zip(data["symbol"], data["source_date"], data["normalized_title"], strict=True)
    ]
    data["announcement_id"] = data["_event_key"].map(lambda value: f"ann_{value[:24]}")
    data["trade_date"] = data["source_date"]
    data["announcement_type_codes"] = data.get("column_code", "")
    data["cninfo_announcement_id"] = ""
    data["eastmoney_art_code"] = data.get("art_code", "")
    data["org_id"] = ""
    code = data.get("stock_code", data["symbol"].str.split(".").str[0])
    data["url"] = [
        f"https://data.eastmoney.com/notices/detail/{item}/{article}.html" if _text(article) else ""
        for item, article in zip(code, data["eastmoney_art_code"], strict=True)
    ]
    data["pdf_url"] = ""
    data["file_size_kb"] = math.nan
    data["cninfo_present"] = False
    data["eastmoney_present"] = True
    data["source_disagreement"] = False
    data["source"] = "eastmoney_announcement_metadata"
    valid = data["source_date"].between(START_DATE, END_DATE) & data["normalized_title"].ne("")
    return data.loc[valid, list(ANNOUNCEMENT_COLUMNS)].drop_duplicates().reset_index(drop=True)


def _normalize_announcement_sources(
    *,
    workspace: Path,
    open_dates: np.ndarray,
    fallback_symbols: list[str],
) -> tuple[list[Path], dict[str, int], dict[str, list[Path]]]:
    normalized_root = _runtime(workspace) / "normalized" / "announcements"
    selected_paths = _announcement_raw_source_paths(workspace, fallback_symbols=fallback_symbols)
    source_specs = (
        ("cninfo", selected_paths["cninfo"], _normalize_cninfo_announcements),
        ("eastmoney", selected_paths["eastmoney"], _normalize_eastmoney_announcements),
    )
    normalized_paths: list[Path] = []
    source_rows: dict[str, int] = {}
    for source_name, raw_paths, normalize in source_specs:
        count = 0
        for raw_path in raw_paths:
            output_path = normalized_root / source_name / raw_path.name
            if not output_path.is_file() or raw_path.stat().st_mtime_ns > output_path.stat().st_mtime_ns:
                _write_parquet(normalize(pd.read_parquet(raw_path), open_dates=open_dates), output_path)
            count += int(pq.ParquetFile(output_path).metadata.num_rows)
            normalized_paths.append(output_path)
        source_rows[source_name] = count
    if not normalized_paths:
        raise ResearchEventUpdateError("announcement_cache_missing")
    return normalized_paths, source_rows, selected_paths


def _merge_announcement_sources(paths: list[Path], prepared: Path) -> None:
    prepared.parent.mkdir(parents=True, exist_ok=True)
    temporary = prepared.with_suffix(".tmp.parquet")
    scans = ",".join(f"'{str(path).replace(chr(39), chr(39) * 2)}'" for path in paths)
    sql = f"""
    COPY (
      WITH source AS (
        SELECT * FROM read_parquet([{scans}], union_by_name=true)
      ), ranked AS (
        SELECT *,
          row_number() OVER (
            PARTITION BY announcement_id
            ORDER BY cninfo_present DESC, eastmoney_present DESC, source
          ) AS rn,
          bool_or(cninfo_present) OVER (PARTITION BY announcement_id) AS any_cninfo,
          bool_or(eastmoney_present) OVER (PARTITION BY announcement_id) AS any_eastmoney,
          count(DISTINCT category) OVER (PARTITION BY announcement_id) > 1 AS disagrees
        FROM source
      )
      SELECT
        announcement_id,symbol,trade_date,source_date,feature_available_date,
        publish_time,title,normalized_title,category,announcement_type_codes,
        cninfo_announcement_id,eastmoney_art_code,org_id,url,pdf_url,file_size_kb,
        any_cninfo AS cninfo_present,any_eastmoney AS eastmoney_present,
        source_disagreement OR disagrees AS source_disagreement,
        CASE WHEN any_cninfo AND any_eastmoney
          THEN 'cninfo+eastmoney_announcement_metadata' ELSE source END AS source
      FROM ranked WHERE rn=1
      ORDER BY trade_date,symbol,announcement_id
    ) TO '{str(temporary).replace(chr(39), chr(39) * 2)}'
      (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    connection = duckdb.connect()
    try:
        connection.execute(sql)
    finally:
        connection.close()
    os.replace(temporary, prepared)


def _announcement_summary(
    *,
    prepared: Path,
    source_rows: dict[str, int],
    source_policy: dict[str, Any],
    selected_paths: dict[str, list[Path]],
) -> dict[str, Any]:
    frame = pd.read_parquet(
        prepared,
        columns=["announcement_id", "source_date", "cninfo_present", "eastmoney_present"],
    )
    return {
        "status": "completed",
        "announcement_path": str(prepared),
        "announcement_row_count": len(frame),
        "source_normalized_rows": source_rows,
        "source_policy": source_policy,
        "selected_cninfo_cache_count": len(selected_paths["cninfo"]),
        "selected_eastmoney_fallback_cache_count": len(selected_paths["eastmoney"]),
        "matched_announcement_count": int((frame["cninfo_present"] & frame["eastmoney_present"]).sum()),
        "earliest_announcement_date": str(frame["source_date"].min()),
        "latest_announcement_date": str(frame["source_date"].max()),
    }


def prepare_announcements(
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    open_dates = _open_dates(workspace)
    state = _read_state(workspace)
    source_policy = dict(state.get("announcement_source_policy", {}) or {})
    if not source_policy:
        raise ResearchEventUpdateError("announcement_source_policy_missing")
    fallback_symbols = list(source_policy.get("eastmoney_fallback_symbols", []) or [])
    normalized_paths, source_rows, selected_paths = _normalize_announcement_sources(
        workspace=workspace,
        open_dates=open_dates,
        fallback_symbols=fallback_symbols,
    )
    prepared = _runtime(workspace) / "prepared" / "announcement.parquet"
    _merge_announcement_sources(normalized_paths, prepared)
    result = _announcement_summary(
        prepared=prepared,
        source_rows=source_rows,
        source_policy=source_policy,
        selected_paths=selected_paths,
    )
    state = _read_state(workspace)
    state["announcements_prepared"] = result
    state["status"] = "announcements_prepared"
    _write_state(workspace, state)
    return result
