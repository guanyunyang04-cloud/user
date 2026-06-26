from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from quant_data_platform.lake.catalog import LakeDatasetRecord, ResearchDataLake


DEFAULT_INDUSTRY_SOURCE_PATH = "daily_research/cache/industry_map_tq.csv"
DEFAULT_BOARD_SOURCE_PATH = "daily_research/cache/imported_board_membership/infoharbor_block.dat"
SNAPSHOT_SEMANTICS = "latest_static_snapshot"
INDUSTRY_SOURCE_KIND_FILE = "file"
INDUSTRY_SOURCE_KIND_LAKE_SIDE_CAR = "lake_sidecar"
BOARD_SOURCE_KIND_FILE = "file"
BOARD_SOURCE_KIND_EMPTY = "empty"


@dataclass(frozen=True)
class SectorBoardViewSpec:
    source_market_dataset_id: str
    industry_source_path: str = DEFAULT_INDUSTRY_SOURCE_PATH
    board_source_path: str = DEFAULT_BOARD_SOURCE_PATH
    industry_source_kind: str = INDUSTRY_SOURCE_KIND_FILE
    board_source_kind: str = BOARD_SOURCE_KIND_FILE
    allow_empty_board: bool = False
    as_of_date: str = ""
    view_kind: str = SNAPSHOT_SEMANTICS
    view_name: str = "sector_board_latest_static"
    snapshot_semantics: str = SNAPSHOT_SEMANTICS

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_market_dataset_id"] = str(payload["source_market_dataset_id"] or "").strip()
        payload["industry_source_path"] = str(payload["industry_source_path"] or DEFAULT_INDUSTRY_SOURCE_PATH).strip()
        payload["board_source_path"] = str(payload["board_source_path"] or DEFAULT_BOARD_SOURCE_PATH).strip()
        payload["industry_source_kind"] = str(payload.get("industry_source_kind", "") or INDUSTRY_SOURCE_KIND_FILE).strip().lower()
        payload["board_source_kind"] = str(payload.get("board_source_kind", "") or BOARD_SOURCE_KIND_FILE).strip().lower()
        payload["allow_empty_board"] = bool(payload.get("allow_empty_board", False))
        payload["as_of_date"] = _normalize_date(payload.get("as_of_date", ""))
        payload["view_kind"] = str(payload.get("view_kind", "") or SNAPSHOT_SEMANTICS).strip().lower()
        payload["view_name"] = str(payload.get("view_name", "") or "sector_board_latest_static").strip().lower()
        payload["snapshot_semantics"] = str(payload.get("snapshot_semantics", "") or SNAPSHOT_SEMANTICS).strip().lower()
        return payload


@dataclass(frozen=True)
class SectorBoardViewRecord:
    dataset_id: str
    dataset_kind: str
    fingerprint: str
    status: str
    content_paths: dict[str, str]
    metadata: dict[str, Any]
    industry_map_frame: pd.DataFrame
    board_membership_frame: pd.DataFrame
    board_summary_frame: pd.DataFrame


def _normalize_date(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return pd.Timestamp(raw).strftime("%Y-%m-%d")


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _normalize_symbols(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _normalize_symbol(raw)
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _normalize_symbol(raw: Any) -> str:
    value = str(raw or "").strip().upper()
    if not value:
        return ""
    if "#" in value:
        value = value.split("#")[-1].strip()
    value = value.replace(" ", "")
    if "." in value:
        code, suffix = value.split(".", 1)
        code = "".join(ch for ch in code if ch.isdigit())
        suffix = suffix.strip().upper()
        return f"{code}.{suffix}" if code and suffix else ""
    code = "".join(ch for ch in value if ch.isdigit())
    if len(code) != 6:
        return ""
    if code.startswith("6"):
        suffix = "SH"
    elif code.startswith(("0", "2", "3")):
        suffix = "SZ"
    elif code.startswith(("4", "8", "9")):
        suffix = "BJ"
    else:
        return ""
    return f"{code}.{suffix}"


def _read_lake_symbols(lake: ResearchDataLake, dataset_id: str) -> list[str]:
    metadata = lake.describe_dataset(str(dataset_id))
    if str(metadata.get("dataset_kind", "")) != "policy_input_bundle":
        raise ValueError(f"sector_board_view_blocker: source dataset is not policy_input_bundle: {dataset_id}")
    market_path = str(dict(metadata.get("content_paths", {}) or {}).get("bronze_market_data", "") or "")
    if not market_path or not Path(market_path).exists():
        raise ValueError(f"sector_board_view_blocker: source bundle has no bronze_market_data: {dataset_id}")
    market = pd.read_parquet(market_path, columns=["symbol"])
    return _normalize_symbols(market["symbol"].dropna().astype(str).unique().tolist())


def _read_industry_map(path: Path, *, lake_symbols: list[str], as_of_date: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"sector_board_view_blocker: industry source file does not exist: {path}")
    raw = pd.read_csv(path, dtype=str)
    symbol_col = "symbol" if "symbol" in raw.columns else "stock" if "stock" in raw.columns else ""
    if not symbol_col or "industry" not in raw.columns:
        raise ValueError(f"sector_board_view_blocker: industry source must contain stock/symbol and industry columns: {path}")
    frame = pd.DataFrame(
        {
            "symbol": [_normalize_symbol(item) for item in raw[symbol_col].tolist()],
            "industry": raw["industry"].fillna("").astype(str).str.strip(),
        }
    )
    frame = frame.loc[(frame["symbol"] != "") & (frame["industry"] != "")].drop_duplicates(subset=["symbol"])
    source_symbols = set(frame["symbol"].tolist())
    lake_set = set(lake_symbols)
    effective = frame.loc[frame["symbol"].isin(lake_set)].copy()
    if effective.empty:
        raise ValueError(f"sector_board_view_blocker: industry source has no overlap with lake symbols: {path}")
    effective["source"] = "tq_industry_cache"
    effective["as_of_date"] = str(as_of_date)
    effective = effective.sort_values("symbol").reset_index(drop=True)
    missing = sorted(lake_set - source_symbols)
    extra = sorted(source_symbols - lake_set)
    coverage = {
        "source_symbols": int(len(source_symbols)),
        "lake_symbols": int(len(lake_set)),
        "covered_symbols": int(len(effective)),
        "missing_symbols": missing,
        "extra_source_symbols": extra[:100],
        "coverage_ratio": float(len(effective) / len(lake_set)) if lake_set else 0.0,
    }
    return effective[["symbol", "industry", "source", "as_of_date"]], coverage


def _read_industry_map_from_lake_sidecar(
    lake: ResearchDataLake,
    *,
    source_market_dataset_id: str,
    lake_symbols: list[str],
    as_of_date: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    market_metadata = lake.describe_dataset(str(source_market_dataset_id))
    if str(market_metadata.get("dataset_kind", "")) != "policy_input_bundle":
        raise ValueError(f"sector_board_view_blocker: source dataset is not policy_input_bundle: {source_market_dataset_id}")
    parameters = dict(market_metadata.get("parameters", {}) or {})
    sidecar_ids = dict(parameters.get("sidecar_dataset_ids", {}) or {})
    industry_dataset_id = str(sidecar_ids.get("industry_concept", "") or "").strip()
    if not industry_dataset_id:
        raise ValueError(
            f"sector_board_view_blocker: policy_input_bundle has no industry_concept sidecar: {source_market_dataset_id}"
        )
    industry_metadata = lake.describe_dataset(industry_dataset_id)
    if str(industry_metadata.get("dataset_kind", "")) != "data_platform_industry_concept":
        raise ValueError(
            "sector_board_view_blocker: sidecar dataset is not data_platform_industry_concept: "
            f"{industry_dataset_id}"
        )
    paths = dict(industry_metadata.get("content_paths", {}) or {})
    industry_path = Path(str(paths.get("silver_domain_data", "") or ""))
    if not industry_path.exists():
        raise ValueError(
            "sector_board_view_blocker: industry_concept sidecar is missing silver_domain_data: "
            f"{industry_dataset_id}"
        )
    raw = pd.read_parquet(industry_path)
    symbol_col = "symbol" if "symbol" in raw.columns else "code" if "code" in raw.columns else ""
    industry_col = "industry" if "industry" in raw.columns else "industryClassification" if "industryClassification" in raw.columns else ""
    trade_date_col = "trade_date" if "trade_date" in raw.columns else "updateDate" if "updateDate" in raw.columns else "date" if "date" in raw.columns else ""
    if not symbol_col or not industry_col:
        raise ValueError(
            "sector_board_view_blocker: sidecar industry source must contain symbol/code and industry/industryClassification columns: "
            f"{industry_dataset_id}"
        )
    frame = pd.DataFrame(
        {
            "symbol": [_normalize_symbol(item) for item in raw[symbol_col].tolist()],
            "industry": raw[industry_col].fillna("").astype(str).str.strip(),
        }
    )
    if trade_date_col:
        frame["trade_date"] = pd.to_datetime(raw[trade_date_col], errors="coerce")
    else:
        frame["trade_date"] = pd.NaT
    frame = frame.loc[(frame["symbol"] != "") & (frame["industry"] != "")].copy()
    if frame.empty:
        raise ValueError(f"sector_board_view_blocker: sidecar industry source has no usable rows: {industry_dataset_id}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    if str(as_of_date or "").strip():
        as_of_ts = pd.Timestamp(as_of_date)
        frame = frame.loc[frame["trade_date"].isna() | (frame["trade_date"] <= as_of_ts)].copy()
    lake_set = set(lake_symbols)
    frame = frame.loc[frame["symbol"].isin(lake_set)].copy()
    if frame.empty:
        raise ValueError(f"sector_board_view_blocker: sidecar industry source has no overlap with lake symbols: {industry_dataset_id}")
    frame["trade_date"] = frame["trade_date"].fillna(pd.Timestamp(as_of_date) if str(as_of_date or "").strip() else pd.Timestamp.now().normalize())
    frame = frame.sort_values(["symbol", "trade_date"]).drop_duplicates(subset=["symbol"], keep="last").reset_index(drop=True)
    resolved_trade_date = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
    frame["trade_date"] = resolved_trade_date
    frame["concept_tags"] = ""
    frame["source"] = "baostock_lake_sidecar"
    frame["as_of_date"] = str(as_of_date)
    source_symbols = set(_normalize_symbols(raw[symbol_col].dropna().astype(str).unique().tolist()))
    coverage = {
        "source_kind": INDUSTRY_SOURCE_KIND_LAKE_SIDE_CAR,
        "source_dataset_id": industry_dataset_id,
        "source_symbols": int(len(source_symbols)),
        "lake_symbols": int(len(lake_set)),
        "covered_symbols": int(len(frame)),
        "missing_symbols": sorted(lake_set - set(frame["symbol"].tolist())),
        "extra_source_symbols": sorted(source_symbols - lake_set)[:100],
        "coverage_ratio": float(len(frame) / len(lake_set)) if lake_set else 0.0,
        "as_of_date": str(as_of_date),
    }
    return frame[["symbol", "trade_date", "industry", "concept_tags", "source", "as_of_date"]], coverage


def _parse_board_header(line: str) -> dict[str, str]:
    parts = [item.strip() for item in line.lstrip("#").split(",")]
    raw_name = parts[0] if parts else ""
    if "_" in raw_name:
        kind, name = raw_name.split("_", 1)
    else:
        kind, name = raw_name, raw_name
    return {
        "board_kind": kind.strip().upper(),
        "board_name": name.strip(),
        "board_code": parts[2].strip() if len(parts) > 2 else "",
        "source_start_date": _tdx_date(parts[3]) if len(parts) > 3 else "",
        "source_update_date": _tdx_date(parts[4]) if len(parts) > 4 else "",
    }


def _tdx_date(value: str) -> str:
    raw = str(value or "").strip()
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return ""


def _read_board_membership(path: Path, *, lake_symbols: list[str], as_of_date: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"sector_board_view_blocker: board source file does not exist: {path}")
    text = path.read_text(encoding="gbk", errors="ignore")
    rows: list[dict[str, Any]] = []
    raw_counts: dict[tuple[str, str, str], set[str]] = {}
    current: dict[str, str] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            current = _parse_board_header(line)
            continue
        if current is None:
            continue
        key = (current["board_kind"], current["board_name"], current["board_code"])
        raw_counts.setdefault(key, set())
        for token in line.split(","):
            symbol = _normalize_symbol(token)
            if not symbol:
                continue
            raw_counts[key].add(symbol)
            if symbol not in set(lake_symbols):
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "board_kind": current["board_kind"],
                    "board_name": current["board_name"],
                    "board_code": current["board_code"],
                    "source": "tdx_infoharbor_block",
                    "source_start_date": current["source_start_date"],
                    "source_update_date": current["source_update_date"],
                    "as_of_date": str(as_of_date),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"sector_board_view_blocker: board source has no lake-covered membership: {path}")
    frame = frame.drop_duplicates(subset=["symbol", "board_kind", "board_name", "board_code"]).sort_values(
        ["board_kind", "board_name", "symbol"]
    ).reset_index(drop=True)
    summary_rows: list[dict[str, Any]] = []
    for key, source_symbols in sorted(raw_counts.items()):
        kind, name, code = key
        covered = frame.loc[
            (frame["board_kind"] == kind)
            & (frame["board_name"] == name)
            & (frame["board_code"] == code),
            "symbol",
        ].nunique()
        if int(covered) <= 0:
            continue
        summary_rows.append(
            {
                "board_kind": kind,
                "board_name": name,
                "board_code": code,
                "source": "tdx_infoharbor_block",
                "source_member_count": int(len(source_symbols)),
                "lake_covered_count": int(covered),
                "as_of_date": str(as_of_date),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["board_kind", "board_name"]).reset_index(drop=True)
    source_symbols = {symbol for members in raw_counts.values() for symbol in members}
    board_coverage = {
        "board_count": int(len(summary)),
        "source_members": int(len(source_symbols)),
        "lake_covered_members": int(frame["symbol"].nunique()),
        "missing_symbols": sorted(set(lake_symbols) - set(frame["symbol"].unique())),
        "extra_source_symbols": sorted(source_symbols - set(lake_symbols))[:100],
    }
    return frame, summary, board_coverage


def _empty_board_membership(*, lake_symbols: list[str], as_of_date: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = pd.DataFrame(
        columns=[
            "symbol",
            "board_kind",
            "board_name",
            "board_code",
            "source",
            "source_start_date",
            "source_update_date",
            "as_of_date",
        ]
    )
    summary = pd.DataFrame(
        columns=[
            "board_kind",
            "board_name",
            "board_code",
            "source",
            "source_member_count",
            "lake_covered_count",
            "as_of_date",
        ]
    )
    board_coverage = {
        "status": "empty_explicit",
        "board_count": 0,
        "source_members": 0,
        "lake_covered_members": 0,
        "missing_symbol_count": int(len(lake_symbols)),
        "missing_symbols": [],
        "extra_source_symbols": [],
        "as_of_date": str(as_of_date),
    }
    return frame, summary, board_coverage


def build_sector_board_view_from_policy_bundle(
    *,
    lake: ResearchDataLake,
    spec: SectorBoardViewSpec | Mapping[str, Any],
    reuse: bool = True,
) -> LakeDatasetRecord:
    resolved = spec if isinstance(spec, SectorBoardViewSpec) else SectorBoardViewSpec(**dict(spec))
    payload = resolved.to_dict()
    if not payload["source_market_dataset_id"]:
        raise ValueError("sector_board_view_blocker: source_market_dataset_id is required.")
    as_of_date = str(payload.get("as_of_date") or "")
    industry_source_kind = str(payload.get("industry_source_kind", INDUSTRY_SOURCE_KIND_FILE) or INDUSTRY_SOURCE_KIND_FILE).strip().lower()
    board_source_kind = str(payload.get("board_source_kind", BOARD_SOURCE_KIND_FILE) or BOARD_SOURCE_KIND_FILE).strip().lower()
    allow_empty_board = bool(payload.get("allow_empty_board", False))
    industry_path = Path(payload["industry_source_path"])
    board_path = Path(payload["board_source_path"])
    lake_symbols = _read_lake_symbols(lake, str(payload["source_market_dataset_id"]))
    if industry_source_kind == INDUSTRY_SOURCE_KIND_FILE:
        if not industry_path.exists():
            raise ValueError(f"sector_board_view_blocker: industry source file does not exist: {industry_path}")
        if not as_of_date:
            as_of_date = pd.Timestamp.fromtimestamp(industry_path.stat().st_mtime).strftime("%Y-%m-%d")
            payload["as_of_date"] = as_of_date
        industry, industry_coverage = _read_industry_map(industry_path, lake_symbols=lake_symbols, as_of_date=as_of_date)
    elif industry_source_kind == INDUSTRY_SOURCE_KIND_LAKE_SIDE_CAR:
        industry, industry_coverage = _read_industry_map_from_lake_sidecar(
            lake,
            source_market_dataset_id=str(payload["source_market_dataset_id"]),
            lake_symbols=lake_symbols,
            as_of_date=as_of_date,
        )
        if not as_of_date and not industry.empty and "trade_date" in industry.columns:
            as_of_date = str(pd.to_datetime(industry["trade_date"]).max().strftime("%Y-%m-%d"))
            payload["as_of_date"] = as_of_date
    else:
        raise ValueError(f"sector_board_view_blocker: unsupported industry_source_kind: {industry_source_kind}")
    payload["snapshot_semantics"] = SNAPSHOT_SEMANTICS
    source_files: dict[str, Any] = {
        "industry": _file_signature(industry_path) if industry_source_kind == INDUSTRY_SOURCE_KIND_FILE else {
            "source_kind": INDUSTRY_SOURCE_KIND_LAKE_SIDE_CAR,
            "source_market_dataset_id": str(payload["source_market_dataset_id"]),
            "source_dataset_id": str(industry_coverage.get("source_dataset_id", "")),
        },
    }
    if board_source_kind == BOARD_SOURCE_KIND_FILE:
        if not board_path.exists():
            raise ValueError(f"sector_board_view_blocker: board source file does not exist: {board_path}")
        if not as_of_date:
            as_of_date = pd.Timestamp.fromtimestamp(board_path.stat().st_mtime).strftime("%Y-%m-%d")
            payload["as_of_date"] = as_of_date
        board, summary, board_coverage = _read_board_membership(board_path, lake_symbols=lake_symbols, as_of_date=as_of_date)
        source_files["board"] = _file_signature(board_path)
    elif board_source_kind == BOARD_SOURCE_KIND_EMPTY:
        if not allow_empty_board:
            raise ValueError("sector_board_view_blocker: empty board requires allow_empty_board")
        board, summary, board_coverage = _empty_board_membership(lake_symbols=lake_symbols, as_of_date=as_of_date)
        source_files["board"] = {"source_kind": BOARD_SOURCE_KIND_EMPTY}
    else:
        raise ValueError(f"sector_board_view_blocker: unsupported board_source_kind: {board_source_kind}")
    payload["source_files"] = source_files
    source_cache = {
        "source_market_dataset_id": payload["source_market_dataset_id"],
        "snapshot_semantics": SNAPSHOT_SEMANTICS,
        "as_of_date": as_of_date,
        "industry_source_kind": industry_source_kind,
        "board_source_kind": board_source_kind,
        "allow_empty_board": allow_empty_board,
        "industry_source_path": str(industry_path.resolve()) if industry_source_kind == INDUSTRY_SOURCE_KIND_FILE else "",
        "board_source_path": str(board_path.resolve()) if board_source_kind == BOARD_SOURCE_KIND_FILE else "",
        "industry_source_dataset_id": str(industry_coverage.get("source_dataset_id", "")),
        "industry_coverage": industry_coverage,
        "board_coverage": board_coverage,
    }
    return lake.save_sector_board_view(
        spec=payload,
        industry_map_frame=industry,
        board_membership_frame=board,
        board_summary_frame=summary,
        source_cache=source_cache,
        reuse=bool(reuse),
    )


def load_sector_board_view(*, lake: ResearchDataLake, sector_board_view_id: str) -> SectorBoardViewRecord:
    metadata = lake.describe_dataset(str(sector_board_view_id))
    if str(metadata.get("dataset_kind", "")) != "policy_sector_board_view":
        raise ValueError(f"sector_board_view_blocker: dataset is not policy_sector_board_view: {sector_board_view_id}")
    paths = dict(metadata.get("content_paths", {}) or {})
    industry_path = Path(str(paths.get("industry_map", "") or ""))
    board_path = Path(str(paths.get("board_membership", "") or ""))
    summary_path = Path(str(paths.get("board_summary", "") or ""))
    if not industry_path.exists():
        raise ValueError(f"sector_board_view_blocker: missing industry_map frame: {industry_path}")
    if not board_path.exists():
        raise ValueError(f"sector_board_view_blocker: missing board_membership frame: {board_path}")
    industry = pd.read_parquet(industry_path)
    board = pd.read_parquet(board_path)
    summary = pd.read_parquet(summary_path) if summary_path.exists() else pd.DataFrame()
    return SectorBoardViewRecord(
        dataset_id=str(metadata["dataset_id"]),
        dataset_kind=str(metadata["dataset_kind"]),
        fingerprint=str(metadata["fingerprint"]),
        status=str(metadata.get("status", "") or "loaded"),
        content_paths=paths,
        metadata=metadata,
        industry_map_frame=industry,
        board_membership_frame=board,
        board_summary_frame=summary,
    )


def resolve_sector_board_view_for_policy_inputs(
    *,
    lake: ResearchDataLake,
    dataset_id: str,
    sector_board_view_id: str = "",
    sector_board_view_spec: SectorBoardViewSpec | Mapping[str, Any] | None = None,
) -> SectorBoardViewRecord | None:
    if str(sector_board_view_id or "").strip():
        return load_sector_board_view(lake=lake, sector_board_view_id=str(sector_board_view_id).strip())
    if sector_board_view_spec is None:
        return None
    payload = sector_board_view_spec if isinstance(sector_board_view_spec, SectorBoardViewSpec) else SectorBoardViewSpec(**dict(sector_board_view_spec))
    spec_dict = payload.to_dict()
    if not str(spec_dict.get("source_market_dataset_id", "") or "").strip():
        spec_dict["source_market_dataset_id"] = str(dataset_id)
        payload = SectorBoardViewSpec(**spec_dict)
    record = build_sector_board_view_from_policy_bundle(lake=lake, spec=payload)
    return load_sector_board_view(lake=lake, sector_board_view_id=record.dataset_id)
