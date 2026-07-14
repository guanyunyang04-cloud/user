from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from quant_data_platform.core.paths import qdp_paths


IDENTITY_COLUMNS = [
    "security_id",
    "official_org_id",
    "issuer_name",
    "exchange",
    "list_date",
    "current_symbol",
    "identity_source",
]

SYMBOL_HISTORY_COLUMNS = [
    "security_id",
    "symbol",
    "effective_from",
    "effective_to",
    "name_on_date",
    "board_on_date",
    "evidence_source",
    "official_document_hash",
]


def normalize_symbol(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return ""
    if raw.lower().startswith("sh."):
        return f"{raw[3:9]}.SH"
    if raw.lower().startswith("sz."):
        return f"{raw[3:9]}.SZ"
    if raw.lower().startswith("bj."):
        return f"{raw[3:9]}.BJ"
    raw = raw.replace("_", ".")
    if re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", raw):
        return raw
    if re.fullmatch(r"(SH|SZ|BJ)\d{6}", raw):
        return f"{raw[2:]}.{raw[:2]}"
    if re.fullmatch(r"\d{6}", raw):
        if raw.startswith(("4", "8", "92")):
            suffix = "BJ"
        elif raw.startswith(("5", "6", "9")):
            suffix = "SH"
        else:
            suffix = "SZ"
        return f"{raw}.{suffix}"
    return raw


def exchange_for_symbol(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    suffix = normalized.rsplit(".", 1)[-1] if "." in normalized else ""
    return {"SH": "SSE", "SZ": "SZSE", "BJ": "BSE"}.get(suffix, "UNKNOWN")


def board_for_symbol(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    code = normalized.split(".", 1)[0]
    if normalized.endswith(".SH") and code.startswith(("688", "689")):
        return "STAR"
    if normalized.endswith(".SZ") and code.startswith(("300", "301", "302")):
        return "ChiNext"
    if normalized.endswith(".BJ"):
        return "BSE"
    if normalized.endswith((".SH", ".SZ")):
        return "MainBoard"
    return "UNKNOWN"


def default_security_id(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    code = normalized.split(".", 1)[0]
    exchange = exchange_for_symbol(normalized)
    if not code or exchange == "UNKNOWN":
        raise ValueError(f"cannot_assign_security_id:{symbol!r}")
    return f"QDP-CN-{exchange}-{code}"


def _iso_date_or_blank(value: Any) -> str:
    raw = str(value or "").strip()[:10]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return ""
    try:
        return pd.Timestamp(raw).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return ""


def default_symbol_history_config(workspace_root: str | Path | None = None) -> Path:
    return qdp_paths(workspace_root).project_root / "configs" / "qdp_v3_symbol_history.json"


def _read_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        return {"identities": [], "symbol_history": []}
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"identity_config_must_be_object:{resolved}")
    return payload


@dataclass
class SecurityIdentityRegistry:
    identities: pd.DataFrame
    symbol_history: pd.DataFrame

    def __post_init__(self) -> None:
        self.identities = self.identities.loc[:, IDENTITY_COLUMNS].copy()
        self.symbol_history = self.symbol_history.loc[:, SYMBOL_HISTORY_COLUMNS].copy()
        for column in IDENTITY_COLUMNS:
            self.identities[column] = self.identities[column].fillna("").astype(str).str.strip()
        for column in SYMBOL_HISTORY_COLUMNS:
            self.symbol_history[column] = self.symbol_history[column].fillna("").astype(str).str.strip()
        self.symbol_history["symbol"] = self.symbol_history["symbol"].map(normalize_symbol)
        self._validate()
        self._alias_to_security = self._build_alias_map()
        self._history_by_security = {
            security_id: group.sort_values(["effective_from", "effective_to", "symbol"]).reset_index(drop=True)
            for security_id, group in self.symbol_history.groupby("security_id", sort=False)
        }

    @classmethod
    def from_sources(
        cls,
        *,
        provider_symbols: Iterable[str],
        security_master: pd.DataFrame | None = None,
        config_path: str | Path | None = None,
        workspace_root: str | Path | None = None,
    ) -> "SecurityIdentityRegistry":
        config = _read_config(config_path or default_symbol_history_config(workspace_root))
        identities = pd.DataFrame(list(config.get("identities", []) or []), columns=IDENTITY_COLUMNS)
        history = pd.DataFrame(list(config.get("symbol_history", []) or []), columns=SYMBOL_HISTORY_COLUMNS)
        if identities.empty:
            identities = pd.DataFrame(columns=IDENTITY_COLUMNS)
        if history.empty:
            history = pd.DataFrame(columns=SYMBOL_HISTORY_COLUMNS)
        master = security_master.copy() if isinstance(security_master, pd.DataFrame) else pd.DataFrame()
        if not master.empty:
            symbol_column = "symbol" if "symbol" in master.columns else "provider_symbol" if "provider_symbol" in master.columns else ""
            if symbol_column:
                master["_normalized_symbol"] = master[symbol_column].map(normalize_symbol)
        known_aliases = set(history["symbol"].map(normalize_symbol).tolist()) if not history.empty else set()
        symbols = {normalize_symbol(item) for item in provider_symbols if normalize_symbol(item)}
        if not master.empty and "_normalized_symbol" in master.columns:
            symbols.update(master["_normalized_symbol"].dropna().astype(str).tolist())
        identity_rows = identities.to_dict("records")
        history_rows = history.to_dict("records")
        known_security_ids = {str(item.get("security_id", "")) for item in identity_rows}
        for symbol in sorted(symbols):
            if symbol in known_aliases:
                continue
            security_id = default_security_id(symbol)
            master_row: dict[str, Any] = {}
            if not master.empty and "_normalized_symbol" in master.columns:
                matched = master.loc[master["_normalized_symbol"].eq(symbol)]
                if not matched.empty:
                    master_row = matched.iloc[-1].to_dict()
            list_date = _iso_date_or_blank(master_row.get("list_date", ""))
            delist_date = _iso_date_or_blank(master_row.get("delist_date", ""))
            # BaoStock includes a terminal status row on outDate for ordinary
            # delistings/absorptions.  Keep that date inside a single-symbol
            # identity interval so the final suspended/delisted fact remains
            # mappable.  Official multi-symbol change records override this
            # generic interval explicitly (for example 300114 ends on
            # 2025-02-16 before 302132 starts on 2025-02-17).
            effective_to = delist_date or "9999-12-31"
            name = str(master_row.get("name", "") or master_row.get("code_name", "") or "").strip()
            provider_board = str(master_row.get("board", "") or "").strip()
            board = provider_board if provider_board in {"MainBoard", "ChiNext", "STAR", "BSE"} else board_for_symbol(symbol)
            if security_id not in known_security_ids:
                identity_rows.append(
                    {
                        "security_id": security_id,
                        "official_org_id": str(master_row.get("official_org_id", "") or ""),
                        "issuer_name": name,
                        "exchange": exchange_for_symbol(symbol),
                        "list_date": list_date,
                        "current_symbol": symbol,
                        "identity_source": "baostock_stock_basic" if master_row else "provider_symbol_stable",
                    }
                )
                known_security_ids.add(security_id)
            history_rows.append(
                {
                    "security_id": security_id,
                    "symbol": symbol,
                    "effective_from": list_date or "1900-01-01",
                    "effective_to": effective_to,
                    # query_stock_basic is a current snapshot.  Its name must
                    # not be backfilled into history; date-local query_all_stock
                    # observations split this interval later.
                    "name_on_date": "",
                    "board_on_date": board,
                    "evidence_source": "baostock.query_stock_basic" if master_row else "provider_symbol_stable_no_change_evidence",
                    "official_document_hash": "",
                }
            )
        return cls(
            identities=pd.DataFrame(identity_rows, columns=IDENTITY_COLUMNS).drop_duplicates("security_id", keep="last"),
            symbol_history=pd.DataFrame(history_rows, columns=SYMBOL_HISTORY_COLUMNS).drop_duplicates(
                ["security_id", "symbol", "effective_from", "effective_to"], keep="last"
            ),
        )

    def with_pit_name_observations(self, observations: pd.DataFrame) -> "SecurityIdentityRegistry":
        """Split symbol intervals only from names observed on explicit dates."""

        if observations is None or observations.empty:
            return self
        data = observations.copy()
        symbol_column = "provider_symbol" if "provider_symbol" in data.columns else "symbol" if "symbol" in data.columns else ""
        date_column = "trade_date" if "trade_date" in data.columns else "date" if "date" in data.columns else ""
        name_column = "name" if "name" in data.columns else "code_name" if "code_name" in data.columns else ""
        if not symbol_column or not date_column or not name_column:
            raise ValueError("pit_name_observation_columns_missing")
        data["symbol"] = data[symbol_column].map(normalize_symbol)
        data["trade_date"] = data[date_column].fillna("").astype(str).str.slice(0, 10)
        data["name"] = data[name_column].fillna("").astype(str).str.strip()
        data = data.loc[data["symbol"].ne("") & data["trade_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}") & data["name"].ne("")]
        data = data.drop_duplicates(["symbol", "trade_date"], keep="last").sort_values(["symbol", "trade_date"])
        observation_groups = {
            symbol: group.loc[:, ["trade_date", "name"]].reset_index(drop=True)
            for symbol, group in data.groupby("symbol", sort=False)
        }
        rows: list[dict[str, Any]] = []
        for base in self.symbol_history.sort_values(["security_id", "effective_from"]).to_dict("records"):
            start = str(base["effective_from"])
            end = str(base["effective_to"])
            observed = observation_groups.get(str(base["symbol"]), pd.DataFrame(columns=["trade_date", "name"]))
            observed = observed.loc[observed["trade_date"].between(start, end)].copy()
            changes: list[tuple[str, str]] = []
            previous_name = str(base.get("name_on_date", "") or "")
            for item in observed.itertuples(index=False):
                date, name = str(item.trade_date), str(item.name)
                if name != previous_name:
                    changes.append((date, name))
                    previous_name = name
            cursor = start
            current_name = str(base.get("name_on_date", "") or "")
            name_from_observation = False
            base_evidence_source = str(base.get("evidence_source", "") or "")

            def segment_evidence() -> str:
                return ";".join(
                    item
                    for item in (
                        base_evidence_source,
                        "baostock.query_all_stock(date_snapshot)" if name_from_observation else "",
                    )
                    if item
                )

            for change_date, changed_name in changes:
                if change_date > cursor:
                    rows.append(
                        {
                            **base,
                            "effective_from": cursor,
                            "effective_to": (pd.Timestamp(change_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                            "name_on_date": current_name,
                            "evidence_source": segment_evidence(),
                        }
                    )
                cursor = change_date
                current_name = changed_name
                name_from_observation = True
            rows.append(
                {
                    **base,
                    "effective_from": cursor,
                    "effective_to": end,
                    "name_on_date": current_name,
                    "evidence_source": segment_evidence(),
                }
            )
        return SecurityIdentityRegistry(
            identities=self.identities.copy(),
            symbol_history=pd.DataFrame(rows, columns=SYMBOL_HISTORY_COLUMNS),
        )

    def _validate(self) -> None:
        if self.identities["security_id"].eq("").any():
            raise ValueError("blank_security_id")
        if self.identities["security_id"].duplicated().any():
            values = self.identities.loc[self.identities["security_id"].duplicated(False), "security_id"].unique().tolist()
            raise ValueError(f"duplicate_security_identity:{values[:10]}")
        if self.symbol_history[["security_id", "symbol", "effective_from", "effective_to"]].eq("").any().any():
            raise ValueError("blank_symbol_history_key")
        missing_identity = sorted(set(self.symbol_history["security_id"]) - set(self.identities["security_id"]))
        if missing_identity:
            raise ValueError(f"symbol_history_identity_missing:{missing_identity[:10]}")
        for security_id, group in self.symbol_history.groupby("security_id", sort=False):
            ordered = group.sort_values(["effective_from", "effective_to"])
            previous_to = ""
            for row in ordered.itertuples(index=False):
                if str(row.effective_from) > str(row.effective_to):
                    raise ValueError(f"symbol_history_invalid_interval:{security_id}:{row.symbol}")
                if previous_to and str(row.effective_from) <= previous_to:
                    raise ValueError(f"symbol_history_overlapping_intervals:{security_id}")
                previous_to = str(row.effective_to)
            if group["symbol"].nunique() > 1:
                missing_evidence = group["evidence_source"].eq("") | ~group["official_document_hash"].str.fullmatch(
                    r"[0-9a-fA-F]{64}", na=False
                )
                if missing_evidence.any():
                    aliases = group.loc[missing_evidence, "symbol"].astype(str).tolist()
                    raise ValueError(f"multi_symbol_identity_official_evidence_missing:{security_id}:{aliases[:10]}")

    def _build_alias_map(self) -> dict[str, str]:
        alias_map: dict[str, str] = {}
        for symbol, group in self.symbol_history.groupby("symbol", sort=False):
            security_ids = sorted(set(group["security_id"].astype(str)))
            if len(security_ids) != 1:
                raise ValueError(f"symbol_reused_across_security_ids:{symbol}:{security_ids}")
            alias_map[str(symbol)] = security_ids[0]
        return alias_map

    def security_id_for_provider_symbol(self, provider_symbol: str) -> str:
        return str(self._alias_to_security.get(normalize_symbol(provider_symbol), ""))

    def symbol_for_date(self, security_id: str, trade_date: str) -> str:
        history = self._history_by_security.get(str(security_id))
        if history is None or history.empty:
            return ""
        date = str(trade_date)
        matched = history.loc[(history["effective_from"] <= date) & (history["effective_to"] >= date)]
        return str(matched.iloc[-1]["symbol"]) if len(matched) == 1 else ""

    def map_frame(
        self,
        frame: pd.DataFrame,
        *,
        provider_symbol_column: str = "provider_symbol",
        date_column: str = "trade_date",
    ) -> pd.DataFrame:
        if frame is None or frame.empty:
            out = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
            for column in ("security_id", "symbol_on_date", "identity_mapping_status"):
                if column not in out.columns:
                    out[column] = pd.Series(dtype="string")
            return out
        if provider_symbol_column not in frame.columns or date_column not in frame.columns:
            raise ValueError(f"identity_mapping_columns_missing:{provider_symbol_column},{date_column}")
        out = frame.copy()
        out[provider_symbol_column] = out[provider_symbol_column].map(normalize_symbol)
        out[date_column] = out[date_column].astype(str).str.slice(0, 10)
        out["security_id"] = out[provider_symbol_column].map(self._alias_to_security).fillna("")
        out["symbol_on_date"] = ""
        for security_id, indices in out.groupby("security_id", sort=False).groups.items():
            if not security_id:
                continue
            history = self._history_by_security.get(str(security_id))
            if history is None:
                continue
            dates = out.loc[indices, date_column]
            for row in history.itertuples(index=False):
                mask = dates.ge(str(row.effective_from)) & dates.le(str(row.effective_to))
                if mask.any():
                    out.loc[dates.index[mask], "symbol_on_date"] = str(row.symbol)
        out["identity_mapping_status"] = "mapped"
        out.loc[out["security_id"].eq(""), "identity_mapping_status"] = "unmapped_provider_symbol"
        out.loc[out["security_id"].ne("") & out["symbol_on_date"].eq(""), "identity_mapping_status"] = "no_pit_symbol_interval"
        return out

    def identity_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return (
            self.identities.sort_values("security_id").reset_index(drop=True),
            self.symbol_history.sort_values(["security_id", "effective_from", "symbol"]).reset_index(drop=True),
        )
