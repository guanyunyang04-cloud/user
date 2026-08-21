"""Margin Eligibility Update: config responsibilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

UPDATE_ID = "margin_eligibility_exchange_history_v1"


PROVENANCE_REPAIR_ID = "margin_detail_provenance_repair_v1"


SSE_SEMANTICS_REPAIR_ID = "margin_eligibility_sse_semantics_repair_v1"


MARGIN_ELIGIBILITY_CONTRACT_V1 = "qdp_v2_margin_eligibility_exchange_tristate_v1"


MARGIN_ELIGIBILITY_CONTRACT_V2 = "qdp_v2_margin_eligibility_exchange_tristate_v2"


START_DATE = "2011-01-01"


END_DATE = "2025-12-31"


MAX_WORKERS = 4


ENDPOINTS = ("sse_detail", "szse_detail", "szse_eligibility")


SSE_URL = "https://query.sse.com.cn/marketdata/tradedata/queryMargin.do"


SZSE_URL = "https://www.szse.cn/api/report/ShowReport"


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"


OFFICIAL_COLUMNS = (
    "symbol",
    "trade_date",
    "exchange",
    "endpoint",
    "name",
    "eligible",
    "finance_eligible",
    "securities_lending_eligible",
    "rzye",
    "rqye",
    "rzmre",
    "rqyl",
    "rzche",
    "rqchl",
    "rqmcl",
    "rzrqye",
    "source",
)


MARGIN_DETAIL_COLUMNS = (
    "security_id",
    "symbol",
    "ts_code",
    "trade_date",
    "rzye",
    "rqye",
    "rzmre",
    "rqyl",
    "rzche",
    "rqchl",
    "rqmcl",
    "rzrqye",
    "source_date",
    "feature_available_date",
    "burn_in_only",
    "source",
)


MARGIN_ELIGIBILITY_COLUMNS = (
    "symbol",
    "trade_date",
    "exchange",
    "eligibility_state",
    "eligible",
    "finance_eligible",
    "securities_lending_eligible",
    "detail_observed",
    "source_available",
    "eligibility_source_available",
    "detail_source_available",
    "source_date",
    "feature_available_date",
    "burn_in_only",
    "source",
)


class MarginEligibilityUpdateError(RuntimeError):
    pass


@dataclass(frozen=True)
class EndpointResult:
    endpoint: str
    frame: pd.DataFrame
    response_sha256: str
    response_bytes: int
