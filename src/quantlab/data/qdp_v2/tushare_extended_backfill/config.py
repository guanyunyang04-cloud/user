"""Tushare Extended Backfill: config responsibilities."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quantlab.data.domains.contracts import DataDomain
from quantlab.data.qdp_v2.auxiliary_update import (
    _MinuteLimiter,
)
from quantlab.data.qdp_v2.provider_credentials import (
    ProviderCredentialError,
    resolve_tushare_api_url,
    resolve_tushare_rate_limit,
)

UPDATE_ID = "tushare_extended_backfill_v1"


BURN_IN_START = "2010-01-01"


RESEARCH_START = "2011-01-01"


END_DATE = "2025-12-31"


FORBIDDEN_YEAR = 2026


PAGE_SIZE = 5_000


MAX_PAGES_PER_TASK = 1_000


MAX_WORKERS = 3


MAX_REQUESTS_PER_MINUTE = 120


EXPECTED_FACTOR_FIELD_COUNT = 261


EXPECTED_FACTOR_SCHEMA_HASH = "14cf668f77da06b5ddaffe59057e98c25107a870d7ec006ec5e9ee1ae574d488"


PREPARED_TRANSFORM_VERSION = "fixed_provider_types_v2"


MARGIN_FIELDS = (
    "trade_date",
    "exchange_id",
    "rzye",
    "rzmre",
    "rzche",
    "rqye",
    "rqmcl",
    "rzrqye",
    "rqyl",
)


MARGIN_DETAIL_FIELDS = (
    "trade_date",
    "ts_code",
    "rzye",
    "rqye",
    "rzmre",
    "rqyl",
    "rzche",
    "rqchl",
    "rqmcl",
    "rzrqye",
)


MARGIN_SECS_FIELDS = ("trade_date", "ts_code", "name", "exchange")


MONEYFLOW_FIELDS = (
    "ts_code",
    "trade_date",
    "buy_sm_vol",
    "buy_sm_amount",
    "sell_sm_vol",
    "sell_sm_amount",
    "buy_md_vol",
    "buy_md_amount",
    "sell_md_vol",
    "sell_md_amount",
    "buy_lg_vol",
    "buy_lg_amount",
    "sell_lg_vol",
    "sell_lg_amount",
    "buy_elg_vol",
    "buy_elg_amount",
    "sell_elg_vol",
    "sell_elg_amount",
    "net_mf_vol",
    "net_mf_amount",
)


class TushareExtendedBackfillError(RuntimeError):
    pass


def _resolve_tushare_api_url(
    workspace_root: str | Path | None = None,
) -> str:
    try:
        return resolve_tushare_api_url(workspace_root)
    except ProviderCredentialError as exc:
        raise TushareExtendedBackfillError(str(exc)) from exc


class ResilientTushareClient:
    """Use one HTTP session per worker while sharing a global rate limiter."""

    def __init__(
        self,
        token: str,
        *,
        rpm: int = MAX_REQUESTS_PER_MINUTE,
        timeout: int = 60,
        workspace_root: str | Path | None = None,
    ) -> None:
        secret = str(token or "").strip()
        if not secret:
            raise TushareExtendedBackfillError("tushare_token_missing")
        self._token = secret
        self._timeout = int(timeout)
        self._limiter = _MinuteLimiter(resolve_tushare_rate_limit(rpm, workspace_root=workspace_root))
        self._local = threading.local()
        self._url = _resolve_tushare_api_url(workspace_root)

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"Accept-Encoding": "gzip"})
            self._local.session = session
        return session

    def fetch(
        self,
        api_name: str,
        *,
        params: Mapping[str, Any],
        fields: Sequence[str],
        retries: int = 5,
    ) -> pd.DataFrame:
        last_error = ""
        for attempt in range(max(1, int(retries))):
            self._limiter.wait()
            try:
                response = self._session().post(
                    self._url,
                    json={
                        "api_name": str(api_name),
                        "token": self._token,
                        "params": dict(params),
                        "fields": ",".join(fields),
                    },
                    timeout=self._timeout,
                )
                response.raise_for_status()
                payload = response.json()
                if int(payload.get("code", -1)) != 0:
                    raise TushareExtendedBackfillError(f"tushare_api_error:{api_name}:code={payload.get('code')}")
                data = dict(payload.get("data", {}) or {})
                names = [str(item) for item in list(data.get("fields", []) or [])]
                rows = list(data.get("items", []) or [])
                return pd.DataFrame(rows, columns=names or list(fields))
            except Exception as exc:  # noqa: BLE001 - bounded provider retry
                last_error = f"{type(exc).__name__}:{str(exc)[:200]}"
                if attempt + 1 < max(1, int(retries)):
                    time.sleep(min(2**attempt, 8))
        raise TushareExtendedBackfillError(f"tushare_request_failed:{api_name}:{last_error}")


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    api_name: str
    domain: str
    mode: str
    fields: tuple[str, ...]
    primary_key: tuple[str, ...]
    frequency: str
    contract_version: str
    optional: bool = False


SPECS = {
    "stk-factor-pro": EndpointSpec(
        name="stk-factor-pro",
        api_name="stk_factor_pro",
        domain=DataDomain.STK_FACTOR_PRO_RAW,
        mode="trade_date",
        fields=(),
        primary_key=("trade_date", "security_id"),
        frequency="1d",
        contract_version="qdp_v2_tushare_stk_factor_pro_raw_v1",
    ),
    "margin": EndpointSpec(
        name="margin",
        api_name="margin",
        domain=DataDomain.MARGIN_MARKET,
        mode="year",
        fields=MARGIN_FIELDS,
        primary_key=("trade_date", "exchange_id"),
        frequency="1d_market",
        contract_version="qdp_v2_tushare_margin_market_raw_v1",
    ),
    "margin-detail": EndpointSpec(
        name="margin-detail",
        api_name="margin_detail",
        domain=DataDomain.MARGIN_DETAIL,
        mode="trade_date",
        fields=MARGIN_DETAIL_FIELDS,
        primary_key=("trade_date", "security_id"),
        frequency="1d",
        contract_version="qdp_v2_tushare_margin_detail_raw_v1",
    ),
    "margin-secs": EndpointSpec(
        name="margin-secs",
        api_name="margin_secs",
        domain=DataDomain.MARGIN_SECS,
        mode="trade_date",
        fields=MARGIN_SECS_FIELDS,
        primary_key=("trade_date", "security_id"),
        frequency="1d_membership",
        contract_version="qdp_v2_tushare_margin_secs_raw_v1",
        optional=True,
    ),
    "moneyflow": EndpointSpec(
        name="moneyflow",
        api_name="moneyflow",
        domain=DataDomain.MONEYFLOW_RAW,
        mode="trade_date",
        fields=MONEYFLOW_FIELDS,
        primary_key=("trade_date", "security_id"),
        frequency="1d",
        contract_version="qdp_v2_tushare_moneyflow_raw_v1",
    ),
}
