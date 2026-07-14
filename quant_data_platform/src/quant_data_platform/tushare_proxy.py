from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlparse

import pandas as pd
import requests

from quant_data_platform.domains.contracts import (
    DataDomain,
    DomainFetchRequest,
    FetchRequest,
    HistoryPageFetchRequest,
    HistoryPageResult,
    ProviderResult,
    normalize_domain_frame,
    normalize_market_frame,
    validate_provider_name,
)


TUSHARE_PROXY_NAME = "tushare_proxy"
TUSHARE_PROXY_PROTOCOL = "tushare_compatible_http"
TUSHARE_PROXY_DEFAULT_URL = "https://ts.gyzcloud.top/api"
TUSHARE_PROXY_TOKEN_ENV = "QDP_TUSHARE_PROXY_TOKEN"
TUSHARE_PROXY_URL_ENV = "QDP_TUSHARE_PROXY_URL"
TUSHARE_PROXY_SAFE_RATE_PER_MINUTE = 96
TUSHARE_PROXY_BURST = 1
TUSHARE_PROXY_HISTORY_PAGE_SIZE = 8_000
TUSHARE_PROXY_DEFAULT_MINUTE_DAILY_LIMIT = 20_000
TUSHARE_PROXY_DEFAULT_MAX_IN_FLIGHT = 3
TUSHARE_PROXY_MIN_ADAPTIVE_RATE_PER_MINUTE = 90
TUSHARE_PROXY_RATE_REDUCTION_COOLDOWN_SECONDS = 60.0

_SECRET_KEY_FRAGMENTS = ("token", "secret", "password", "authorization", "api_key", "apikey")
_TIMESTAMP_ALIASES = ("trade_time", "datetime", "trade_datetime", "trade_date", "date", "time")
_RETRYABLE_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class TushareProxyError(RuntimeError):
    """Base error whose message is guaranteed not to contain credentials."""


class TushareProxyProtocolError(TushareProxyError):
    pass


class TushareProxyQuotaError(TushareProxyError):
    pass


def token_fingerprint(token: str) -> str:
    value = str(token or "").strip()
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def redact_secrets(value: Any, *, secrets: Iterable[str] = ()) -> Any:
    """Return a recursively redacted, JSON-safe view of diagnostic data."""

    secret_values = tuple(item for item in (str(secret or "") for secret in secrets) if item)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if any(fragment in key_text.lower() for fragment in _SECRET_KEY_FRAGMENTS):
                out[key_text] = "<redacted>"
            else:
                out[key_text] = redact_secrets(item, secrets=secret_values)
        return out
    if isinstance(value, (list, tuple, set)):
        return [redact_secrets(item, secrets=secret_values) for item in value]
    if isinstance(value, BaseException):
        value = f"{type(value).__name__}: {value}"
    text = str(value) if not isinstance(value, (str, int, float, bool, type(None))) else value
    if isinstance(text, str):
        for secret in secret_values:
            text = text.replace(secret, "<redacted>")
        # MCP-style URLs embed the credential in the path.  They are never a
        # supported QDP endpoint, but redact them defensively in diagnostics.
        text = _redact_token_url(text)
    return text


def _redact_token_url(text: str) -> str:
    marker = "/token="
    output = str(text)
    start = output.lower().find(marker)
    while start >= 0:
        value_start = start + len(marker)
        value_end = len(output)
        for separator in ("/", "?", "#", " ", "\n", "\r", "\t"):
            candidate = output.find(separator, value_start)
            if candidate >= 0:
                value_end = min(value_end, candidate)
        output = output[:value_start] + "<redacted>" + output[value_end:]
        start = output.lower().find(marker, value_start + len("<redacted>"))
    return output


def _safe_error_message(exc: BaseException, *, token: str) -> str:
    return str(redact_secrets(f"{type(exc).__name__}: {exc}", secrets=(token,)))


@dataclass(frozen=True)
class TushareProxyConfig:
    token: str
    url: str = TUSHARE_PROXY_DEFAULT_URL
    safe_rate_per_minute: int = TUSHARE_PROXY_SAFE_RATE_PER_MINUTE
    burst: int = TUSHARE_PROXY_BURST
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 120.0
    retries: int = 3

    @classmethod
    def from_env(cls) -> "TushareProxyConfig":
        return cls(
            token=str(os.environ.get(TUSHARE_PROXY_TOKEN_ENV, "") or "").strip(),
            url=str(os.environ.get(TUSHARE_PROXY_URL_ENV, TUSHARE_PROXY_DEFAULT_URL) or TUSHARE_PROXY_DEFAULT_URL).strip(),
        ).validated()

    def validated(self) -> "TushareProxyConfig":
        token = str(self.token or "").strip()
        if not token:
            raise TushareProxyError(f"{TUSHARE_PROXY_TOKEN_ENV}_not_configured")
        url = str(self.url or TUSHARE_PROXY_DEFAULT_URL).strip().rstrip("/")
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise TushareProxyError("tushare_proxy_url_must_be_https")
        if token in url or "token=" in url.lower():
            raise TushareProxyError("tushare_proxy_url_must_not_embed_token")
        rate = int(self.safe_rate_per_minute)
        burst = int(self.burst)
        retries = int(self.retries)
        if rate < 1 or rate > 150:
            raise ValueError(f"tushare_proxy_rate_out_of_range:{rate}")
        if burst < 1 or burst > 4:
            raise ValueError(f"tushare_proxy_burst_out_of_range:{burst}")
        if retries < 0 or retries > 3:
            raise ValueError(f"tushare_proxy_retries_out_of_range:{retries}")
        return TushareProxyConfig(
            token=token,
            url=url,
            safe_rate_per_minute=rate,
            burst=burst,
            connect_timeout_seconds=float(self.connect_timeout_seconds),
            read_timeout_seconds=float(self.read_timeout_seconds),
            retries=retries,
        )

    @property
    def token_sha256(self) -> str:
        return token_fingerprint(self.token)

    def public_metadata(self) -> dict[str, Any]:
        return {
            "provider": TUSHARE_PROXY_NAME,
            "protocol": TUSHARE_PROXY_PROTOCOL,
            "endpoint": self.url,
            "upstream_provenance": "not_exposed",
            "production_role": "historical_bootstrap",
            "token_sha256": self.token_sha256,
            "safe_rate_per_minute": int(self.safe_rate_per_minute),
            "burst": int(self.burst),
            "stk_mins_daily_limit_default": TUSHARE_PROXY_DEFAULT_MINUTE_DAILY_LIMIT,
            "max_in_flight_default": TUSHARE_PROXY_DEFAULT_MAX_IN_FLIGHT,
        }


class TushareProxyRateLimiter:
    """Thread-safe token bucket shared by every worker in one process."""

    def __init__(
        self,
        *,
        rate_per_minute: int = TUSHARE_PROXY_SAFE_RATE_PER_MINUTE,
        burst: int = TUSHARE_PROXY_BURST,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rate_per_minute = int(rate_per_minute)
        self.capacity = float(int(burst))
        self._tokens = self.capacity
        self._updated_at = float(clock())
        self._blocked_until = self._updated_at
        self._clock = clock
        self._sleeper = sleeper
        self._guard = threading.Lock()
        self._request_count = 0
        self._next_rate_reduction_at = float("-inf")
        self._rate_reduction_count = 0

    def acquire(self) -> None:
        refill_per_second = float(self.rate_per_minute) / 60.0
        while True:
            wait_seconds = 0.0
            with self._guard:
                now = float(self._clock())
                if now < self._blocked_until:
                    wait_seconds = max(self._blocked_until - now, 0.001)
                    self._updated_at = now
                else:
                    elapsed = max(0.0, now - self._updated_at)
                    self._tokens = min(self.capacity, self._tokens + elapsed * refill_per_second)
                    self._updated_at = now
                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        self._request_count += 1
                        return
                    wait_seconds = max((1.0 - self._tokens) / refill_per_second, 0.001)
            self._sleeper(wait_seconds)

    def defer(self, seconds: float, *, rate_limited: bool = False) -> None:
        """Apply one gateway throttle response to every worker sharing this limiter."""

        delay = max(float(seconds), 0.0)
        with self._guard:
            now = float(self._clock())
            self._blocked_until = max(self._blocked_until, now + delay)
            if rate_limited:
                self._tokens = 0.0
                if now >= self._next_rate_reduction_at:
                    reduced = max(
                        TUSHARE_PROXY_MIN_ADAPTIVE_RATE_PER_MINUTE,
                        int(self.rate_per_minute * 0.9),
                    )
                    if reduced < self.rate_per_minute:
                        self.rate_per_minute = reduced
                        self._rate_reduction_count += 1
                    self._next_rate_reduction_at = (
                        now + TUSHARE_PROXY_RATE_REDUCTION_COOLDOWN_SECONDS
                    )

    def snapshot(self) -> dict[str, Any]:
        with self._guard:
            return {
                "rate_per_minute": self.rate_per_minute,
                "burst": int(self.capacity),
                "request_count": int(self._request_count),
                "available_tokens": float(self._tokens),
                "blocked_for_seconds": max(0.0, float(self._blocked_until - self._clock())),
                "rate_reduction_count": int(self._rate_reduction_count),
            }


@dataclass(frozen=True)
class ProxyFrameResult:
    api_name: str
    frame: pd.DataFrame
    fields: tuple[str, ...]
    response_sha256: str
    elapsed_seconds: float
    attempts: int
    request_metadata_without_token: dict[str, Any]


@dataclass
class TushareProxyMetrics:
    request_count: int = 0
    success_count: int = 0
    network_error_count: int = 0
    protocol_error_count: int = 0
    rate_limit_count: int = 0
    latencies: deque[float] = field(default_factory=lambda: deque(maxlen=10_000))
    _guard: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def record(self, *, elapsed: float, status: str) -> None:
        with self._guard:
            self.request_count += 1
            self.latencies.append(float(elapsed))
            if status == "success":
                self.success_count += 1
            elif status == "rate_limit":
                self.rate_limit_count += 1
            elif status == "protocol_error":
                self.protocol_error_count += 1
            else:
                self.network_error_count += 1

    def snapshot(self) -> dict[str, Any]:
        with self._guard:
            values = sorted(self.latencies)
            p50 = median(values) if values else 0.0
            p95 = values[min(len(values) - 1, max(0, int(len(values) * 0.95) - 1))] if values else 0.0
            errors = self.network_error_count + self.protocol_error_count + self.rate_limit_count
            return {
                "request_count": int(self.request_count),
                "success_count": int(self.success_count),
                "network_error_count": int(self.network_error_count),
                "protocol_error_count": int(self.protocol_error_count),
                "rate_limit_count": int(self.rate_limit_count),
                "error_rate": float(errors / max(self.request_count, 1)),
                "p50_seconds": float(p50),
                "p95_seconds": float(p95),
            }


class TushareProxyClient:
    """Strict Tushare-compatible HTTP client with no SDK/global token state."""

    def __init__(
        self,
        config: TushareProxyConfig | None = None,
        *,
        limiter: TushareProxyRateLimiter | None = None,
        session_factory: Callable[[], requests.Session] = requests.Session,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = (config or TushareProxyConfig.from_env()).validated()
        self.limiter = limiter or TushareProxyRateLimiter(
            rate_per_minute=self.config.safe_rate_per_minute,
            burst=self.config.burst,
        )
        self.metrics = TushareProxyMetrics()
        self._session_factory = session_factory
        self._thread_local = threading.local()
        self._sleeper = sleeper
        # Three workers overlap connection and parsing work. Any throttle
        # response feeds a shared limiter cooldown so retries do not stampede.
        self._in_flight = threading.BoundedSemaphore(TUSHARE_PROXY_DEFAULT_MAX_IN_FLIGHT)

    def _session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = self._session_factory()
            session.headers.update(
                {
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "User-Agent": "QDP-v3-tushare-proxy-bootstrap/1",
                }
            )
            self._thread_local.session = session
        return session

    def close_thread_session(self) -> None:
        session = getattr(self._thread_local, "session", None)
        if session is not None:
            try:
                session.close()
            finally:
                self._thread_local.session = None

    def operational_metrics(self) -> dict[str, Any]:
        return {**self.metrics.snapshot(), "limiter": self.limiter.snapshot()}

    def fetch_frame(
        self,
        *,
        api_name: str,
        params: Mapping[str, Any] | None = None,
        fields: Iterable[str] | str = (),
    ) -> ProxyFrameResult:
        normalized_api = str(api_name or "").strip()
        if not normalized_api or not normalized_api.replace("_", "").isalnum():
            raise ValueError(f"tushare_proxy_invalid_api_name:{normalized_api!r}")
        if isinstance(fields, str):
            field_text = ",".join(item.strip() for item in fields.split(",") if item.strip())
        else:
            field_text = ",".join(str(item).strip() for item in fields if str(item).strip())
        public_request = {
            "api_name": normalized_api,
            "params": redact_secrets(dict(params or {}), secrets=(self.config.token,)),
            "fields": field_text,
        }
        body = {
            "api_name": normalized_api,
            "token": self.config.token,
            "params": dict(params or {}),
            "fields": field_text,
        }
        delays = (0.0, 2.0, 5.0, 15.0)
        last_error: BaseException | None = None
        attempts = int(self.config.retries) + 1
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                self._sleeper(delays[min(attempt - 1, len(delays) - 1)])
            self.limiter.acquire()
            started = time.monotonic()
            try:
                with self._in_flight:
                    response = self._session().post(
                        self.config.url,
                        json=body,
                        timeout=(self.config.connect_timeout_seconds, self.config.read_timeout_seconds),
                    )
                elapsed = time.monotonic() - started
                if int(getattr(response, "status_code", 0) or 0) in _RETRYABLE_HTTP_STATUS:
                    if int(response.status_code) == 429:
                        self.metrics.record(elapsed=elapsed, status="rate_limit")
                        headers = getattr(response, "headers", {}) or {}
                        retry_after = 2.0
                        try:
                            retry_after = max(2.0, min(float(headers.get("Retry-After", 2.0)), 60.0))
                        except (TypeError, ValueError):
                            retry_after = 2.0
                        self.limiter.defer(retry_after, rate_limited=True)
                    else:
                        self.metrics.record(elapsed=elapsed, status="network_error")
                    last_error = TushareProxyError(f"tushare_proxy_http_retryable:{response.status_code}")
                    continue
                response.raise_for_status()
                try:
                    payload = response.json()
                except Exception as exc:
                    self.metrics.record(elapsed=elapsed, status="protocol_error")
                    last_error = TushareProxyProtocolError(f"tushare_proxy_invalid_json:{type(exc).__name__}")
                    continue
                parsed = self._parse_payload(
                    payload,
                    api_name=normalized_api,
                    public_request=public_request,
                    elapsed=elapsed,
                    attempts=attempt,
                )
                self.metrics.record(elapsed=elapsed, status="success")
                return parsed
            except TushareProxyQuotaError:
                raise
            except TushareProxyProtocolError as exc:
                elapsed = time.monotonic() - started
                self.metrics.record(elapsed=elapsed, status="protocol_error")
                last_error = exc
                # Repeated schema errors are deterministic; retrying four
                # workers would only contaminate staging faster.
                break
            except Exception as exc:
                elapsed = time.monotonic() - started
                self.metrics.record(elapsed=elapsed, status="network_error")
                last_error = TushareProxyError(_safe_error_message(exc, token=self.config.token))
        if isinstance(last_error, TushareProxyError):
            raise last_error
        raise TushareProxyError("tushare_proxy_request_failed")

    def fetch_key_status(self) -> dict[str, Any]:
        """Read the public entitlement status and return an allow-listed view.

        The gateway exposes this as a GET endpoint. The credential necessarily
        appears in that one request URL, so no prepared request, response URL,
        or exception object is ever returned or logged.
        """

        parsed = urlparse(self.config.url)
        endpoint = f"{parsed.scheme}://{parsed.netloc}/api/key/status"
        started = time.monotonic()
        try:
            self.limiter.acquire()
            with self._in_flight:
                response = self._session().get(
                    endpoint,
                    params={"token": self.config.token},
                    timeout=(self.config.connect_timeout_seconds, self.config.read_timeout_seconds),
                )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise TushareProxyProtocolError("tushare_proxy_key_status_not_object")
            allowed = {
                "status",
                "expires_at",
                "remaining_days",
                "remaining_hours",
                "rate_limit",
                "plan",
                "tier",
                "is_active",
                "stk_mins_daily_limit",
                "max_concurrency",
            }
            result = {str(key): payload[key] for key in allowed if key in payload}
            result["token_sha256"] = self.config.token_sha256
            result.setdefault("stk_mins_daily_limit", TUSHARE_PROXY_DEFAULT_MINUTE_DAILY_LIMIT)
            result.setdefault("max_concurrency", TUSHARE_PROXY_DEFAULT_MAX_IN_FLIGHT)
            self.metrics.record(elapsed=time.monotonic() - started, status="success")
            return result
        except Exception as exc:
            self.metrics.record(elapsed=time.monotonic() - started, status="network_error")
            raise TushareProxyError(_safe_error_message(exc, token=self.config.token)) from None

    def _parse_payload(
        self,
        payload: Any,
        *,
        api_name: str,
        public_request: dict[str, Any],
        elapsed: float,
        attempts: int,
    ) -> ProxyFrameResult:
        if not isinstance(payload, Mapping):
            raise TushareProxyProtocolError("tushare_proxy_payload_not_object")
        code = payload.get("code")
        try:
            success = int(code) == 0
        except (TypeError, ValueError):
            success = False
        if not success:
            message = str(redact_secrets(payload.get("msg", ""), secrets=(self.config.token,)))
            lower = message.lower()
            if any(fragment in lower for fragment in ("quota", "limit", "频率", "次数", "额度", "过期", "expired")):
                raise TushareProxyQuotaError(f"tushare_proxy_quota_or_entitlement:{code}:{message[:240]}")
            raise TushareProxyProtocolError(f"tushare_proxy_api_error:{code}:{message[:240]}")
        data = payload.get("data")
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise TushareProxyProtocolError("tushare_proxy_data_not_object")
        raw_fields = data.get("fields") or []
        raw_items = data.get("items") or []
        if not isinstance(raw_fields, list) or not all(isinstance(item, str) and item for item in raw_fields):
            raise TushareProxyProtocolError("tushare_proxy_fields_invalid")
        if len(set(raw_fields)) != len(raw_fields):
            raise TushareProxyProtocolError("tushare_proxy_fields_duplicate")
        if not isinstance(raw_items, list):
            raise TushareProxyProtocolError("tushare_proxy_items_not_list")
        width = len(raw_fields)
        for index, row in enumerate(raw_items):
            if not isinstance(row, (list, tuple)) or len(row) != width:
                raise TushareProxyProtocolError(f"tushare_proxy_field_width_mismatch:{index}:{width}")
        frame = pd.DataFrame(raw_items, columns=raw_fields)
        response_sha = hashlib.sha256(
            json.dumps(
                {"fields": raw_fields, "items": raw_items},
                ensure_ascii=False,
                sort_keys=False,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        return ProxyFrameResult(
            api_name=api_name,
            frame=frame,
            fields=tuple(raw_fields),
            response_sha256=response_sha,
            elapsed_seconds=float(elapsed),
            attempts=int(attempts),
            request_metadata_without_token=dict(public_request),
        )

    def fetch_history_page(self, request: HistoryPageFetchRequest) -> HistoryPageResult:
        normalized = request.normalized()
        fields = (
            "ts_code",
            "trade_time",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "amount",
        )
        result = self.fetch_frame(
            api_name="stk_mins",
            params={
                "ts_code": normalized.provider_symbol,
                "freq": "5min",
                "start_date": normalized.start_at,
                "end_date": normalized.end_at,
                "limit": normalized.page_size,
            },
            fields=fields,
        )
        frame = result.frame
        if len(frame) > normalized.page_size:
            raise TushareProxyProtocolError(
                f"tushare_proxy_history_page_exceeds_limit:{len(frame)}:{normalized.page_size}"
            )
        if frame.empty:
            return HistoryPageResult(
                provider=TUSHARE_PROXY_NAME,
                request=normalized,
                raw_data=frame,
                fields=result.fields,
                row_count=0,
                min_timestamp="",
                max_timestamp="",
                next_end_at="",
                response_sha256=result.response_sha256,
                is_complete=True,
                request_metadata_without_token=result.request_metadata_without_token,
            )
        timestamp_column = next((item for item in _TIMESTAMP_ALIASES if item in frame.columns), "")
        if not timestamp_column:
            raise TushareProxyProtocolError("tushare_proxy_history_timestamp_field_missing")
        timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
        if timestamps.isna().any():
            raise TushareProxyProtocolError("tushare_proxy_history_timestamp_invalid")
        if timestamps.duplicated().any():
            raise TushareProxyProtocolError("tushare_proxy_history_page_duplicate_timestamp")
        values = timestamps.reset_index(drop=True)
        if len(values) > 1 and not values.is_monotonic_decreasing:
            raise TushareProxyProtocolError("tushare_proxy_history_not_reverse_chronological")
        start_ts = pd.Timestamp(normalized.start_at)
        end_ts = pd.Timestamp(normalized.end_at)
        min_ts = pd.Timestamp(timestamps.min())
        max_ts = pd.Timestamp(timestamps.max())
        if min_ts < start_ts or max_ts > end_ts:
            raise TushareProxyProtocolError(
                f"tushare_proxy_history_outside_request:{min_ts}:{max_ts}"
            )
        next_end = min_ts - pd.Timedelta(minutes=1)
        complete = bool(len(frame) < normalized.page_size or min_ts <= start_ts)
        return HistoryPageResult(
            provider=TUSHARE_PROXY_NAME,
            request=normalized,
            raw_data=frame,
            fields=result.fields,
            row_count=int(len(frame)),
            min_timestamp=min_ts.strftime("%Y-%m-%d %H:%M:%S"),
            max_timestamp=max_ts.strftime("%Y-%m-%d %H:%M:%S"),
            next_end_at="" if complete else next_end.strftime("%Y-%m-%d %H:%M:%S"),
            response_sha256=result.response_sha256,
            is_complete=complete,
            request_metadata_without_token=result.request_metadata_without_token,
        )


@dataclass
class TushareProxyProvider:
    """Limited-life historical bootstrap provider for QDP v3."""

    name: str = TUSHARE_PROXY_NAME
    client: TushareProxyClient | None = None

    def __post_init__(self) -> None:
        validate_provider_name(self.name)
        if self.client is None:
            self.client = TushareProxyClient()

    @property
    def public_metadata(self) -> dict[str, Any]:
        assert self.client is not None
        return self.client.config.public_metadata()

    def fetch_history_page(self, request: HistoryPageFetchRequest) -> HistoryPageResult:
        assert self.client is not None
        return self.client.fetch_history_page(request)

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        normalized = request.normalized()
        assert self.client is not None
        frames: list[pd.DataFrame] = []
        for symbol in normalized.symbols:
            result = self.client.fetch_frame(
                api_name="daily",
                params={
                    "ts_code": symbol,
                    "start_date": normalized.start_date.replace("-", ""),
                    "end_date": normalized.end_date.replace("-", ""),
                },
                fields="ts_code,trade_date,open,high,low,close,pre_close,vol,amount",
            )
            if not result.frame.empty:
                frames.append(result.frame)
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not raw.empty:
            raw = raw.rename(columns={"ts_code": "symbol", "vol": "volume"})
            raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce") * 100.0
            raw["amount"] = pd.to_numeric(raw["amount"], errors="coerce") * 1_000.0
        data = normalize_market_frame(
            raw,
            source=self.name,
            adjusted_flag=normalized.adjusted_flag,
            require_columns=False,
        )
        return ProviderResult(provider=self.name, data=data)

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        normalized = request.normalized()
        assert self.client is not None
        if normalized.domain == DataDomain.MARKET_DAILY:
            return self.fetch_market_bars(
                FetchRequest(
                    symbols=normalized.symbols,
                    start_date=normalized.start_date,
                    end_date=normalized.end_date,
                    adjusted_flag=normalized.adjusted_flag,
                )
            )
        if normalized.domain == DataDomain.TRADING_CALENDAR:
            raw = self.client.fetch_frame(
                api_name="trade_cal",
                params={
                    "start_date": normalized.start_date.replace("-", ""),
                    "end_date": normalized.end_date.replace("-", ""),
                },
                fields="exchange,cal_date,is_open,pretrade_date",
            ).frame.rename(columns={"cal_date": "trade_date"})
        elif normalized.domain == DataDomain.UNIVERSE_SNAPSHOT:
            frames = []
            for status in ("L", "D", "P"):
                frame = self.client.fetch_frame(
                    api_name="stock_basic",
                    params={"list_status": status},
                    fields="ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date,is_hs",
                ).frame
                frames.append(frame)
            raw = pd.concat(frames, ignore_index=True).rename(columns={"ts_code": "symbol", "market": "board"})
            raw["trade_date"] = normalized.end_date
        elif normalized.domain == DataDomain.VALUATION:
            frames = []
            for trade_date in pd.bdate_range(normalized.start_date, normalized.end_date):
                frame = self.client.fetch_frame(
                    api_name="daily_basic",
                    params={"trade_date": trade_date.strftime("%Y%m%d")},
                    fields="ts_code,trade_date,turnover_rate,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,total_share,float_share,free_share,total_mv,circ_mv",
                ).frame
                frames.append(frame)
            raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            raw = raw.rename(columns={"ts_code": "symbol"})
        elif normalized.domain == DataDomain.LIMIT_STATUS:
            frames = []
            for trade_date in pd.bdate_range(normalized.start_date, normalized.end_date):
                frame = self.client.fetch_frame(
                    api_name="stk_limit",
                    params={"trade_date": trade_date.strftime("%Y%m%d")},
                    fields="trade_date,ts_code,pre_close,up_limit,down_limit",
                ).frame
                frames.append(frame)
            raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            raw = raw.rename(columns={"ts_code": "symbol"})
        elif normalized.domain == DataDomain.ADJUST_FACTOR:
            frames = []
            for symbol in normalized.symbols:
                frame = self.client.fetch_frame(
                    api_name="adj_factor",
                    params={
                        "ts_code": symbol,
                        "start_date": normalized.start_date.replace("-", ""),
                        "end_date": normalized.end_date.replace("-", ""),
                    },
                    fields="ts_code,trade_date,adj_factor",
                ).frame
                frames.append(frame)
            raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            raw = raw.rename(columns={"ts_code": "symbol", "adj_factor": "adjust_factor"})
        else:
            raise RuntimeError(f"unsupported_domain:{self.name}:{normalized.domain}")
        data = normalize_domain_frame(
            raw,
            domain=normalized.domain,
            source=self.name,
            as_of_date=normalized.end_date,
            adjusted_flag=normalized.adjusted_flag,
            require_columns=False,
        )
        return ProviderResult(provider=self.name, data=data)


def history_page_result_to_receipt(result: HistoryPageResult) -> dict[str, Any]:
    return {
        "provider": result.provider,
        "endpoint": "stk_mins",
        "request": redact_secrets(result.request_metadata_without_token),
        "fields": list(result.fields),
        "row_count": int(result.row_count),
        "min_timestamp": result.min_timestamp,
        "max_timestamp": result.max_timestamp,
        "next_end_at": result.next_end_at,
        "response_sha256": result.response_sha256,
        "is_complete": bool(result.is_complete),
    }
