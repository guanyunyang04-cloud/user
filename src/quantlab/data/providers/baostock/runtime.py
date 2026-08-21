"""Baostock runtime responsibilities."""

from __future__ import annotations

import io
import ipaddress
import os
import socket
import threading
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any

import requests

BAOSTOCK_BATCH_VERSION = "0.9.3"


BAOSTOCK_BULK_PER_PAGE_COUNT = 20_000


BAOSTOCK_FAKE_IP_NETWORK = ipaddress.ip_network("198.18.0.0/15")


def _configure_baostock_endpoint() -> str:
    """Bypass proxy Fake-IP DNS for BaoStock's native TCP protocol."""

    from baostock.common import contants as baostock_constants  # type: ignore

    override = str(os.environ.get("QDP_BAOSTOCK_SERVER_IP", "") or "").strip()
    if override:
        ipaddress.ip_address(override)
        baostock_constants.BAOSTOCK_SERVER_IP = override
        return override
    host = str(baostock_constants.BAOSTOCK_SERVER_IP)
    try:
        resolved = socket.gethostbyname(host)
        address = ipaddress.ip_address(resolved)
    except (OSError, ValueError):
        return host
    if address not in BAOSTOCK_FAKE_IP_NETWORK:
        return resolved
    response = requests.get(
        "https://dns.google/resolve",
        params={"name": host, "type": "A"},
        timeout=10,
    )
    response.raise_for_status()
    answers = list(dict(response.json() or {}).get("Answer", []) or [])
    for answer in answers:
        candidate = str(dict(answer or {}).get("data", "") or "").strip()
        try:
            candidate_ip = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if candidate_ip.version == 4 and candidate_ip not in BAOSTOCK_FAKE_IP_NETWORK:
            baostock_constants.BAOSTOCK_SERVER_IP = candidate
            return candidate
    raise RuntimeError(f"baostock_fake_ip_dns_unresolved:{host}:{resolved}")


def _quiet_baostock_call(operation: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Keep provider banner text out of machine-readable CLI stdout/stderr."""

    if str(getattr(operation, "__name__", "")) == "login" and str(getattr(operation, "__module__", "")).startswith(
        "baostock"
    ):
        _configure_baostock_endpoint()
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return operation(*args, **kwargs)


class _BaostockGlobalLimiter:
    """Process-local limiter shared by every BaoStock endpoint.

    BaoStock sessions are stateful and its public service is sensitive to
    bursts. QDP therefore defaults to one in-flight request and never raises
    that ceiling automatically. A live two-login probe showed that BaoStock
    can invalidate one session when another logs in, even at a zero historical
    error rate.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active = 0
        self._limit = 1
        self._requests = 0
        self._network_errors = 0

    @contextmanager
    def slot(self):
        with self._condition:
            while self._active >= self._limit:
                self._condition.wait()
            self._active += 1
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            with self._condition:
                self._active -= 1
                self._requests += 1
                if failed:
                    self._network_errors += 1
                self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "active": self._active,
                "limit": self._limit,
                "requests": self._requests,
                "network_errors": self._network_errors,
                "network_error_rate": self._network_errors / max(self._requests, 1),
            }

    def configure_limit(self, limit: int) -> dict[str, Any]:
        normalized = int(limit)
        if normalized not in {1, 2}:
            raise ValueError("baostock_global_concurrency_must_be_1_or_2")
        with self._condition:
            self._limit = normalized
            self._condition.notify_all()
        return self.snapshot()


_BAOSTOCK_GLOBAL_LIMITER = _BaostockGlobalLimiter()


def configure_baostock_global_concurrency(max_workers: int) -> dict[str, Any]:
    """Set the process-wide BaoStock request ceiling to one or two slots.

    Production date-partition ingestion always configures one slot. The
    two-slot setting remains an explicit low-level test hook; it is never
    selected from historical error-rate statistics.
    """

    return _BAOSTOCK_GLOBAL_LIMITER.configure_limit(int(max_workers))


def baostock_global_limiter_snapshot() -> dict[str, Any]:
    return _BAOSTOCK_GLOBAL_LIMITER.snapshot()


def baostock_runtime_version() -> str:
    try:
        return str(package_version("baostock"))
    except PackageNotFoundError as exc:
        raise RuntimeError("baostock is not installed in the yolos environment") from exc


def assert_baostock_batch_runtime() -> str:
    installed = baostock_runtime_version()
    if installed != BAOSTOCK_BATCH_VERSION:
        raise RuntimeError(
            "baostock_batch_version_mismatch: "
            f"expected={BAOSTOCK_BATCH_VERSION} installed={installed}; "
            "batch data is forbidden from canonical staging"
        )
    return installed
