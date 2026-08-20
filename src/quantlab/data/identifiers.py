"""Stable A-share identity and exchange mappings."""

from __future__ import annotations


def security_id(symbol: str) -> str:
    code, suffix = str(symbol).strip().upper().split(".", 1)
    exchange = {"SH": "SSE", "SZ": "SZSE", "BJ": "BSE"}.get(suffix, suffix)
    return f"QDP-CN-{exchange}-{code}"


def identity_exchange(symbol: str) -> str:
    return {"SH": "SSE", "SZ": "SZSE", "BJ": "BSE"}.get(
        str(symbol).strip().upper()[-2:], ""
    )


def short_exchange(symbol: str) -> str:
    return str(symbol).strip().upper()[-2:]


__all__ = ["identity_exchange", "security_id", "short_exchange"]
