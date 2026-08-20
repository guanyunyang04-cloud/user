"""Canonical symbol conversions shared by market-data providers."""

from __future__ import annotations

from typing import Any

from quantlab.data.core.security_status import st_status_from_name


def strip_suffix(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    if "." in raw:
        return raw.split(".", 1)[0]
    if raw.startswith(("SH", "SZ", "BJ")) and raw[2:].isdigit():
        return raw[2:]
    return raw


def mootdx_symbol(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    if "." in raw:
        raw = raw.split(".", 1)[0]
    if raw.startswith(("SH", "SZ", "BJ")) and raw[2:].isdigit():
        raw = raw[2:]
    return raw[-6:].zfill(6)


def is_mootdx_index_symbol(symbol: str) -> bool:
    raw = str(symbol or "").strip().upper()
    code = mootdx_symbol(raw)
    suffix = raw.rsplit(".", 1)[1] if "." in raw else ""
    if suffix == "SH" and code.startswith(
        ("000", "880", "881", "882", "883", "884", "885", "886", "887", "889")
    ):
        return True
    if suffix == "SZ" and code.startswith("399"):
        return True
    return False


def to_baostock_code(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    if raw.endswith(".SH"):
        return f"sh.{raw[:6]}"
    if raw.endswith(".SZ"):
        return f"sz.{raw[:6]}"
    if raw.endswith(".BJ"):
        return f"bj.{raw[:6]}"
    if raw.startswith(("5", "6", "9")):
        return f"sh.{raw[:6]}"
    if raw.startswith(("4", "8")):
        return f"bj.{raw[:6]}"
    return f"sz.{raw[:6]}"


def to_tencent_simple_code(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    code = raw.split(".", 1)[0] if "." in raw else raw[-6:]
    exchange = (
        raw.split(".", 1)[1]
        if "." in raw
        else ("SH" if code.startswith(("5", "6", "9")) else "SZ")
    )
    prefix = "sh" if exchange == "SH" else "sz"
    return f"s_{prefix}{code}"


def from_tencent_code(value: Any) -> str:
    raw = str(value or "").strip().lower().removeprefix("s_")
    if raw.startswith("sh"):
        return f"{raw[2:8].upper()}.SH"
    if raw.startswith("sz"):
        return f"{raw[2:8].upper()}.SZ"
    return str(value or "").strip().upper()


def from_baostock_code(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith(("sh.", "sz.", "bj.")):
        return f"{raw[3:].upper()}.{raw[:2].upper()}"
    return str(value or "").strip().upper()


def is_baostock_a_share_code(value: Any) -> bool:
    raw = str(value or "").strip().lower()
    if raw.startswith("sh."):
        code = raw[3:9]
        return code.startswith(("600", "601", "603", "605", "688"))
    if raw.startswith("sz."):
        code = raw[3:9]
        return code.startswith(("000", "001", "002", "003", "300", "301"))
    if raw.startswith("bj."):
        return raw[3:9].isdigit()
    return False


def baostock_exchange(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if raw.endswith(".SH"):
        return "SH"
    if raw.endswith(".SZ"):
        return "SZ"
    if raw.endswith(".BJ"):
        return "BJ"
    return ""


def baostock_board(value: Any) -> str:
    raw = str(value or "").strip().upper()
    code = raw.split(".", 1)[0]
    exchange = raw.split(".", 1)[1] if "." in raw else ""
    if exchange == "BJ":
        return "beijing"
    if exchange == "SH" and code.startswith("688"):
        return "star"
    if exchange == "SZ" and code.startswith(("300", "301")):
        return "chi_next"
    if exchange in {"SH", "SZ"}:
        return "main"
    return "unknown"


def baostock_name_is_st(value: Any) -> bool | None:
    return st_status_from_name(value)


__all__ = [
    "baostock_board",
    "baostock_exchange",
    "baostock_name_is_st",
    "from_baostock_code",
    "from_tencent_code",
    "is_baostock_a_share_code",
    "is_mootdx_index_symbol",
    "mootdx_symbol",
    "strip_suffix",
    "to_baostock_code",
    "to_tencent_simple_code",
]
