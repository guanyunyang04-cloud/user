"""Universe rules for the first traditional quant research dataset."""

from __future__ import annotations

import re
from typing import Iterable


SH_MAINBOARD_PREFIXES = ("600", "601", "603", "605")
SZ_MAINBOARD_PREFIXES = ("000", "001", "002", "003")
STOCK_CODE_RE = re.compile(r"^\d{6}\.(SH|SZ|BJ)$")
ST_NAME_MARKERS = ("ST", "*ST", "S*ST", "S ST")


def is_sh_sz_mainboard_a_share(code: str) -> bool:
    """Return True for Shanghai/Shenzhen mainboard A-share common-stock codes."""
    normalized = str(code).strip().upper()
    if not STOCK_CODE_RE.match(normalized):
        return False

    symbol, market = normalized.split(".", 1)
    if market == "SH":
        return symbol.startswith(SH_MAINBOARD_PREFIXES)
    if market == "SZ":
        return symbol.startswith(SZ_MAINBOARD_PREFIXES)
    return False


def is_st_name(name: str) -> bool:
    """Return True when a security display name carries an ST marker."""
    normalized = str(name).strip().upper().replace(" ", "")
    return any(marker.replace(" ", "") in normalized for marker in ST_NAME_MARKERS)


def is_active_common_stock_info(info: dict[str, object]) -> bool:
    """Screen stock metadata for active non-ST common stocks when fields exist."""
    name = str(info.get("Name", "") or "")
    if is_st_name(name):
        return False
    if str(info.get("IsZS", "") or "0") == "1":
        return False
    if str(info.get("IsHKGP", "") or "0") == "1":
        return False
    if str(info.get("IsQH", "") or "0") == "1":
        return False
    if str(info.get("IsQQ", "") or "0") == "1":
        return False
    return True


def filter_sh_sz_mainboard_a_shares(codes: Iterable[str]) -> list[str]:
    """Filter and preserve order for Shanghai/Shenzhen mainboard A-share codes."""
    seen: set[str] = set()
    selected: list[str] = []
    for raw_code in codes:
        code = str(raw_code).strip().upper()
        if code in seen or not is_sh_sz_mainboard_a_share(code):
            continue
        seen.add(code)
        selected.append(code)
    return selected


# Backward-compatible aliases kept for early notebooks and scratch scripts.
is_sh_sz_a_share = is_sh_sz_mainboard_a_share
filter_sh_sz_a_shares = filter_sh_sz_mainboard_a_shares
