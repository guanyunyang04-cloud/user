from __future__ import annotations

import re
from typing import Any


_MISSING_TEXT = {"", "<NA>", "NAN", "NAT", "NONE"}
_WHITESPACE = re.compile(r"[\s\u3000]+")
_ST_PREFIX = re.compile(r"^(?:\*?ST|S\*?ST|G\*?ST)")


def normalize_security_name(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    if text in _MISSING_TEXT:
        return ""
    return _WHITESPACE.sub("", text)


def st_status_from_name(value: Any) -> bool | None:
    """Return the exchange-visible ST state, preserving a missing name."""

    name = normalize_security_name(value)
    if not name:
        return None
    return bool(_ST_PREFIX.match(name))


def security_name_implies_st(value: Any) -> bool:
    return st_status_from_name(value) is True
