"""PIT history config operations."""

from __future__ import annotations

from pathlib import Path

from quantlab.data.qdp_v2 import normalization as _normalization
from quantlab.data.qdp_v2 import pit_normalization as _pit_normalization

CNINFO_SHARE_NORMALIZED_COLUMNS = _normalization.CNINFO_SHARE_NORMALIZED_COLUMNS


NAME_INTERVAL_COLUMNS = _normalization.NAME_INTERVAL_COLUMNS


_normalize_cninfo_share_change = _normalization.normalize_cninfo_share_change


_factor_rows = _pit_normalization.factor_rows


_historical_names = _pit_normalization.historical_names


_identity_exchange = _pit_normalization.identity_exchange


_is_mainboard = _pit_normalization.is_mainboard


_name_implies_st = _pit_normalization.name_implies_st


_normalize_eastmoney_history = _pit_normalization.normalize_eastmoney_history


_normalize_sina_factors = _pit_normalization.normalize_sina_factors


_security_id = _pit_normalization.security_id


_short_exchange = _pit_normalization.short_exchange


DEFAULT_START_DATE = "2010-01-04"


PART_SEMANTIC_VERSION = "pit_historical_mainboard_daily_nullable_security_status"


MAINBOARD_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")


ARCHIVE_ROOT = Path(
    "data/qdp/source_archives/baostock/traditional_quant_baostock_archive_v1/raw"
)


ARCHIVE_RECENT = ARCHIVE_ROOT / "baostock_daily_mainboard_v2_pit"


ARCHIVE_RECOVERY = ARCHIVE_ROOT / "baostock_daily_mainboard_v2_pit_recovery_2012_2015"


SSE_ST_TRANSITIONS = Path(__file__).resolve().parent.parent / "resources" / "sse_st_transitions_2010_2011.csv"


SSE_FACTBOOK_EVIDENCE = {
    "2011": {
        "url": "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170571/files/f43f33c247f242d48780c3097548281b.pdf",
    },
    "2012": {
        "url": "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170570/files/9a7b8e0d00e84d358d02fbba056f7bda.pdf",
    },
}


PRE_ARCHIVE_SECURITIES = (
    ("000578.SZ", "盐湖集团", "1995-03-03", "2011-03-22"),
    ("600003.SH", "ST东北高", "1999-08-10", "2010-02-26"),
    ("600553.SH", "太行水泥", "2002-08-22", "2011-02-18"),
    ("600591.SH", "*ST上航", "2002-10-11", "2010-01-25"),
    ("600607.SH", "上实医药", "1992-03-27", "2010-02-12"),
    ("600631.SH", "百联股份", "1993-02-19", "2011-08-23"),
    ("600842.SH", "中西药业", "1994-03-11", "2010-02-12"),
    ("600849.SH", "上药转换", "1994-03-24", "2010-03-05"),
)


RESTORE_DOMAINS = (
    "security_identity",
    "symbol_history",
    "market_daily_raw",
    "universe_snapshot",
    "security_status",
    "adjust_factor",
    "industry_concept",
    "share_capital",
    "valuation",
    "name_change",
)


LIFECYCLE_NORMALIZE_DOMAINS = (
    "market_daily_raw",
    "universe_snapshot",
    "security_status",
    "adjust_factor",
    "industry_concept",
    "share_capital",
    "valuation",
    "name_change",
    "corporate_actions",
)
