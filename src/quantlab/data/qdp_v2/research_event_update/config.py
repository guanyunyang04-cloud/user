"""Research Event Update: config responsibilities."""

from __future__ import annotations

UPDATE_ID = "research_report_rc_backfill_v2"


LEGACY_UPDATE_ID = "research_report_rc_backfill_v1"


START_DATE = "2010-01-01"


END_DATE = "2025-12-31"


REPORT_EASTMONEY_START = "2017-01-01"


REPORT_RC_PAGE_SIZE = 3_000


REPORT_RC_EMPTY_CONFIRMATIONS = 2


MAX_WORKERS = 3


REPORT_RC_FIELDS = (
    "ts_code",
    "name",
    "report_date",
    "report_title",
    "report_type",
    "classify",
    "org_name",
    "author_name",
    "quarter",
    "op_rt",
    "op_pr",
    "tp",
    "np",
    "eps",
    "pe",
    "rd",
    "roe",
    "ev_ebitda",
    "rating",
    "max_price",
    "min_price",
)


REPORT_COLUMNS = (
    "report_id",
    "source_report_key",
    "symbol",
    "trade_date",
    "source_date",
    "feature_available_date",
    "title",
    "normalized_title",
    "institution",
    "normalized_institution",
    "analyst",
    "report_type",
    "classification",
    "rating",
    "rating_change",
    "target_price_min",
    "target_price_max",
    "tushare_present",
    "eastmoney_present",
    "tushare_source_ids",
    "eastmoney_info_codes",
    "url",
    "pdf_file_size_kb",
    "pdf_pages",
    "source_disagreement",
    "identity_conflict_reason",
    "source",
)


FORECAST_COLUMNS = (
    "report_id",
    "symbol",
    "trade_date",
    "source_date",
    "feature_available_date",
    "forecast_quarter",
    "forecast_year",
    "operating_revenue",
    "operating_profit",
    "total_profit",
    "net_profit",
    "eps",
    "pe",
    "research_development",
    "roe",
    "ev_ebitda",
    "source_disagreement",
    "source",
)


ANNOUNCEMENT_COLUMNS = (
    "announcement_id",
    "symbol",
    "trade_date",
    "source_date",
    "feature_available_date",
    "publish_time",
    "title",
    "normalized_title",
    "category",
    "announcement_type_codes",
    "cninfo_announcement_id",
    "eastmoney_art_code",
    "org_id",
    "url",
    "pdf_url",
    "file_size_kb",
    "cninfo_present",
    "eastmoney_present",
    "source_disagreement",
    "source",
)


class ResearchEventUpdateError(RuntimeError):
    pass
