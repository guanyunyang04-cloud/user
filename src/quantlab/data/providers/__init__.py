"""Market-data providers grouped by source with a stable public import path."""

# This module intentionally re-exports the former monolithic provider API.
# ruff: noqa: F401

import requests

from quantlab.data.provider_symbols import (
    baostock_board as _baostock_board,
)
from quantlab.data.provider_symbols import (
    baostock_exchange as _baostock_exchange,
)
from quantlab.data.provider_symbols import (
    baostock_name_is_st as _baostock_name_is_st,
)
from quantlab.data.provider_symbols import (
    from_baostock_code as _from_baostock_code,
)
from quantlab.data.provider_symbols import (
    from_tencent_code as _from_tencent_code,
)
from quantlab.data.provider_symbols import (
    is_baostock_a_share_code as _is_baostock_a_share_code,
)
from quantlab.data.provider_symbols import (
    is_mootdx_index_symbol as _is_mootdx_index_symbol,
)
from quantlab.data.provider_symbols import (
    mootdx_symbol as _mootdx_symbol,
)
from quantlab.data.provider_symbols import (
    strip_suffix as _strip_suffix,
)
from quantlab.data.provider_symbols import (
    to_baostock_code as _to_baostock_code,
)
from quantlab.data.provider_symbols import (
    to_tencent_simple_code as _to_tencent_simple_code,
)

from .baostock import (
    _BAOSTOCK_GLOBAL_LIMITER,
    BAOSTOCK_BATCH_VERSION,
    BAOSTOCK_BULK_PER_PAGE_COUNT,
    BAOSTOCK_FAKE_IP_NETWORK,
    BaostockProvider,
    _baostock_adjust_factor_frame_from_bs,
    _baostock_adjust_factor_worker,
    _baostock_all_stock_frame,
    _baostock_all_stock_raw_frame,
    _baostock_all_stock_worker,
    _baostock_bulk_adjust_factor_event_frame,
    _baostock_bulk_daily_domain_frame,
    _baostock_bulk_partition_worker,
    _baostock_bulk_query_to_frame,
    _baostock_financial_quarterly_frame_from_bs,
    _baostock_financial_quarterly_worker,
    _baostock_history_frame,
    _baostock_history_worker,
    _baostock_index_constituents_frame,
    _baostock_index_constituents_worker,
    _baostock_industry_frame,
    _baostock_industry_worker,
    _baostock_intraday_5m_frame,
    _baostock_intraday_5m_worker,
    _baostock_performance_frame_from_bs,
    _baostock_performance_worker,
    _baostock_persistent_session_worker,
    _baostock_persistent_symbol_frames,
    _baostock_query_to_frame,
    _baostock_query_to_frame_with_relogin,
    _baostock_security_lifecycle_history_frame,
    _baostock_status_frame_from_all_stock,
    _baostock_stock_basic_frame,
    _baostock_stock_basic_worker,
    _baostock_symbol_error,
    _baostock_trade_calendar_worker,
    _baostock_valuation_frame_from_history,
    _baostock_valuation_symbol_frame_with_relogin,
    _baostock_valuation_worker,
    _BaostockGlobalLimiter,
    _BaostockPersistentSession,
    _configure_baostock_endpoint,
    _exact_column,
    _fetch_baostock_adjust_factor_frame_with_timeout,
    _fetch_baostock_all_stock_frame_with_timeout,
    _fetch_baostock_bulk_partition_once,
    _fetch_baostock_bulk_partition_with_retry,
    _fetch_baostock_financial_quarterly_frame_with_timeout,
    _fetch_baostock_history_frame_with_timeout,
    _fetch_baostock_index_constituents_frame_with_timeout,
    _fetch_baostock_industry_frame_with_timeout,
    _fetch_baostock_intraday_5m_frame_with_timeout,
    _fetch_baostock_payload_with_timeout,
    _fetch_baostock_performance_frame_with_timeout,
    _fetch_baostock_stock_basic_frame_with_timeout,
    _fetch_baostock_trade_calendar_frame_with_timeout,
    _fetch_baostock_valuation_frame_with_timeout,
    _is_baostock_not_logged_in_error,
    _quarter_points,
    _quiet_baostock_call,
    assert_baostock_batch_runtime,
    baostock_global_limiter_snapshot,
    baostock_runtime_version,
    configure_baostock_global_concurrency,
)
from .cninfo import (
    _CNINFO_ORG_MAP,
    _CNINFO_ORG_MAP_LOCK,
    CninfoAnnouncementProvider,
    _cninfo_org_map,
    _fetch_cninfo_announcements,
    _fetch_cninfo_org_id,
)
from .mootdx import (
    _MOOTDX_PROTOCOL_CACHE_AT,
    _MOOTDX_PROTOCOL_CACHE_LOCK,
    _MOOTDX_PROTOCOL_ERROR,
    _MOOTDX_PROTOCOL_SERVERS,
    _MOOTDX_REACHABLE_CACHE,
    _MOOTDX_REACHABLE_CACHE_AT,
    _MOOTDX_REACHABLE_CACHE_LOCK,
    MootdxOnlineProvider,
    _cache_mootdx_protocol_failure,
    _call_mootdx_bars_endpoint,
    _clear_mootdx_protocol_cache,
    _close_mootdx_client,
    _estimated_mootdx_pages,
    _fetch_mootdx_bars_window,
    _filter_domain_date_window,
    _mootdx_complete_5m_probe,
    _mootdx_hq_server_candidates,
    _mootdx_last_good_path,
    _mootdx_payload_frame,
    _mootdx_persisted_last_good_servers,
    _mootdx_protocol_cache_snapshot,
    _mootdx_protocol_servers_snapshot,
    _mootdx_reachable_server_candidates,
    _mootdx_xdxr_domain_frame,
    _normalize_mootdx_quote_snapshot,
    _open_mootdx_client,
    _persist_mootdx_last_good_servers,
    _prepare_mootdx_bars_frame,
    _probe_mootdx_protocol_clients,
)
from .registry import (
    _PROVIDER_CAPABILITIES,
    FORMAL_FREE_V3_OPTIONAL_DOMAINS,
    FORMAL_FREE_V3_REQUIRED_DOMAINS,
    FORMAL_FREE_V3_RESEARCH_FUTURE_DOMAINS,
    QDP_CURRENT_REQUIRED_DOMAINS,
    QDP_PRODUCTION_V1_OPTIONAL_DOMAINS,
    QDP_PRODUCTION_V1_REQUIRED_DOMAINS,
    QDP_PRODUCTION_V1_RESEARCH_FUTURE_DOMAINS,
    RESEARCH_REBUILD_MINIMAL_REQUIRED_DOMAINS,
    QdpProductionV1Provider,
    ResearchRebuildMinimalFreeProvider,
    _formal_requirement,
    build_default_providers,
    provider_capability_matrix,
)
from .web import (
    AkshareEastmoneyProvider,
    EastmoneyEfinanceProvider,
    SinaTencentRealtimeProvider,
    TencentFinanceProvider,
    TonghuashunHotspotProvider,
    _akshare_industry_members,
    _akshare_limit_status,
    _akshare_money_flow,
    _date_range_strings,
    _ths_concept_frame,
    _ths_hotspot_frame,
)
