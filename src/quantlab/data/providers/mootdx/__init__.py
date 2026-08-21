"""Mootdx provider package."""

# Compatibility facade for the former single-module implementation.
# ruff: noqa: F401

from .frames import (
    _call_mootdx_bars_endpoint,
    _estimated_mootdx_pages,
    _fetch_mootdx_bars_window,
    _filter_domain_date_window,
    _mootdx_payload_frame,
    _mootdx_xdxr_domain_frame,
    _normalize_mootdx_quote_snapshot,
    _prepare_mootdx_bars_frame,
)
from .provider import (
    MootdxOnlineProvider,
)
from .transport import (
    _MOOTDX_PROTOCOL_CACHE_AT,
    _MOOTDX_PROTOCOL_CACHE_LOCK,
    _MOOTDX_PROTOCOL_ERROR,
    _MOOTDX_PROTOCOL_SERVERS,
    _MOOTDX_REACHABLE_CACHE,
    _MOOTDX_REACHABLE_CACHE_AT,
    _MOOTDX_REACHABLE_CACHE_LOCK,
    _cache_mootdx_protocol_failure,
    _clear_mootdx_protocol_cache,
    _close_mootdx_client,
    _mootdx_complete_5m_probe,
    _mootdx_hq_server_candidates,
    _mootdx_last_good_path,
    _mootdx_persisted_last_good_servers,
    _mootdx_protocol_cache_snapshot,
    _mootdx_protocol_servers_snapshot,
    _mootdx_reachable_server_candidates,
    _open_mootdx_client,
    _persist_mootdx_last_good_servers,
    _probe_mootdx_protocol_clients,
)
