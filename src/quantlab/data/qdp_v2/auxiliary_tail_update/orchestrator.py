"""Auxiliary Tail Update: orchestrator responsibilities."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.qdp_v2.auxiliary_update import (
    AUXILIARY_DOMAINS,
    _context,
)

from .common import (
    _checked_through,
)
from .corporate import (
    update_corporate_actions_tail,
)
from .index import (
    update_index_tail,
)
from .industry import (
    update_industry_tail,
)
from .names import (
    update_name_change_tail,
)
from .shares import (
    _fetch_mootdx_evidence,
    update_share_capital_tail,
)
from .valuation import (
    update_valuation_tail,
)


def run_auxiliary_tail_update(
    *,
    as_of_date: str,
    workspace_root: str | Path | None = None,
    domains: Sequence[str] = AUXILIARY_DOMAINS,
    baostock_valuation_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    ctx = _context(as_of_date=as_of_date, workspace_root=workspace_root)
    selected = tuple(str(item) for item in domains)
    unknown = sorted(set(selected).difference(AUXILIARY_DOMAINS))
    if unknown:
        raise ValueError(f"unknown_auxiliary_domains:{','.join(unknown)}")
    results: dict[str, Any] = {}
    if "industry_concept" in selected:
        results["industry_concept"] = update_industry_tail(ctx)
    if "name_change" in selected:
        results["name_change"] = update_name_change_tail(ctx)
    if "index_constituents" in selected:
        results["index_constituents"] = update_index_tail(ctx)

    needs_xdxr = bool(set(selected).intersection({"share_capital", "corporate_actions"}))
    share_events = pd.DataFrame()
    corporate_events = pd.DataFrame()
    if needs_xdxr:
        start = min(
            _checked_through(ctx, domain) for domain in ("share_capital", "corporate_actions") if domain in selected
        )
        share_events, corporate_events = _fetch_mootdx_evidence(
            ctx,
            start_date=start,
        )
    if "share_capital" in selected:
        results["share_capital"] = update_share_capital_tail(
            ctx,
            detected_events=share_events,
        )
    if "valuation" in selected:
        results["valuation"] = update_valuation_tail(
            ctx,
            baostock_valuation_cache_path=baostock_valuation_cache_path,
        )
    if "corporate_actions" in selected:
        results["corporate_actions"] = update_corporate_actions_tail(
            ctx,
            detected_events=corporate_events,
        )
    (ctx.runtime / "mootdx_last_good_5m.json").unlink(missing_ok=True)
    if baostock_valuation_cache_path is not None:
        Path(baostock_valuation_cache_path).resolve().unlink(missing_ok=True)
    return {
        "status": "updated",
        "as_of_date": ctx.target_date,
        "domains": results,
        "provider_policy": "baostock+mootdx_detection+cninfo_confirmation; no_tushare",
    }
