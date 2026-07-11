from __future__ import annotations

import importlib
from typing import Callable

from quant_data_platform.qdp_v2.environment import assert_yolos_environment


CommandMain = Callable[[list[str] | None], int]

HELP_TEXT = """usage: qdp [--workspace-root WORKSPACE_ROOT] <command> [options]

QDP local data base CLI.

commands:
  status                  Show active data base scope and table summary.
  list                    List active tables.
  describe <table>        Describe an active table or dataset id.
  check --quick           Fast manifest and contract check.
  check --full            Full data-quality audit.
  check meta              Deep proof for PIT/meta/factor/index domains.
  rebuild 5m              Rebuild 5m cache from active 1m bars.
  rebuild daily-panel     Rebuild daily raw/panel tables from an input panel.
  rebuild valuation       Rebuild normalized valuation table.
  rebuild adjust-factor   Rebuild standard daily adjustment factor table.
  rebuild industry-concept Rebuild complete industry table with UNKNOWN gaps.
  rebuild industry-concept-filled Fill UNKNOWN industry labels from adjacent known labels.
  rebuild share-capital-daily Rebuild daily PIT share-capital table.
  rebuild valuation-market-cap Rebuild valuation with market-cap fields.
  rebuild index-constituents-daily Rebuild daily PIT index constituents.
  rebuild pit-signal-universe Build date-local main-board non-ST signal eligibility and daily hashes.
  rebuild pit-market-substrate Build a non-active PIT daily market/factor/status/limit dataset view.
  rebuild scope-active    Rebuild active tables under the current mainboard non-delisted scope.
  rebuild limit-intraday  Rebuild 1m-derived limit-board features.
  rebuild training-pack   Build research training pack from active data.
  verify pit-market-view  Verify all five atomic overrides in a non-active PIT dataset view.
  gc --dry-run            Show unreferenced data directories.
  update                  Update the active data base.

The active data base is manifest-first: parquet + dataset.json + active.json.
Indexes, memmaps, and archived repair tools are not part of active state.
"""

REBUILD_HELP_TEXT = """usage: qdp rebuild <target> [options]

Rebuild cache and feature tables that are reproducible from active raw data.

targets:
  5m                 Rebuild 48-bar 5m cache from active 1m data.
  daily-panel        Rebuild daily raw/panel tables from an input panel.
  valuation          Rebuild normalized valuation table.
  adjust-factor      Rebuild one-row-per-symbol-day standard adjustment factors.
  industry-concept   Rebuild industry table aligned to universe.
  industry-concept-filled Fill UNKNOWN industry labels from same-symbol known labels.
  share-capital-daily Rebuild one-row-per-symbol-day share-capital facts.
  valuation-market-cap Rebuild market-cap fields from close and share capital.
  index-constituents-daily Expand index snapshots to daily PIT membership.
  pit-signal-universe Build date-local main-board non-ST signal eligibility and daily hashes.
  pit-market-substrate Build non-active market_daily_raw, security_status, limit_status,
                       adjust_factor, and an explicit research dataset view.
  scope-active        Rebuild all active symbol tables under current scope.
  limit-intraday     Rebuild 1m-derived limit-board features.
  training-pack       Build sharded memmap and training pack from active data.
"""


COMMAND_MODULES: dict[tuple[str, ...], str] = {
    ("status",): "quant_data_platform.qdp_v2.status",
    ("list",): "quant_data_platform.qdp_v2.dataset",
    ("describe",): "quant_data_platform.qdp_v2.dataset",
    ("check", "meta"): "quant_data_platform.qdp_v2.meta_quality",
    ("check",): "quant_data_platform.qdp_v2.check",
    ("gc",): "quant_data_platform.qdp_v2.gc",
    ("update",): "quant_data_platform.qdp_v2.update",
    ("rebuild", "5m"): "quant_data_platform.qdp_v2.cleaning",
    ("rebuild", "daily-panel"): "quant_data_platform.qdp_v2.cleaning",
    ("rebuild", "valuation"): "quant_data_platform.qdp_v2.cleaning",
    ("rebuild", "adjust-factor"): "quant_data_platform.qdp_v2.meta_quality",
    ("rebuild", "industry-concept"): "quant_data_platform.qdp_v2.meta_quality",
    ("rebuild", "industry-concept-filled"): "quant_data_platform.qdp_v2.completion",
    ("rebuild", "share-capital-daily"): "quant_data_platform.qdp_v2.completion",
    ("rebuild", "valuation-market-cap"): "quant_data_platform.qdp_v2.completion",
    ("rebuild", "index-constituents-daily"): "quant_data_platform.qdp_v2.completion",
    ("rebuild", "pit-signal-universe"): "quant_data_platform.qdp_v2.completion",
    ("rebuild", "pit-market-substrate"): "quant_data_platform.qdp_v2.pit_market_substrate",
    ("verify", "pit-market-view"): "quant_data_platform.qdp_v2.pit_market_substrate",
    ("rebuild", "scope-active"): "quant_data_platform.qdp_v2.completion",
    ("rebuild", "limit-intraday"): "quant_data_platform.qdp_v2.limit_intraday_features",
    ("rebuild", "training-pack"): "quant_data_platform.qdp_v2.training_pack",
}

ENV_GUARDED_PREFIXES = {
    ("check",),
    ("rebuild", "5m"),
    ("rebuild", "daily-panel"),
    ("rebuild", "valuation"),
    ("rebuild", "adjust-factor"),
    ("rebuild", "industry-concept"),
    ("rebuild", "industry-concept-filled"),
    ("rebuild", "share-capital-daily"),
    ("rebuild", "valuation-market-cap"),
    ("rebuild", "index-constituents-daily"),
    ("rebuild", "pit-signal-universe"),
    ("rebuild", "pit-market-substrate"),
    ("rebuild", "scope-active"),
    ("rebuild", "limit-intraday"),
    ("rebuild", "training-pack"),
}

ARG_ALIASES: dict[tuple[str, ...], list[str]] = {
    ("list",): ["list"],
    ("describe",): ["describe"],
    ("rebuild", "5m"): ["5m-from-1m"],
    ("rebuild", "daily-panel"): ["daily-market"],
    ("rebuild", "valuation"): ["valuation"],
    ("rebuild", "adjust-factor"): ["rebuild-adjust-factor"],
    ("rebuild", "industry-concept"): ["rebuild-industry-concept"],
    ("rebuild", "industry-concept-filled"): ["industry-concept-filled"],
    ("rebuild", "share-capital-daily"): ["share-capital-daily"],
    ("rebuild", "valuation-market-cap"): ["valuation-market-cap"],
    ("rebuild", "index-constituents-daily"): ["index-constituents-daily"],
    ("rebuild", "pit-signal-universe"): ["pit-signal-universe"],
    ("rebuild", "pit-market-substrate"): ["build"],
    ("verify", "pit-market-view"): ["verify-view"],
    ("rebuild", "scope-active"): ["scope-active"],
    ("check", "meta"): ["audit"],
}


def dispatch(argv: list[str]) -> int | None:
    raw = list(argv or [])
    if not raw or raw in (["-h"], ["--help"]):
        print(HELP_TEXT)
        return 0
    if raw[0] == "rebuild" and (len(raw) == 1 or raw[1] in {"-h", "--help"}):
        print(REBUILD_HELP_TEXT)
        return 0
    for prefix, module_name in sorted(COMMAND_MODULES.items(), key=lambda item: len(item[0]), reverse=True):
        if tuple(raw[: len(prefix)]) == prefix:
            if _requires_yolos(prefix, raw[len(prefix) :]):
                assert_yolos_environment(command=" ".join(prefix))
            module = importlib.import_module(module_name)
            main = getattr(module, "main", None)
            if not callable(main):
                raise RuntimeError(f"{module_name} does not expose callable main(argv)")
            forwarded = list(ARG_ALIASES.get(prefix, [])) + raw[len(prefix) :]
            return int(main(forwarded) or 0)
    return None


def _requires_yolos(prefix: tuple[str, ...], args: list[str]) -> bool:
    if prefix in ENV_GUARDED_PREFIXES:
        return True
    if prefix == ("update",):
        return "--dry-run" not in args
    if prefix == ("gc",):
        return "--delete" in args
    return False


def main(argv: list[str] | None = None) -> int:
    result = dispatch(list(argv or []))
    if result is None:
        raise ValueError("unsupported_qdp_v2_command")
    return result
