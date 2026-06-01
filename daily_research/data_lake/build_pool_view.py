from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.data_lake.catalog import ResearchDataLake
from daily_research.data_lake.pool_views import PoolViewSpec, build_pool_view_from_policy_bundle

ACTIVE_ARTIFACT = "daily_research/output/active_execution_strategy.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an auditable policy pool view from an existing policy input bundle.")
    parser.add_argument("--data-lake-root", default="")
    parser.add_argument("--source-market-dataset-id", required=True)
    parser.add_argument("--view-kind", required=True, choices=("learned_all_a", "rolling_liquidity", "exchange", "static_symbols"))
    parser.add_argument("--view-name", default="")
    parser.add_argument("--pool-name", default="")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--rebalance-every-days", type=int, default=21)
    parser.add_argument("--adv-window", type=int, default=20)
    parser.add_argument("--min-price", type=float, default=2.0)
    parser.add_argument("--max-price", type=float, default=300.0)
    parser.add_argument("--exchange-suffix", default="")
    parser.add_argument("--symbols", default="")
    parser.add_argument(
        "--exclude-symbol-prefixes",
        default="",
        help="Comma-separated stock-code prefixes to exclude before building the view, e.g. 300,301,688,689.",
    )
    parser.add_argument("--refresh", action="store_true")
    return parser


def _default_view_name(args: argparse.Namespace) -> str:
    explicit = str(args.view_name or "").strip().lower()
    if explicit:
        return explicit
    if args.view_kind == "rolling_liquidity":
        return f"rolling_{str(args.pool_name or '').strip().lower()}"
    if args.view_kind == "exchange":
        suffix = str(args.exchange_suffix or "").strip().lower().lstrip(".")
        return f"exchange_{suffix}"
    return str(args.view_kind or "").strip().lower()


def _active_artifact_has_diff() -> bool:
    result = subprocess.run(
        ["git", "diff", "--", ACTIVE_ARTIFACT],
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(str(result.stdout or "").strip() or str(result.stderr or "").strip())


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_ARTIFACT} has uncommitted diff.")
    lake = ResearchDataLake(str(args.data_lake_root or "").strip() or None)
    symbols = tuple(item.strip().upper() for item in str(args.symbols or "").split(",") if item.strip())
    exclude_symbol_prefixes = tuple(
        item.strip().upper() for item in str(args.exclude_symbol_prefixes or "").split(",") if item.strip()
    )
    spec = PoolViewSpec(
        source_market_dataset_id=str(args.source_market_dataset_id).strip(),
        view_kind=str(args.view_kind).strip().lower(),
        view_name=_default_view_name(args),
        start_date=str(args.start_date or "").strip(),
        end_date=str(args.end_date or "").strip(),
        pool_name=str(args.pool_name or "").strip().lower(),
        rebalance_every_days=int(args.rebalance_every_days),
        adv_window=int(args.adv_window),
        min_price=float(args.min_price),
        max_price=float(args.max_price),
        exchange_suffix=str(args.exchange_suffix or "").strip().upper(),
        symbols=symbols,
        exclude_symbol_prefixes=exclude_symbol_prefixes,
    )
    record = build_pool_view_from_policy_bundle(lake=lake, spec=spec, reuse=not bool(args.refresh))
    manifest = {
        "status": "ok",
        "dataset_id": record.dataset_id,
        "dataset_kind": record.dataset_kind,
        "fingerprint": record.fingerprint,
        "view_kind": spec.to_dict()["view_kind"],
        "view_name": spec.to_dict()["view_name"],
        "source_market_dataset_id": spec.source_market_dataset_id,
        "row_counts": record.row_counts,
        "content_paths": record.content_paths,
    }
    lake.write_catalog_manifest()
    print(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    main()
