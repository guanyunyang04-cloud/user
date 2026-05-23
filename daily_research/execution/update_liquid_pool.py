from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json

from daily_research.execution.liquidity_universe import parse_pool_sizes, update_liquidity_pool_files


def parse_args():
    parser = argparse.ArgumentParser(description="Update the daily high-liquidity stock pools used by the execution workflow")
    parser.add_argument("--pool-sizes", default="300,500,800", help="Comma-separated pool sizes to export.")
    parser.add_argument("--start-date", default="20240101")
    parser.add_argument("--signal-date", default="", help="Optional completed trading date override, e.g. 20260320.")
    parser.add_argument("--universe-scope", default="all_a")
    parser.add_argument("--data-source", choices=["lake", "tq"], default="lake")
    parser.add_argument("--lake-dataset-id", default="")
    parser.add_argument("--data-lake-root", default="")
    parser.add_argument("--lookback-days", type=int, default=80)
    parser.add_argument("--min-price", type=float, default=2.0)
    parser.add_argument("--max-price", type=float, default=300.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    pool_sizes = parse_pool_sizes(args.pool_sizes)
    artifacts = update_liquidity_pool_files(
        pool_sizes=pool_sizes,
        start_date=args.start_date,
        signal_date=args.signal_date or None,
        universe_scope=args.universe_scope,
        data_source=args.data_source,
        lake_dataset_id=args.lake_dataset_id,
        data_lake_root=args.data_lake_root,
        lookback_days=args.lookback_days,
        min_price=args.min_price,
        max_price=args.max_price,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
    )

    summary = {
        "signal_date": artifacts.signal_date,
        "universe_dir": str(artifacts.universe_dir),
        "latest_files": {str(size): str(path) for size, path in artifacts.latest_files.items()},
        "snapshot_files": {str(size): str(path) for size, path in artifacts.snapshot_files.items()},
        "ranking_csv": str(artifacts.ranking_csv),
        "summary_csv": str(artifacts.summary_csv),
        "cache": artifacts.cache_meta,
    }
    print("High-liquidity execution pools updated.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
