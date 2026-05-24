from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.execution.production_signal import refresh_production_live_panels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh production live score/target panels from the active lake dataset without retraining."
    )
    parser.add_argument("--as-of-date", default="", help="Latest completed trading date to cover.")
    parser.add_argument("--no-anchor-sync", action="store_true", help="Skip production anchor sync before panel inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = refresh_production_live_panels(
        as_of_date=args.as_of_date,
        sync_anchor=not bool(args.no_anchor_sync),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
