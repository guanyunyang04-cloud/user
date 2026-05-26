from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tiny execution console smoke job.")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--output-dir", default="daily_research/output/execution_app/smoke")
    parser.add_argument("--run-id", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    steps = max(int(args.steps or 0), 1)
    delay = max(float(args.sleep or 0), 0.0)
    run_id = str(args.run_id or f"execution_smoke_{int(time.time())}").strip()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.json"
    for idx in range(1, steps + 1):
        payload = {
            "status": "running" if idx < steps else "ok",
            "run_id": run_id,
            "step": idx,
            "steps": steps,
            "updated_at_epoch": time.time(),
        }
        progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"execution_smoke step={idx}/{steps}")
        if idx < steps and delay > 0:
            time.sleep(delay)
    manifest_path = run_dir / "smoke_manifest.json"
    manifest_path.write_text(
        json.dumps({"status": "ok", "run_id": run_id, "progress_path": str(progress_path.resolve())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"execution_smoke_manifest={manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
