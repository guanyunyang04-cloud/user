from __future__ import annotations

import json
import sys
from typing import Any

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.progress import suppress_progress
from quant_data_platform.qdp_v2.cli import HELP_TEXT, dispatch


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in raw
    if as_json:
        reconfigure = getattr(sys.stdout, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass
    try:
        with suppress_progress() if as_json else _null_context():
            return _dispatch(raw)
    except SystemExit:
        raise
    except Exception as exc:
        payload: dict[str, Any] = {
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        if as_json:
            print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
        else:
            print(
                f"status: error\nerror_type: {payload['error_type']}\n"
                f"message: {payload['message']}",
                file=sys.stderr,
            )
        return 2


def _dispatch(raw: list[str]) -> int:
    workspace, positional = _global_options(raw)
    if not positional or positional in (["-h"], ["--help"]):
        print(HELP_TEXT)
        return 0
    forwarded = list(positional)
    if workspace:
        forwarded.extend(["--workspace-root", workspace])
    result = dispatch(forwarded)
    if result is None:
        raise ValueError(f"unsupported_qdp_command:{' '.join(positional)}")
    return int(result)


def _global_options(raw: list[str]) -> tuple[str, list[str]]:
    workspace = ""
    positional: list[str] = []
    index = 0
    while index < len(raw):
        item = raw[index]
        if item == "--workspace-root":
            if index + 1 >= len(raw):
                raise ValueError(f"missing_value:{item}")
            workspace = str(raw[index + 1])
            index += 2
            continue
        if item.startswith("--workspace-root="):
            workspace = item.split("=", 1)[1]
        else:
            positional.append(item)
        index += 1
    return workspace, positional


class _null_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
