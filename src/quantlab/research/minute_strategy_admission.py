"""Memory admission for optional date-parallel minute studies.

Date workers duplicate month-level context and DuckDB/pandas buffers.  Merely
checking the memory floor in each child is too late: all children can start at
once and collectively exhaust the machine before any one watchdog fires.  This
module performs a conservative parent-side admission calculation.  It does
not start workers; a future scheduler can call it before each batch and again
before launching another worker.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any

import psutil

GIB = 1024**3
DEFAULT_MEMORY_FLOOR_GIB = 0.5
DEFAULT_SAFETY_MARGIN_GIB = 0.5
# The full-universe single-month run reached about 3.4 GiB of process-tree RSS.
# Round upward for admission rather than treating either that measurement or
# the smaller two-process smoke test as an exact allocation limit.
DEFAULT_ESTIMATED_WORKER_GIB = 3.5
DEFAULT_MAX_WORKERS = 2


class DateParallelAdmissionError(ValueError):
    """Raised when an admission policy argument is invalid."""


@dataclass(frozen=True)
class DateParallelAdmission:
    """Result of one parent-side date-worker admission check."""

    requested_workers: int
    active_workers: int
    admitted_workers: int
    available_bytes: int
    floor_bytes: int
    safety_margin_bytes: int
    estimated_worker_bytes: int
    max_workers: int
    reason: str

    @property
    def admitted(self) -> bool:
        return self.admitted_workers > 0

    @property
    def remaining_after_admission_bytes(self) -> int:
        return max(
            0,
            self.available_bytes
            - self.floor_bytes
            - self.safety_margin_bytes
            - (self.active_workers + self.admitted_workers)
            * self.estimated_worker_bytes,
        )

    @property
    def total_workers_after_admission(self) -> int:
        return self.active_workers + self.admitted_workers

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_workers": self.requested_workers,
            "active_workers": self.active_workers,
            "admitted_workers": self.admitted_workers,
            "available_bytes": self.available_bytes,
            "available_gib": self.available_bytes / GIB,
            "floor_bytes": self.floor_bytes,
            "floor_gib": self.floor_bytes / GIB,
            "safety_margin_bytes": self.safety_margin_bytes,
            "safety_margin_gib": self.safety_margin_bytes / GIB,
            "estimated_worker_bytes": self.estimated_worker_bytes,
            "estimated_worker_gib": self.estimated_worker_bytes / GIB,
            "max_workers": self.max_workers,
            "total_workers_after_admission": self.total_workers_after_admission,
            "remaining_after_admission_bytes": self.remaining_after_admission_bytes,
            "remaining_after_admission_gib": self.remaining_after_admission_bytes / GIB,
            "admitted": self.admitted,
            "reason": self.reason,
        }


def _bytes_from_gib(value: float, name: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DateParallelAdmissionError(f"date_parallel_{name}_invalid") from exc
    if not math.isfinite(number) or number < 0.0:
        raise DateParallelAdmissionError(f"date_parallel_{name}_invalid")
    return int(number * GIB)


def admit_date_workers(
    requested_workers: int,
    *,
    active_workers: int = 0,
    available_bytes: int | None = None,
    floor_gib: float = DEFAULT_MEMORY_FLOOR_GIB,
    safety_margin_gib: float = DEFAULT_SAFETY_MARGIN_GIB,
    estimated_worker_gib: float = DEFAULT_ESTIMATED_WORKER_GIB,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> DateParallelAdmission:
    """Return how many date workers can be started safely right now.

    ``requested_workers`` is the number of *new* workers the caller wants to
    start; ``active_workers`` are already-running workers counted against the
    explicit cap and memory budget. ``available_bytes`` is injectable for
    deterministic tests. The admission budget reserves both the hard machine
    floor and an additional transient allocation margin before charging each
    estimated worker. A requested count is capped even when abundant memory is
    available; this keeps the experimental date-parallel path bounded until a
    larger-universe benchmark justifies a different limit.
    """

    try:
        requested = int(requested_workers)
        active = int(active_workers)
        maximum = int(max_workers)
    except (TypeError, ValueError) as exc:
        raise DateParallelAdmissionError("date_parallel_workers_invalid") from exc
    if requested < 1 or active < 0 or maximum < 1 or requested > 64 or active > maximum:
        raise DateParallelAdmissionError("date_parallel_workers_invalid")
    available_slots = max(0, maximum - active)
    requested_for_budget = min(requested, available_slots)
    if requested > available_slots:
        # A caller asking for more than the explicit cap is not an error: the
        # cap is the policy's purpose and is reported in ``reason``.
        cap_limited = True
    else:
        cap_limited = False
    floor_bytes = _bytes_from_gib(floor_gib, "floor_gib")
    safety_bytes = _bytes_from_gib(safety_margin_gib, "safety_margin_gib")
    worker_bytes = _bytes_from_gib(estimated_worker_gib, "estimated_worker_gib")
    if worker_bytes < 1:
        raise DateParallelAdmissionError("date_parallel_estimated_worker_gib_invalid")
    available = (
        int(psutil.virtual_memory().available)
        if available_bytes is None
        else int(available_bytes)
    )
    if available < 0:
        raise DateParallelAdmissionError("date_parallel_available_bytes_invalid")
    budget = max(0, available - floor_bytes - safety_bytes)
    memory_capacity = budget // worker_bytes
    admitted_total_capacity = max(0, int(memory_capacity) - active)
    admitted = min(requested_for_budget, admitted_total_capacity)
    if admitted <= 0:
        reason = (
            "no_additional_worker_slot"
            if available_slots <= 0
            else "insufficient_memory_for_one_worker"
        )
    elif cap_limited and admitted == requested_for_budget:
        reason = "capped_by_max_workers"
    elif admitted < requested_for_budget:
        reason = "capped_by_memory_budget"
    else:
        reason = "admitted"
    return DateParallelAdmission(
        requested_workers=requested,
        active_workers=active,
        admitted_workers=admitted,
        available_bytes=available,
        floor_bytes=floor_bytes,
        safety_margin_bytes=safety_bytes,
        estimated_worker_bytes=worker_bytes,
        max_workers=maximum,
        reason=reason,
    )


def require_date_workers(*args: Any, **kwargs: Any) -> DateParallelAdmission:
    """Perform admission and raise if no worker can be safely started."""

    result = admit_date_workers(*args, **kwargs)
    if not result.admitted:
        raise DateParallelAdmissionError(
            f"date_parallel_memory_admission_failed:{result.available_bytes}:{result.floor_bytes}"
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantlab minute-strategy-admission")
    parser.add_argument("--requested-workers", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--floor-gib", type=float, default=DEFAULT_MEMORY_FLOOR_GIB)
    parser.add_argument(
        "--safety-margin-gib", type=float, default=DEFAULT_SAFETY_MARGIN_GIB
    )
    parser.add_argument(
        "--estimated-worker-gib",
        type=float,
        default=DEFAULT_ESTIMATED_WORKER_GIB,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = admit_date_workers(
        args.requested_workers,
        max_workers=args.max_workers,
        floor_gib=args.floor_gib,
        safety_margin_gib=args.safety_margin_gib,
        estimated_worker_gib=args.estimated_worker_gib,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0 if result.admitted else 2


__all__ = [
    "DEFAULT_ESTIMATED_WORKER_GIB",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_MEMORY_FLOOR_GIB",
    "DEFAULT_SAFETY_MARGIN_GIB",
    "DateParallelAdmission",
    "DateParallelAdmissionError",
    "admit_date_workers",
    "main",
    "require_date_workers",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
