from __future__ import annotations

import pytest

from quantlab.research.minute_strategy_admission import (
    DateParallelAdmissionError,
    admit_date_workers,
    require_date_workers,
)


def test_admission_reserves_floor_and_transient_margin() -> None:
    result = admit_date_workers(
        2,
        available_bytes=int(6.2 * 1024**3),
        floor_gib=0.5,
        safety_margin_gib=0.5,
        estimated_worker_gib=2.5,
    )

    assert result.admitted_workers == 2
    assert result.reason == "admitted"
    assert result.remaining_after_admission_bytes == pytest.approx(0.2 * 1024**3)


def test_admission_reduces_workers_when_memory_budget_is_tight() -> None:
    result = admit_date_workers(
        2,
        available_bytes=int(4.7 * 1024**3),
        estimated_worker_gib=2.5,
    )

    assert result.admitted_workers == 1
    assert result.reason == "capped_by_memory_budget"


def test_admission_never_starts_when_one_worker_would_cross_floor() -> None:
    result = admit_date_workers(
        1,
        available_bytes=int(3.4 * 1024**3),
        estimated_worker_gib=2.5,
    )

    assert result.admitted_workers == 0
    assert not result.admitted
    with pytest.raises(DateParallelAdmissionError, match="memory_admission_failed"):
        require_date_workers(
            1,
            available_bytes=int(3.4 * 1024**3),
            estimated_worker_gib=2.5,
        )


def test_admission_applies_explicit_worker_cap() -> None:
    result = admit_date_workers(
        4,
        available_bytes=int(20 * 1024**3),
        max_workers=2,
        estimated_worker_gib=2.5,
    )

    assert result.admitted_workers == 2
    assert result.reason == "capped_by_max_workers"


def test_admission_counts_active_workers_against_memory_and_cap() -> None:
    result = admit_date_workers(
        2,
        active_workers=1,
        available_bytes=int(8 * 1024**3),
        max_workers=2,
        estimated_worker_gib=2.5,
    )

    assert result.admitted_workers == 1
    assert result.total_workers_after_admission == 2
    assert result.remaining_after_admission_bytes == pytest.approx(2.0 * 1024**3)


def test_admission_rejects_invalid_active_worker_count() -> None:
    with pytest.raises(DateParallelAdmissionError, match="workers_invalid"):
        admit_date_workers(1, active_workers=2, max_workers=1)
