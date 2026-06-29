from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    duckdb_memory_limit: str
    duckdb_threads: int
    baostock_workers: int
    baostock_task_timeout_seconds: int
    baostock_max_tasks_per_session: int
    baostock_retry: int
    mootdx_workers: int
    mootdx_retries: int
    mootdx_page_size: int
    cninfo_workers: int
    cninfo_timeout_seconds: int
    cninfo_retry: int
    cninfo_backoff_seconds: tuple[int, ...]
    max_task_rss_gb: int = 12


RUNTIME_PROFILES: dict[str, RuntimeProfile] = {
    "safe": RuntimeProfile(
        name="safe",
        duckdb_memory_limit="4GB",
        duckdb_threads=2,
        baostock_workers=1,
        baostock_task_timeout_seconds=180,
        baostock_max_tasks_per_session=200,
        baostock_retry=2,
        mootdx_workers=1,
        mootdx_retries=3,
        mootdx_page_size=800,
        cninfo_workers=2,
        cninfo_timeout_seconds=20,
        cninfo_retry=3,
        cninfo_backoff_seconds=(1, 3, 8),
    ),
    "balanced": RuntimeProfile(
        name="balanced",
        duckdb_memory_limit="8GB",
        duckdb_threads=3,
        baostock_workers=2,
        baostock_task_timeout_seconds=180,
        baostock_max_tasks_per_session=200,
        baostock_retry=2,
        mootdx_workers=3,
        mootdx_retries=3,
        mootdx_page_size=800,
        cninfo_workers=4,
        cninfo_timeout_seconds=20,
        cninfo_retry=3,
        cninfo_backoff_seconds=(1, 3, 8),
    ),
    "fast": RuntimeProfile(
        name="fast",
        duckdb_memory_limit="10GB",
        duckdb_threads=4,
        baostock_workers=3,
        baostock_task_timeout_seconds=180,
        baostock_max_tasks_per_session=200,
        baostock_retry=2,
        mootdx_workers=4,
        mootdx_retries=3,
        mootdx_page_size=800,
        cninfo_workers=6,
        cninfo_timeout_seconds=20,
        cninfo_retry=3,
        cninfo_backoff_seconds=(1, 3, 8),
    ),
}


def resolve_runtime_profile(name: str | None) -> RuntimeProfile:
    resolved = str(name or "balanced").strip().lower() or "balanced"
    if resolved not in RUNTIME_PROFILES:
        allowed = ", ".join(sorted(RUNTIME_PROFILES))
        raise ValueError(f"unknown_runtime_profile: {resolved}; allowed={allowed}")
    return RUNTIME_PROFILES[resolved]
