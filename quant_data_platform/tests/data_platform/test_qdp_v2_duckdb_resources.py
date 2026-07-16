from __future__ import annotations

from quant_data_platform.qdp_v2.duckdb_resources import (
    DEFAULT_LOW_MEMORY_SECONDS,
    DEFAULT_MEMORY_FLOOR_BYTES,
    DuckDbMemoryWatchdog,
    LOW_MEMORY_REASON,
    dynamic_memory_limit_bytes,
    open_guarded_duckdb,
)


def test_dynamic_memory_limit_has_no_two_gib_cap() -> None:
    limit = dynamic_memory_limit_bytes(
        available_bytes=8 * 1024**3,
        total_bytes=16 * 1024**3,
    )

    assert limit == 8 * 1024**3 - DEFAULT_MEMORY_FLOOR_BYTES
    assert limit > 2 * 1024**3


def test_guarded_connection_uses_available_ram_ceiling(tmp_path) -> None:
    with open_guarded_duckdb(
        temp_directory=tmp_path / "spill",
        threads=1,
        memory_sampler=lambda: 8 * 1024**3,
        total_memory_sampler=lambda: 16 * 1024**3,
    ) as connection:
        assert connection.settings.memory_limit_bytes == 8053063680
        assert connection.execute("SELECT 42").fetchone() == (42,)


def test_default_memory_floor_contract_is_half_gib_for_two_seconds() -> None:
    assert DEFAULT_MEMORY_FLOOR_BYTES == 512 * 1024**2
    assert DEFAULT_LOW_MEMORY_SECONDS == 2.0
    assert LOW_MEMORY_REASON == "available_memory_below_0.5_gib_for_2_seconds"


def test_watchdog_requires_two_continuously_low_seconds() -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.interrupts = 0

        def interrupt(self) -> None:
            self.interrupts += 1

    connection = FakeConnection()
    watchdog = DuckDbMemoryWatchdog(
        connection,
        floor_bytes=500,
    )

    assert watchdog.sample_once(available_bytes=400, now=0.0) is False
    assert watchdog.sample_once(available_bytes=600, now=1.5) is False
    assert watchdog.sample_once(available_bytes=400, now=10.0) is False
    assert watchdog.sample_once(available_bytes=400, now=11.99) is False
    assert connection.interrupts == 0

    assert watchdog.sample_once(available_bytes=400, now=12.0) is True
    assert connection.interrupts == 1


def test_watchdog_interrupts_only_once_after_trigger() -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.interrupts = 0

        def interrupt(self) -> None:
            self.interrupts += 1

    connection = FakeConnection()
    watchdog = DuckDbMemoryWatchdog(
        connection,
        floor_bytes=500,
    )

    watchdog.sample_once(available_bytes=400, now=1.0)
    watchdog.sample_once(available_bytes=400, now=3.0)
    watchdog.sample_once(available_bytes=300, now=4.0)

    assert watchdog.triggered is True
    assert connection.interrupts == 1
    assert watchdog.minimum_available_bytes == 300
