from __future__ import annotations

import pandas as pd

from daily_research.path_policy import seq100_strict_chan_panel as panel


def _event(
    event_type: str,
    event_id: str,
    confirmed_time: str,
    *,
    candidate_id: str | None = None,
    action: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "event_type": event_type,
        "id": event_id,
        "event_index": 1,
        "confirmed_index": 1,
        "confirmed_time": confirmed_time,
    }
    if candidate_id is not None:
        value["candidate_id"] = candidate_id
    if action is not None:
        value["action"] = action
    return value


def test_bucket_membership_is_deterministic_complete_and_contiguous() -> None:
    symbols = [f"{value:06d}.SZ" for value in range(10)]
    buckets = [
        panel._bucket_members(symbols, bucket_id=index, bucket_count=3)
        for index in range(3)
    ]
    assert buckets == [symbols[:3], symbols[3:6], symbols[6:]]
    assert [symbol for bucket in buckets for symbol in bucket] == symbols


def test_snapshot_keeps_formal_dates_and_only_active_pending_candidates() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["2011-12-30", "2012-01-03", "2012-01-04"],
            "timestamp": pd.to_datetime(
                ["2011-12-30 15:00", "2012-01-03 15:00", "2012-01-04 15:00"]
            ),
        }
    )
    records = [
        _event(
            "segment_state",
            "candidate:opened",
            "2012-01-03 15:00:00",
            candidate_id="candidate",
            action="opened",
        ),
        _event(
            "segment_state",
            "candidate:confirmed",
            "2012-01-03 15:00:00",
            candidate_id="candidate",
            action="confirmed",
        ),
        _event("center", "center:1", "2012-01-04 15:00:00"),
    ]
    rows = panel._snapshot_rows(
        "000001.SZ",
        0,
        frame,
        records,
        formal_start="2012-01-01",
        formal_end="2025-12-31",
    )
    assert [row["trade_date"] for row in rows] == ["2012-01-03", "2012-01-04"]
    assert rows[0]["pending_open_count_visible"] == 0
    assert rows[1]["pending_open_count_visible"] == 0


def test_event_rows_mark_burn_in_without_discarding_provenance() -> None:
    rows = panel._event_rows(
        "000001.SZ",
        0,
        [_event("fractal", "f1", "2011-12-30 15:00:00")],
        formal_start="2012-01-01",
        formal_end="2025-12-31",
    )
    assert len(rows) == 1
    assert rows[0]["is_formal"] is False


def test_config_fingerprint_changes_with_bucket_membership() -> None:
    source = {"pinned_datasets": {"market_intraday_5m": "dataset"}}
    first = panel._config_fingerprint(
        bucket_id=0,
        bucket_count=4,
        symbols=["000001.SZ"],
        definition_sha256="definition",
        implementation_sha256="implementation",
        source=source,
    )
    second = panel._config_fingerprint(
        bucket_id=0,
        bucket_count=4,
        symbols=["000002.SZ"],
        definition_sha256="definition",
        implementation_sha256="implementation",
        source=source,
    )
    assert first != second


def test_event_frame_serializes_heterogeneous_and_nested_fields() -> None:
    frame = panel._parquet_safe_event_frame(
        [
            {"event_type": "fractal", "kind": 1, "component_ids": ("a", "b")},
            {
                "event_type": "movement",
                "kind": "consolidation",
                "component_ids": None,
            },
        ]
    )
    assert frame["kind"].tolist() == ["1", "consolidation"]
    assert frame.loc[0, "component_ids"] == '["a", "b"]'
