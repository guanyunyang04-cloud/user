from __future__ import annotations

import json

import pandas as pd

from daily_research.path_policy import seq100_strict_chan_audit as audit


def _state_record(
    *,
    candidate_id: str,
    action: str,
    index: int,
    time: str,
) -> dict[str, object]:
    return {
        "case_id": "case",
        "category": "test",
        "symbol": "600000.SH",
        "profile": "primary",
        "episode_id": 0,
        "event_type": "segment_state",
        "id": f"state:{candidate_id}:{action}",
        "candidate_id": candidate_id,
        "action": action,
        "direction": 1,
        "event_index": index,
        "event_time": time,
        "confirmed_index": index,
        "confirmed_time": time,
    }


def test_audit_contract_exposes_all_frozen_profiles_and_cases() -> None:
    spec = audit.load_audit_spec()
    assert spec["study_id"] == "seq100_strict_chan_audit_v1"
    assert len(spec["cases"]) == 9
    assert next(iter(spec["profiles"])) == "primary"
    assert spec["boundaries"]["return_test_performed"] is False


def test_event_output_frame_handles_mixed_event_field_types(tmp_path) -> None:
    records = {
        "primary": [
            {
                "case_id": "case",
                "category": "test",
                "symbol": "600000.SH",
                "profile": "primary",
                "episode_id": 0,
                "event_type": "fractal",
                "id": "f1",
                "event_index": 1,
                "confirmed_index": 2,
                "event_time": "2020-01-01T09:35:00",
                "confirmed_time": "2020-01-01T09:40:00",
                "kind": 1,
            },
            {
                "case_id": "case",
                "category": "test",
                "symbol": "600000.SH",
                "profile": "primary",
                "episode_id": 0,
                "event_type": "trend_type",
                "id": "t1",
                "event_index": 3,
                "confirmed_index": 4,
                "event_time": "2020-01-01T09:45:00",
                "confirmed_time": "2020-01-01T09:50:00",
                "kind": "trend",
            },
        ]
    }
    frame = audit._event_output_frame(records)
    path = tmp_path / "events.parquet"
    frame.to_parquet(path, index=False)
    loaded = pd.read_parquet(path)
    assert len(loaded) == 2
    assert loaded["payload_json"].str.contains('"kind"').all()


def test_pending_state_rows_report_resolution_and_episode_end() -> None:
    records = {
        "primary": [
            _state_record(
                candidate_id="c1",
                action="opened",
                index=5,
                time="2020-01-01T10:00:00",
            ),
            _state_record(
                candidate_id="c1",
                action="confirmed",
                index=8,
                time="2020-01-01T10:15:00",
            ),
            _state_record(
                candidate_id="c2",
                action="opened",
                index=10,
                time="2020-01-01T10:25:00",
            ),
        ]
    }
    rows = audit._pending_state_rows(
        {"case_id": "case", "category": "test", "symbol": "600000.SH"},
        records,
        {0: 20},
    )
    result = pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)
    assert result.loc[0, "outcome"] == "confirmed"
    assert result.loc[0, "pending_bars"] == 3
    assert result.loc[1, "outcome"] == "unresolved_at_episode_end"
    assert result.loc[1, "pending_bars"] == 9


def test_prefix_cutoffs_are_unique_and_bounded() -> None:
    assert audit._prefix_cutoffs(10, 4) == [1, 4, 7, 10]
    assert audit._prefix_cutoffs(2, 4) == [1, 2]


def test_visual_review_is_bound_to_case_fingerprint(tmp_path) -> None:
    path = tmp_path / "visual_review.json"
    summaries = [{"case_id": "case", "fingerprint": "abc"}]
    assert audit._visual_review_state(path, summaries)[0] == "unreviewed"
    path.write_text(
        json.dumps(
            {
                "schema": "seq100_strict_chan_visual_review/1",
                "reviews": [
                    {
                        "case_id": "case",
                        "fingerprint": "abc",
                        "review_status": "passed",
                        "parser_error": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert audit._visual_review_state(path, summaries)[0] == "passed"
    summaries[0]["fingerprint"] = "changed"
    assert audit._visual_review_state(path, summaries)[0] == "stale"
