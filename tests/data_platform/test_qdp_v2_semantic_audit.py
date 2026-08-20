from __future__ import annotations

import json

from quantlab.data.qdp_v2 import check, semantic_audit


def test_specialty_record_requires_terminal_status_and_true_checks(tmp_path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(
        json.dumps(
            {
                "status": "applied",
                "checks": {"primary_key_unique": True, "semantic_state": True},
                "request_2026_count": 0,
            }
        ),
        encoding="utf-8",
    )

    record = semantic_audit._specialty_record(path)

    assert record["ok"] is True
    assert record["failed_checks"] == []


def test_specialty_record_exposes_failed_or_missing_evidence(tmp_path) -> None:
    failed = tmp_path / "failed.json"
    failed.write_text(
        json.dumps(
            {
                "status": "applied",
                "checks": {"state_contract": False},
                "request_2026_count": 1,
            }
        ),
        encoding="utf-8",
    )

    record = semantic_audit._specialty_record(failed)
    missing = semantic_audit._specialty_record(tmp_path / "missing.json")

    assert record["ok"] is False
    assert record["failed_checks"] == [
        "state_contract",
        "request_2026_count_nonzero",
    ]
    assert missing["status"] == "missing"


def test_specialty_record_reads_nested_2026_request_count(tmp_path) -> None:
    path = tmp_path / "nested.json"
    path.write_text(
        json.dumps(
            {
                "status": "applied",
                "checks": {"primary_key_unique": True},
                "statistics": {"request_2026_count": 1},
            }
        ),
        encoding="utf-8",
    )

    record = semantic_audit._specialty_record(path)

    assert record["ok"] is False
    assert record["failed_checks"] == ["request_2026_count_nonzero"]


def test_check_semantic_mode_uses_compact_semantic_audit(monkeypatch) -> None:
    monkeypatch.setattr(
        check,
        "audit_semantics",
        lambda **kwargs: {"status": "ok", "mode": "semantic", **kwargs},
    )

    result = check.run_check(workspace_root="workspace", semantic=True, write=True)

    assert result["status"] == "ok"
    assert result["workspace_root"] == "workspace"
    assert result["write"] is True
