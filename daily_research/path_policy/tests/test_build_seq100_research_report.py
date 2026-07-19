from __future__ import annotations

import copy
import json

import pytest

from daily_research.path_policy import build_seq100_research_report as report


def _payload_or_skip() -> dict[str, object]:
    if not (report.COMPARISON_ROOT / "comparison_summary.json").is_file():
        pytest.skip("local Seq100 comparison outputs are unavailable")
    summary = json.loads(
        (report.COMPARISON_ROOT / "comparison_summary.json").read_text(encoding="utf-8")
    )
    if int(summary.get("schema_version", 0)) != 2:
        pytest.skip("local Seq100 comparison has not yet migrated to integrity v2")
    return report.build_artifact_payload()


def test_compact_report_has_fixed_reader_shape_and_source_derived_values() -> None:
    artifact = _payload_or_skip()
    counts = report.validate_compact_artifact(artifact)
    assert counts == {
        "cards": 4,
        "charts": 4,
        "tables": 2,
        "sources": 5,
        "blocks": 16,
    }
    datasets = artifact["snapshot"]["datasets"]  # type: ignore[index]
    assert datasets["finite_headline"][0]["legal_cagr"] == pytest.approx(
        0.6061182316778575
    )
    assert datasets["finite_headline"][0]["structured_cagr"] == pytest.approx(
        0.9515516697428784
    )
    assert datasets["freshness_headline"][0]["signal_dates"] == 48
    assert len(datasets["vintage_metrics"]) == 8
    serialized = json.dumps(artifact, ensure_ascii=False)
    assert "H:\\" not in serialized
    assert "sha256" not in serialized.lower()


def test_compact_report_validator_rejects_growth_and_hash_detail() -> None:
    artifact = _payload_or_skip()
    too_long = copy.deepcopy(artifact)
    too_long["manifest"]["blocks"].append(  # type: ignore[index]
        {"id": "extra", "type": "markdown", "body": "## Extra"}
    )
    with pytest.raises(ValueError, match="count limit"):
        report.validate_compact_artifact(too_long)

    leaked = copy.deepcopy(artifact)
    leaked["manifest"]["description"] = "checkpoint_sha256 detail"  # type: ignore[index]
    with pytest.raises(ValueError, match="leaked"):
        report.validate_compact_artifact(leaked)
