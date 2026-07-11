from __future__ import annotations

import json
from pathlib import Path

import pytest

from daily_research.path_policy.qdp_v2_sequence_path_pack import _apply_research_dataset_view


def _manifest(root: Path, domain: str, dataset_id: str) -> None:
    path = root / "datasets" / domain / dataset_id / "dataset.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"domain": domain, "dataset_id": dataset_id}), encoding="utf-8")


def test_dataset_view_atomically_overrides_full_pit_market_suite(tmp_path: Path) -> None:
    active = {"datasets": {"trading_calendar": "calendar__active", "market_daily_raw": "market__active"}}
    datasets = {"trading_calendar": "calendar__active"}
    overrides = {
        "market_daily_raw": "market__pit",
        "adjust_factor": "factor__pit",
        "limit_status": "limit__pit",
        "security_status": "status__pit",
        "pit_signal_universe": "universe__pit",
    }
    datasets.update(overrides)
    for domain, dataset_id in datasets.items():
        _manifest(tmp_path, domain, dataset_id)
    view_path = tmp_path / "views" / "pit.json"
    view_path.parent.mkdir()
    view_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "qdp_v2_research_dataset_view",
                "view_id": "pit_fixture",
                "datasets": datasets,
                "overrides": overrides,
            }
        ),
        encoding="utf-8",
    )

    selected, view, resolved = _apply_research_dataset_view(tmp_path, active, view_path)

    assert selected["datasets"] == datasets
    assert selected["datasets"]["market_daily_raw"] == "market__pit"
    assert active["datasets"]["market_daily_raw"] == "market__active"
    assert view is not None and view["view_id"] == "pit_fixture"
    assert resolved == str(view_path.resolve())


def test_dataset_view_rejects_partial_override(tmp_path: Path) -> None:
    view_path = tmp_path / "partial.json"
    view_path.write_text(
        json.dumps(
            {
                "kind": "qdp_v2_research_dataset_view",
                "datasets": {},
                "overrides": {"market_daily_raw": "market__pit"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required overrides"):
        _apply_research_dataset_view(tmp_path, {"datasets": {}}, view_path)
