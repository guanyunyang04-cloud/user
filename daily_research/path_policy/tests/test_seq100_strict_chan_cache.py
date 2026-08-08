from __future__ import annotations

import json
from pathlib import Path

from daily_research.path_policy import seq100_strict_chan_cache as cache


def test_bucket_assignment_is_complete_contiguous_and_balanced() -> None:
    symbols = [f"{value:06d}.SZ" for value in range(10)]
    frame = cache._bucket_assignment_frame(symbols, bucket_count=3)
    assert frame["symbol"].tolist() == symbols
    assert frame["bucket_id"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    assert (
        frame.groupby("bucket_id").size().max()
        - frame.groupby("bucket_id").size().min()
        <= 1
    )


def test_cache_fingerprint_changes_with_symbols() -> None:
    study = {
        "_study_path": str(cache.DEFAULT_STUDY_PATH),
        "period": {"burn_in_start": "2010-01-01", "formal_end": "2025-12-31"},
    }
    first = cache._cache_fingerprint(
        study=study,
        pinned_datasets={"market_intraday_5m": "dataset"},
        symbols=["000001.SZ"],
        bucket_count=1,
        maximum_symbols=1,
    )
    second = cache._cache_fingerprint(
        study=study,
        pinned_datasets={"market_intraday_5m": "dataset"},
        symbols=["000002.SZ"],
        bucket_count=1,
        maximum_symbols=1,
    )
    assert first != second


def test_resolve_active_cache_returns_none_without_pointer(tmp_path: Path) -> None:
    assert cache.resolve_active_cache(cache_root=tmp_path, bucket_count=256) is None


def test_completed_domain_manifest_rejects_missing_file(tmp_path: Path) -> None:
    manifest = tmp_path / "domain.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "fingerprint": "fingerprint",
                "files": [{"path": str(tmp_path / "missing.parquet")}],
            }
        ),
        encoding="utf-8",
    )
    assert cache._completed_domain_manifest(manifest, fingerprint="fingerprint") is None
