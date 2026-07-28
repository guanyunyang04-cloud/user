from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy import seq100_time_splits as splits


def _normalization(scope: str, cutoff: str, count_key: str) -> dict[str, object]:
    return {
        "fit_scope": scope,
        "fit_date_end_exclusive": cutoff,
        count_key: 0,
        "daily_raw": {"mean": [0.0], "std": [1.0]},
    }


def test_fixed_oos_split_rejects_label_overlap() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train", "train", "oos", "oos"],
            "trade_date": ["2022-01-03", "2022-01-04", "2023-01-03", "2023-01-04"],
            "date_idx": [0, 1, 4, 5],
            "symbol_idx": [0, 1, 0, 1],
        }
    )
    manifest = {
        "forward_days": 2,
        "feature_channels": {"daily_raw": {"columns": ["close"]}},
        "purged_walkforward": {
            "oos_year": 2023,
            "oos_start": "2023-01-03",
            "oos_start_date_idx": 4,
            "label_overlap_count": 0,
        },
        "normalization": _normalization(
            "feature_dates_before_oos_start", "2023-01-03", "oos_feature_date_count"
        ),
    }

    summary = splits.validate_fixed_oos_split(manifest, sample_frame=frame)
    assert summary["maximum_train_label_end_date_idx"] == 3
    assert summary["evaluation_start_date_idx"] == 4

    overlapping = frame.copy()
    overlapping.loc[1, "date_idx"] = 2
    with pytest.raises(ValueError, match="overlap"):
        splits.validate_fixed_oos_split(manifest, sample_frame=overlapping)


def test_build_development_fold_uses_prediction_horizon_for_purge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(splits, "WORKSPACE_ROOT", tmp_path)
    source_root = tmp_path / "source"
    source_root.mkdir()
    sample_path = source_root / "sample.parquet"
    candidate_path = source_root / "candidates.parquet"
    panel_path = source_root / "daily_raw.float32.dat"
    rows = []
    sample_id = 0
    dates = [
        "2021-12-27",
        "2021-12-28",
        "2021-12-29",
        "2021-12-30",
        "2021-12-31",
        "2022-01-04",
        "2022-01-05",
        "2022-01-06",
        "2022-01-07",
    ]
    for date_idx in (0, 1, 2, 3, 4, 5):
        for symbol_idx in (0, 1):
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": "train" if date_idx < 5 else "test",
                    "year": 2021 if date_idx < 5 else 2022,
                    "trade_date": dates[date_idx],
                    "date_idx": date_idx,
                    "symbol_idx": symbol_idx,
                    "symbol": f"00000{symbol_idx + 1}.SZ",
                    "entry_trade_date": dates[min(date_idx + 1, len(dates) - 1)],
                    "entry_filled": True,
                    "label_valid": True,
                    "price_label_valid": True,
                    "va_aux_valid": True,
                }
            )
            sample_id += 1
    sample = pd.DataFrame(rows)
    candidates = sample[sample["year"].eq(2022)].copy()
    candidates["candidate_id"] = np.arange(len(candidates), dtype=np.int64)
    sample.to_parquet(sample_path, index=False)
    candidates.to_parquet(candidate_path, index=False)
    np.arange(len(dates) * 2, dtype=np.float32).reshape(len(dates), 2, 1).tofile(
        panel_path
    )
    manifest_path = source_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_type": "qdp_v2_sequence_path_pack",
                "forward_days": 2,
                "execution_tail_days": 1,
                "max_label_dependency_days": 2,
                "max_execution_dependency_days": 3,
                "date_values": dates,
                "sample_index_path": str(sample_path),
                "candidate_index_path": str(candidate_path),
                "feature_channels": {
                    "daily_raw": {
                        "path": str(panel_path),
                        "shape": [len(dates), 2, 1],
                        "dtype": "float32",
                        "columns": ["close"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = splits.build_development_fold_view(
        source_manifest=manifest_path,
        output_root=tmp_path / "folds",
        development_year=2022,
    )
    view = json.loads(Path(result["view_path"]).read_text(encoding="utf-8"))
    summary = splits.validate_development_split(view)

    assert result["purge_trading_days"] == 2
    assert summary["training_label_dependency_days"] == 2
    assert summary["execution_dependency_days"] == 3
    assert summary["maximum_train_label_end_date_idx"] < summary[
        "evaluation_start_date_idx"
    ]
    assert "source_view_provenance" not in view
    assert "development_fold_training_contract" not in view
