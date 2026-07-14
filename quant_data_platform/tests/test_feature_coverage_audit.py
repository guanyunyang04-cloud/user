from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quant_data_platform.cli import main as cli_main
from quant_data_platform.core.json_io import read_json, write_json
from quant_data_platform.memmap.coverage_audit import audit_sharded_memmap_feature_coverage


def _write_feature_store(path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    store = np.memmap(path, dtype="float32", mode="w+", shape=values.shape)
    store[:] = values[:]
    store.flush()
    del store


def _fixture_manifest(tmp_path):
    features = [
        "ret_20d",
        "turn",
        "turn_z20",
        "valuation_peTTM_lag1_log",
        "intraday_last_5m_ret",
        "cs_z_intraday_last_5m_ret",
    ]
    shards = []
    for year in (2015, 2024):
        shard_dir = tmp_path / f"year={year}" / "block=0000"
        values = np.ones((2, 3, len(features)), dtype=np.float32)
        if year == 2015:
            values[:, :, features.index("turn")] = np.nan
            values[:, :, features.index("turn_z20")] = np.nan
            values[:, :, features.index("valuation_peTTM_lag1_log")] = np.nan
        values[:, :, features.index("cs_z_intraday_last_5m_ret")] = np.nan
        feature_path = shard_dir / "forecast_feature_store.dat"
        _write_feature_store(feature_path, values)
        shard_manifest_path = shard_dir / "shard_manifest.json"
        write_json(
            shard_manifest_path,
            {
                "status": "completed",
                "feature_manifest": {
                    "column_groups": {
                        "state": ["ret_20d"],
                        "turnover_context": ["turn", "turn_z20"],
                        "valuation_context": ["valuation_peTTM_lag1_log"],
                        "intraday_context": ["intraday_last_5m_ret", "cs_z_intraday_last_5m_ret"],
                    }
                },
            },
        )
        label_manifest_path = shard_dir / "labels_manifest.json"
        write_json(label_manifest_path, {"arrays": {}})
        shards.append(
            {
                "status": "completed",
                "year": year,
                "shard_key": f"year={year}/block=0000",
                "feature_store_path": str(feature_path),
                "feature_store_shape": [2, 3, len(features)],
                "shard_manifest_json": str(shard_manifest_path),
                "label_manifest_json": str(label_manifest_path),
            }
        )
    manifest_path = tmp_path / "sharded_memmap_manifest.json"
    write_json(
        manifest_path,
        {
            "artifact_type": "qdp_sharded_memmap",
            "feature_profile": "unit",
            "canonical_dataset_id": "policy_input_bundle__unit",
            "feature_columns": features,
            "shards": shards,
        },
    )
    return manifest_path


def test_feature_coverage_audit_reports_suspicious_groups(tmp_path) -> None:
    manifest = _fixture_manifest(tmp_path)
    report = audit_sharded_memmap_feature_coverage(
        manifest_json=manifest,
        output_root=tmp_path / "out",
        run_tag="unit",
        train_year_start=2015,
        train_year_end=2015,
        recent_year_start=2024,
        recent_year_end=2024,
    )

    assert report["status"] == "completed"
    assert report["summary"]["suspicious_by_group"]["turnover_context"] == 2
    assert report["summary"]["suspicious_by_group"]["valuation_context"] == 1
    assert report["summary"]["suspicious_by_group"]["intraday_context"] == 1
    assert "turnover_context_has_historical_coverage_gap" in report["summary"]["key_findings"]
    assert "cs_z_intraday_last_5m_ret_has_derived_feature_coverage_anomaly" in report["summary"]["key_findings"]

    suspicious = pd.read_csv(report["outputs"]["suspicious_csv"])
    assert {"turn", "turn_z20", "valuation_peTTM_lag1_log", "cs_z_intraday_last_5m_ret"}.issubset(
        set(suspicious["feature"].astype(str))
    )


def test_cli_audit_sharded_feature_coverage_is_archived(tmp_path, capsys) -> None:
    manifest = _fixture_manifest(tmp_path)
    output_root = tmp_path / "out"

    assert cli_main(
        [
            "audit-sharded-feature-coverage",
            "--manifest-json",
            str(manifest),
            "--output-root",
            str(output_root),
            "--run-tag",
            "cli_unit",
            "--train-year-start",
            "2015",
            "--train-year-end",
            "2015",
            "--recent-year-start",
            "2024",
            "--recent-year-end",
            "2024",
            "--json",
        ]
    ) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "archived"
    assert payload["command"] == "audit-sharded-feature-coverage"
    assert not (output_root / "cli_unit" / "coverage_audit_report.json").exists()
