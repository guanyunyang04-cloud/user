from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantlab.data.domains.contracts import DataDomain, normalize_domain
from quantlab.data.qdp_v2 import tushare_extended_backfill as extended
from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    qdp_v2_root,
    write_active_manifest,
    write_dataset_manifest,
)
from quantlab.data.qdp_v2.provider_credentials import (
    save_tushare_provider_profile,
)


class FakeClient:
    def __init__(self, frames: list[pd.DataFrame]) -> None:
        self.frames = list(frames)
        self.calls = 0

    def fetch(self, api_name, *, params, fields):
        del api_name, params, fields
        self.calls += 1
        return self.frames.pop(0)


def _unit_spec() -> extended.EndpointSpec:
    return extended.EndpointSpec(
        name="unit",
        api_name="unit",
        domain="unit_domain",
        mode="trade_date",
        fields=("trade_date", "ts_code", "value"),
        primary_key=("trade_date", "security_id"),
        frequency="1d",
        contract_version="unit_v1",
    )


def test_new_domains_are_registered() -> None:
    assert normalize_domain("stk_factor_pro") == DataDomain.STK_FACTOR_PRO_RAW
    assert normalize_domain("margin_detail") == DataDomain.MARGIN_DETAIL
    assert normalize_domain("moneyflow") == DataDomain.MONEYFLOW_RAW


def test_api_url_uses_private_profile_and_allows_environment_override(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("QDP_TUSHARE_API_URL", raising=False)
    monkeypatch.delenv("TUSHARE_API_URL", raising=False)
    monkeypatch.delenv("QDP_TUSHARE_PREFER_ENV", raising=False)
    with pytest.raises(
        extended.TushareExtendedBackfillError, match="tushare_api_url_missing"
    ):
        extended._resolve_tushare_api_url(tmp_path)

    save_tushare_provider_profile(
        workspace_root=tmp_path,
        token="unit-secret",
        api_url="https://profile.example/api",
        plan="unit",
        expires_at="2026-08-06T17:48:00+08:00",
        rate_limit_per_minute=150,
    )
    assert extended._resolve_tushare_api_url(tmp_path) == (
        "https://profile.example/api"
    )

    monkeypatch.setenv("QDP_TUSHARE_API_URL", "https://example.invalid/api")
    monkeypatch.setenv("QDP_TUSHARE_PREFER_ENV", "1")
    assert extended._resolve_tushare_api_url(tmp_path) == "https://example.invalid/api"


def test_year_range_task_never_uses_2026() -> None:
    params = extended._task_params(extended.SPECS["margin"], "2011", 0)

    assert params == {
        "start_date": "20110101",
        "end_date": "20111231",
        "limit": 5000,
        "offset": 0,
    }


def test_stock_level_endpoints_use_daily_all_stock_tasks() -> None:
    assert extended.SPECS["margin-detail"].mode == "trade_date"
    assert extended.SPECS["moneyflow"].mode == "trade_date"
    assert extended._task_params(extended.SPECS["moneyflow"], "2023-12-29", 0) == {
        "trade_date": "20231229",
        "limit": 5000,
        "offset": 0,
    }


def test_daily_page_inventory_ignores_truncated_legacy_year_task(tmp_path) -> None:
    spec = extended.SPECS["moneyflow"]
    legacy = (
        extended._runtime(tmp_path)
        / "raw"
        / spec.name
        / "year=2011"
        / "task=2011"
        / "offset=000000000.parquet"
    )
    daily = legacy.parents[1] / "task=20110104" / legacy.name
    legacy.parent.mkdir(parents=True, exist_ok=True)
    daily.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"trade_date": ["20110104"]}).to_parquet(legacy, index=False)
    pd.DataFrame({"trade_date": ["20110104"]}).to_parquet(daily, index=False)

    assert extended._year_page_paths(tmp_path, spec, 2011) == [daily]


def test_page_cache_resumes_without_repeating_completed_task(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(extended, "PAGE_SIZE", 2)
    first = pd.DataFrame(
        {
            "trade_date": ["20110104", "20110104"],
            "ts_code": ["000001.SZ", "600000.SH"],
            "value": [1.0, 2.0],
        }
    )
    second = pd.DataFrame(
        {
            "trade_date": ["20110104"],
            "ts_code": ["000002.SZ"],
            "value": [3.0],
        }
    )
    client = FakeClient([first, second])

    result = extended._download_task(tmp_path, _unit_spec(), "2011-01-04", client)
    reused = extended._download_task(tmp_path, _unit_spec(), "2011-01-04", client)

    assert result["row_count"] == 3
    assert result["page_count"] == 2
    assert reused["reused"] is True
    assert client.calls == 2


def test_provider_row_after_end_date_is_rejected(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20260102"],
            "ts_code": ["000001.SZ"],
            "value": [1.0],
        }
    )
    client = FakeClient([frame])

    with pytest.raises(extended.TushareExtendedBackfillError, match="forbidden_2026"):
        extended._download_task(tmp_path, _unit_spec(), "2025-12-31", client)


def test_state_writer_blocks_token_persistence(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QDP_TUSHARE_PROXY_TOKEN", "unit-secret")

    with pytest.raises(
        extended.TushareExtendedBackfillError,
        match="credential_persistence_blocked",
    ):
        extended._write_state(tmp_path, {"leaked": "unit-secret"})

    extended._write_state(tmp_path, {"status": "ok"})
    payload = json.loads(extended._state_path(tmp_path).read_text(encoding="utf-8"))
    assert "unit-secret" not in json.dumps(payload)


def test_factor_schema_rejects_partial_response() -> None:
    with pytest.raises(
        extended.TushareExtendedBackfillError, match="factor_schema_changed"
    ):
        extended._validate_schema(
            extended.SPECS["stk-factor-pro"],
            ["ts_code", "trade_date", "close"],
        )


def test_prepared_provider_projection_has_stable_semantic_types() -> None:
    numeric = extended._provider_value_expression(
        extended.SPECS["moneyflow"], alias="r", field="buy_sm_vol"
    )
    market_text = extended._provider_value_expression(
        extended.SPECS["margin"], alias="r", field="exchange_id"
    )
    membership_text = extended._provider_value_expression(
        extended.SPECS["margin-secs"], alias="r", field="name"
    )

    assert numeric == 'try_cast(r."buy_sm_vol" AS DOUBLE) AS "buy_sm_vol"'
    assert market_text == ('try_cast(r."exchange_id" AS VARCHAR) AS "exchange_id"')
    assert membership_text == 'try_cast(r."name" AS VARCHAR) AS "name"'


def test_uniform_prepared_schema_rejects_cross_year_drift(tmp_path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pq.write_table(pa.table({"value": pa.array([1.0], type=pa.float64())}), first)
    pq.write_table(pa.table({"value": pa.array([1], type=pa.int64())}), second)

    with pytest.raises(
        extended.TushareExtendedBackfillError, match="prepared_schema_drift"
    ):
        extended._assert_uniform_prepared_schema(
            extended.SPECS["moneyflow"],
            [
                {"path": str(first), "year": 2011},
                {"path": str(second), "year": 2012},
            ],
        )


def test_moneyflow_allows_negative_net_but_rejects_negative_gross(tmp_path) -> None:
    valid = tmp_path / "valid.parquet"
    invalid = tmp_path / "invalid.parquet"
    base = {
        "security_id": ["SZ-1"],
        "symbol": ["000001.SZ"],
        "ts_code": ["000001.SZ"],
        "trade_date": ["2011-01-04"],
        "source_date": ["2011-01-04"],
        "feature_available_date": ["2011-01-05"],
        "burn_in_only": [False],
        "source": ["unit"],
    }
    for field in extended.MONEYFLOW_FIELDS:
        if field not in base:
            base[field] = [1.0]
    base["buy_lg_amount"] = [10.0]
    base["buy_elg_amount"] = [5.0]
    base["sell_lg_amount"] = [8.0]
    base["sell_elg_amount"] = [9.0]
    base["net_mf_amount"] = [-2.0]
    base["net_mf_vol"] = [-1.0]
    pd.DataFrame(base).to_parquet(valid, index=False)

    result = extended._validate_prepared(valid, extended.SPECS["moneyflow"], 2011)
    assert result["allowed_negative_net_value_count"] == 2
    assert result["disallowed_negative_value_count"] == 0
    assert result["moneyflow_main_net_relation_mismatch_count"] == 0

    broken = pd.DataFrame(base)
    broken["buy_sm_amount"] = -1.0
    broken.to_parquet(invalid, index=False)
    with pytest.raises(
        extended.TushareExtendedBackfillError, match="prepared_contract_failed"
    ):
        extended._validate_prepared(invalid, extended.SPECS["moneyflow"], 2011)


def test_margin_market_preserves_documented_bse_rzche_exception(tmp_path) -> None:
    path = tmp_path / "margin.parquet"
    base = {
        "trade_date": ["2023-06-20"],
        "exchange_id": ["BSE"],
        "rzye": [10.0],
        "rzmre": [2.0],
        "rzche": [-1.0],
        "rqye": [0.0],
        "rqmcl": [0.0],
        "rzrqye": [10.0],
        "rqyl": [0.0],
        "source_date": ["2023-06-20"],
        "feature_available_date": ["2023-06-21"],
        "burn_in_only": [False],
        "source": ["unit"],
    }
    pd.DataFrame(base).to_parquet(path, index=False)

    result = extended._validate_prepared(path, extended.SPECS["margin"], 2023)
    assert result["disallowed_negative_value_count"] == 0
    assert result["source_exception_negative_value_count"] == 1

    broken = pd.DataFrame(base)
    broken["exchange_id"] = "SSE"
    broken.to_parquet(path, index=False)
    with pytest.raises(
        extended.TushareExtendedBackfillError, match="prepared_contract_failed"
    ):
        extended._validate_prepared(path, extended.SPECS["margin"], 2023)


def test_self_test_locks_download_contract() -> None:
    result = extended.self_test()

    assert result["status"] == "ok"
    assert result["checks"]["factor_field_count"] == 261
    assert result["checks"]["forbidden_2026"] is True


def test_cleanup_prepared_cache_deletes_only_verified_installed_copy(tmp_path) -> None:
    domain = "unit_domain"
    dataset_id = "unit_domain__installed"
    prepared = (
        extended._runtime(tmp_path)
        / "prepared"
        / domain
        / "year=2011"
        / "part-0000.parquet"
    )
    installed = (
        qdp_v2_root(tmp_path)
        / "datasets"
        / domain
        / dataset_id
        / "shards"
        / "year=2011"
        / "part-0000.parquet"
    )
    prepared.parent.mkdir(parents=True, exist_ok=True)
    installed.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_bytes(b"verified-derived-copy")
    installed.write_bytes(prepared.read_bytes())
    digest = extended._sha256(prepared)
    extended.atomic_write_json(
        prepared.with_suffix(".json"),
        {"sha256": digest, "input_hash": "unit"},
    )
    root = qdp_v2_root(tmp_path)
    write_dataset_manifest(
        root,
        DatasetManifest(
            dataset_id=dataset_id,
            domain=domain,
            layer="raw",
            frequency="1d",
            contract_version="unit_v1",
            primary_key=["trade_date"],
            start_date="2011-01-04",
            end_date="2011-01-04",
            row_count=1,
            shards=[
                ShardManifestEntry(
                    path=str(installed.relative_to(root)).replace("\\", "/"),
                    row_count=1,
                    start_date="2011-01-04",
                    end_date="2011-01-04",
                    file_size=installed.stat().st_size,
                    metadata={"year": 2011, "sha256": digest},
                )
            ],
            source={"provider": "unit"},
            quality={},
        ),
    )
    write_active_manifest(root, {"datasets": {domain: dataset_id}})
    extended._write_state(
        tmp_path,
        {
            "status": "applied",
            "installed_domains": {
                domain: {"status": "installed", "dataset_id": dataset_id}
            },
        },
    )

    dry = extended.cleanup_prepared_cache(workspace_root=tmp_path)
    deleted = extended.cleanup_prepared_cache(
        workspace_root=tmp_path,
        delete=True,
        yes=True,
    )

    assert dry["status"] == "dry_run"
    assert dry["deletable"] is True
    assert deleted["status"] == "deleted"
    assert deleted["destructive_actions_performed"] is True
    assert not prepared.exists()
    assert installed.is_file()
    assert Path(deleted["receipt_path"]).is_file()


def test_credential_stdin_flag_never_accepts_token_value() -> None:
    parser = extended.build_arg_parser()

    args = parser.parse_args(["--credential-stdin", "--status"])

    assert args.credential_stdin is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--credential-stdin=secret", "--status"])
