from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from quant_data_platform.qdp_v2 import auxiliary_tail_update, auxiliary_update
from quant_data_platform.qdp_v2.audit import _manifest_contract_findings
from quant_data_platform.qdp_v2.auxiliary_tail_update import (
    _normalize_cninfo_share_change,
    _only_changed_share_events,
)
from quant_data_platform.qdp_v2.auxiliary_update import (
    _baostock_snapshot_worker,
    _normalize_daily_basic,
    _normalize_cninfo_dividend,
    _normalize_cninfo_industry_history,
    _normalize_comparison_text,
    _normalize_dividend,
    _normalize_industry_comparison,
    _normalize_name_intervals,
    _valuation_secondary_policy,
)
from quant_data_platform.qdp_v2.baostock_update import _valuation_frame
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    qdp_v2_root,
    read_dataset_manifest,
    write_active_manifest,
    write_dataset_manifest,
)
from quant_data_platform.qdp_v2.repair import (
    replace_active_table_from_parquet,
    update_active_manifest_metadata,
)


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    brain = workspace / "brain" / "brain_manifest.json"
    brain.parent.mkdir(parents=True, exist_ok=True)
    brain.write_text(json.dumps({"schema_version": 1, "brain_type": "main"}))
    return workspace


def test_daily_basic_normalization_converts_10k_units_to_base_units() -> None:
    frame = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20260716"],
            "close": [10.0],
            "turnover_rate": [1.2],
            "pe_ttm": [8.0],
            "pb": [1.0],
            "total_share": [123.5],
            "float_share": [100.25],
            "total_mv": [1235.0],
            "circ_mv": [1002.5],
        }
    )

    result = _normalize_daily_basic(frame, "2026-07-16").iloc[0]

    assert result["trade_date"] == "2026-07-16"
    assert result["total_share"] == 1_235_000.0
    assert result["float_share"] == 1_002_500.0
    assert result["total_mv"] == 12_350_000.0
    assert result["circ_mv"] == 10_025_000.0


def test_baostock_core_valuation_cache_reuses_bulk_fields() -> None:
    raw = pd.DataFrame(
        {
            "code": ["sh.600000", "sz.000001"],
            "peTTM": [6.5, 7.5],
            "pbMRQ": [0.7, 0.8],
            "turn": [0.2, 0.3],
        }
    )

    result = _valuation_frame(raw, trade_date="2026-07-16")

    assert result["symbol"].tolist() == ["600000.SH", "000001.SZ"]
    assert result["pe"].tolist() == [6.5, 7.5]
    assert result["pb"].tolist() == [0.7, 0.8]
    assert result["turnover_rate"].tolist() == [0.2, 0.3]


def test_valuation_secondary_policy_only_blocks_comparable_turnover() -> None:
    policy = _valuation_secondary_policy(
        comparison_counts={"pe": 1000, "pb": 1000, "turnover_rate": 1000},
        mismatch_counts={"pe": 30, "pb": 80, "turnover_rate": 9},
        median_ratios={"pe": 1.0, "pb": 1.0, "turnover_rate": 1.0},
    )

    assert policy["secondary_validation_status"] == "not_comparable"
    assert policy["secondary_not_comparable_metrics"] == ["pe", "pb"]
    assert policy["comparable_failures"] == []

    failed = _valuation_secondary_policy(
        comparison_counts={"pe": 1000, "pb": 1000, "turnover_rate": 1000},
        mismatch_counts={"pe": 0, "pb": 0, "turnover_rate": 11},
        median_ratios={"pe": 1.0, "pb": 1.0, "turnover_rate": 1.0},
    )
    assert failed["comparable_failures"] == ["turnover_rate"]


def test_quick_audit_accepts_strict_pit_valuation_contract() -> None:
    schema = [
        {"name": name}
        for name in (
            "symbol",
            "trade_date",
            "total_mv",
            "circ_mv",
            "pe",
            "pb",
            "turnover_rate",
            "source",
        )
    ]

    errors, _ = _manifest_contract_findings(
        "valuation",
        {
            "dataset_id": "valuation__unit",
            "contract_version": "qdp_v2_valuation_strict_pit_v3",
            "schema": schema,
            "quality": {"primary_key_unique": True},
        },
    )

    assert not [item for item in errors if item.startswith("wrong_contract:")]


def test_dividend_normalization_uses_ex_date_and_only_implemented_rows() -> None:
    frame = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "end_date": ["20251231", "20261231"],
            "ann_date": ["20260301", "20270301"],
            "imp_ann_date": ["20260305", None],
            "div_proc": ["实施", "预案"],
            "stk_div": [0.2, 0.0],
            "stk_bo_rate": [0.1, None],
            "stk_co_rate": [0.1, None],
            "cash_div_tax": [0.4, 0.0],
            "cash_div": [0.3, 0.0],
            "record_date": ["20260310", None],
            "ex_date": ["20260311", None],
            "pay_date": ["20260311", None],
        }
    )

    result = _normalize_dividend(frame, target_date="2026-07-16")

    assert len(result) == 1
    row = result.iloc[0]
    assert row["trade_date"] == "2026-03-11"
    assert row["ex_date"] == "2026-03-11"
    assert row["cash_dividend_per_10"] == 4.0
    assert row["bonus_share_per_10"] == 1.0
    assert row["transfer_share_per_10"] == 1.0
    assert row["action_type"] == "cash_stock"


def test_name_interval_normalization_preserves_pit_ranges() -> None:
    frame = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "name": ["Old Name", "New Name", "Future Name"],
            "start_date": ["20100104", "20150105", "20270101"],
            "end_date": ["20150104", None, None],
            "ann_date": ["20100104", "20150105", "20270101"],
            "change_reason": ["initial", "rename", "future"],
        }
    )

    result = _normalize_name_intervals(frame, target_date="2026-07-16")

    assert result["start_date"].tolist() == ["2010-01-04", "2015-01-05"]
    assert result["name"].tolist() == ["Old Name", "New Name"]
    assert result["end_date"].tolist()[0] == "2015-01-04"


def test_cninfo_dividend_normalization_keeps_per_10_units() -> None:
    frame = pd.DataFrame(
        {
            "实施方案公告日期": ["2026-07-10"],
            "送股比例": [1.0],
            "转增比例": [2.0],
            "派息比例": [4.2],
            "股权登记日": ["2026-07-15"],
            "除权日": ["2026-07-16"],
            "派息日": ["2026-07-16"],
            "实施方案分红说明": ["10派4.2元"],
        }
    )

    row = _normalize_cninfo_dividend(
        frame,
        symbol="600000.SH",
        target_date="2026-07-16",
    ).iloc[0]

    assert row["trade_date"] == "2026-07-16"
    assert row["cash_dividend_per_10"] == 4.2
    assert row["bonus_share_per_10"] == 1.0
    assert row["transfer_share_per_10"] == 2.0
    assert row["action_type"] == "cash_stock"
    assert _normalize_comparison_text(" 货币 金融服务 ") == "货币金融服务"
    assert _normalize_industry_comparison("F51批发业") == "批发业"


def test_cninfo_industry_history_keeps_only_csrc_large_class() -> None:
    frame = pd.DataFrame(
        {
            "变更日期": ["2008-01-01", "2008-01-02"],
            "分类标准编码": ["008021", "008003"],
            "分类标准": ["证监会行业分类标准（2012）", "申银万国行业分类标准"],
            "行业大类": ["资本市场服务", "证券"],
        }
    )

    result = _normalize_cninfo_industry_history(
        frame,
        symbol="001236.SZ",
        target_date="2026-07-16",
    )

    assert result.to_dict("records") == [
        {
            "symbol": "001236.SZ",
            "source_date": "2008-01-01",
            "industry": "资本市场服务",
            "industry_standard": "证监会行业分类",
            "source": "akshare_cninfo_csrc_industry_history",
        }
    ]


def test_mootdx_share_detection_ignores_unchanged_xdxr_rows() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["600000.SH"] * 3,
            "trade_date": ["2026-01-01", "2026-02-01", "2026-03-01"],
            "total_share": [100.0, 100.0, 101.0],
            "float_share": [90.0, 90.0, 91.0],
        }
    )

    result = _only_changed_share_events(frame)

    assert result["trade_date"].tolist() == ["2026-01-01", "2026-03-01"]


def test_cninfo_share_change_uses_jointly_visible_source_date() -> None:
    frame = pd.DataFrame(
        {
            "变动日期": ["2025-12-31"],
            "公告日期": ["2026-03-31"],
            "总股本": [123.5],
            "已流通股份": [100.25],
        }
    )

    row = _normalize_cninfo_share_change(
        frame,
        symbol="000001.SZ",
        target_date="2026-07-16",
    ).iloc[0]

    assert row["variation_date"] == "2025-12-31"
    assert row["source_date"] == "2026-03-31"
    assert row["total_share"] == 1_235_000.0
    assert row["float_share"] == 1_002_500.0


def test_cninfo_share_change_prefers_domestic_a_share_float() -> None:
    frame = pd.DataFrame(
        {
            "变动日期": ["2026-06-30"],
            "公告日期": ["2026-07-10"],
            "总股本": [120_925.0],
            "人民币普通股": [94_752.0],
            "已流通股份": [118_535.0],
        }
    )

    row = _normalize_cninfo_share_change(
        frame,
        symbol="600028.SH",
        target_date="2026-07-16",
    ).iloc[0]

    assert row["total_share"] == 1_209_250_000.0
    assert row["float_share"] == 947_520_000.0


def test_cninfo_share_change_never_backfills_future_announcement() -> None:
    frame = pd.DataFrame(
        {
            "变动日期": ["2010-06-30", "2010-08-09"],
            "公告日期": ["2010-08-10", "2010-08-11"],
            "总股本": [100.0, 200.0],
            "已流通股份": [80.0, 160.0],
        }
    )

    result = _normalize_cninfo_share_change(
        frame,
        symbol="000001.SZ",
        target_date="2010-08-10",
    )

    assert result[["source_date", "total_share", "float_share"]].to_dict(
        "records"
    ) == [
        {
            "source_date": "2010-08-10",
            "total_share": 1_000_000.0,
            "float_share": 800_000.0,
        }
    ]


def test_baostock_snapshot_worker_checkpoints_each_date(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class Query:
        error_code = "0"
        fields = ["code", "industry", "industryClassification", "updateDate"]

        def __init__(self) -> None:
            self._remaining = [["sh.600000", "bank", "CSRC", "2026-07-15"]]
            self._current: list[str] = []

        def next(self) -> bool:
            if not self._remaining:
                return False
            self._current = self._remaining.pop(0)
            return True

        def get_row_data(self) -> list[str]:
            return self._current

    def query_stock_industry(*, date: str):
        if date == "2026-07-16":
            raise RuntimeError("injected failure")
        return Query()

    fake_baostock = SimpleNamespace(
        login=lambda: SimpleNamespace(error_code="0"),
        logout=lambda: None,
        query_stock_industry=query_stock_industry,
    )
    monkeypatch.setitem(sys.modules, "baostock", fake_baostock)
    monkeypatch.setattr(auxiliary_update.time, "sleep", lambda _: None)

    output_dir = tmp_path / "parts"
    result = _baostock_snapshot_worker(
        "industry",
        ["2026-07-15", "2026-07-16"],
        str(output_dir),
    )

    checkpoint = output_dir / "snapshot_20260715.parquet"
    assert checkpoint.is_file()
    assert not (output_dir / "snapshot_20260716.parquet").exists()
    assert result["failure_dates"] == ["2026-07-16"]
    assert pd.read_parquet(checkpoint).to_dict("records") == [
        {
            "symbol": "600000.SH",
            "snapshot_query_date": "2026-07-15",
            "industry": "bank",
            "industry_standard": "CSRC",
            "source_date": "2026-07-15",
            "source": "baostock.query_stock_industry",
        }
    ]


def test_index_tail_dates_casts_parquet_dates_before_max(
    tmp_path: Path,
    monkeypatch,
) -> None:
    current = tmp_path / "current_index.parquet"
    calendar = tmp_path / "calendar.parquet"
    with duckdb.connect() as con:
        con.execute("CREATE TABLE current_index(trade_date CHAR(10))")
        con.execute(
            "INSERT INTO current_index VALUES ('2026-07-15'), ('2026-07-16')"
        )
        con.execute("COPY current_index TO ? (FORMAT PARQUET)", [str(current)])
        assert con.execute(
            "SELECT max(trade_date) FROM read_parquet(?)", [str(current)]
        ).fetchone()[0] == "2026-07-"
    pd.DataFrame(
        {
            "trade_date": [
                "2026-07-15",
                "2026-07-16",
                "2026-07-17",
                "2026-07-20",
                "2026-07-21",
            ],
            "exchange": ["SSE"] * 5,
            "is_open": [True] * 5,
        }
    ).to_parquet(calendar, index=False)
    monkeypatch.setattr(
        auxiliary_tail_update,
        "_paths",
        lambda ctx, domain: [
            current if domain == "index_constituents" else calendar
        ],
    )
    ctx = SimpleNamespace(
        runtime=tmp_path / "runtime",
        target_date="2026-07-21",
    )

    assert auxiliary_tail_update._index_tail_dates(ctx) == [
        "2026-07-17",
        "2026-07-20",
        "2026-07-21",
    ]


def test_replace_active_table_allows_schema_and_primary_key_migration(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    dataset_id = "industry_concept__unit"
    dataset_dir = root / "datasets" / "industry_concept" / dataset_id
    old = dataset_dir / "shards" / "old.parquet"
    old.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "trade_date": ["2026-01-05"],
            "industry": ["bank"],
        }
    ).to_parquet(old, index=False)
    write_dataset_manifest(
        root,
        DatasetManifest(
            dataset_id=dataset_id,
            domain="industry_concept",
            layer="raw",
            frequency="1d",
            contract_version="v1",
            primary_key=["trade_date", "symbol", "industry"],
            start_date="2026-01-05",
            end_date="2026-01-05",
            row_count=1,
            schema_hash="old",
            shards=[
                ShardManifestEntry(
                    path=str(old.relative_to(root)).replace("\\", "/"),
                    row_count=1,
                    start_date="2026-01-05",
                    end_date="2026-01-05",
                )
            ],
            source={"provider": "unit"},
            quality={},
        ),
    )
    write_active_manifest(
        root,
        {
            "version": 2,
            "active_as_of_date": "2026-01-05",
            "datasets": {"industry_concept": dataset_id},
        },
    )
    prepared = workspace / "quant_data_platform" / "data" / "qdp_runtime" / "prepared.parquet"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "trade_date": ["2026-01-05"],
            "industry": ["bank"],
            "industry_standard": ["CSRC"],
        }
    ).to_parquet(prepared, index=False)

    result = replace_active_table_from_parquet(
        "industry_concept",
        prepared,
        reason="unit schema migration",
        workspace_root=workspace,
        primary_key=["trade_date", "symbol"],
        contract_version="v2",
        source_updates={"checked_through": "2026-01-05"},
    )

    manifest = read_dataset_manifest(dataset_dir / "dataset.json")
    assert result["status"] == "replaced"
    assert manifest.dataset_id == dataset_id
    assert manifest.primary_key == ["trade_date", "symbol"]
    assert manifest.contract_version == "v2"
    assert manifest.source["checked_through"] == "2026-01-05"
    assert "industry_standard" in pq.read_schema(root / manifest.shards[0].path).names
    assert not old.exists()

    metadata_result = update_active_manifest_metadata(
        "industry_concept",
        reason="unit secondary validation",
        workspace_root=workspace,
        source_updates={
            "secondary_validation_status": "ok",
            "secondary_compared_count": 200,
        },
    )
    updated = read_dataset_manifest(dataset_dir / "dataset.json")
    assert metadata_result["status"] == "metadata_updated"
    assert updated.dataset_id == dataset_id
    assert updated.row_count == 1
    assert updated.shards == manifest.shards
    assert updated.source["secondary_validation_status"] == "ok"
    assert updated.source["secondary_compared_count"] == 200
