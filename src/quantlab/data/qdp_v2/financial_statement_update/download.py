"""Financial Statement Update: download responsibilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from quantlab.data.domains.contracts import DataDomain
from quantlab.data.qdp_v2.auxiliary_update import (
    _resolve_tushare_token,
    _TushareClient,
)
from quantlab.data.qdp_v2.manifest import (
    qdp_v2_root,
    read_active_manifest,
)
from quantlab.data.qdp_v2.research_event_update import (
    _sha256,
    _write_parquet,
)
from quantlab.data.qdp_v2.status import active_dataset_map

from .config import (
    END_DATE,
    MAX_REPORT_PERIOD,
    SOURCE_SCHEMA_VERSION,
    START_DATE,
    UPDATE_ID,
    V2_STATEMENT_SPECS,
)
from .context import (
    _in_scope_provider_rows,
    _raw_path,
    _read_state,
    _report_periods,
    _workspace,
    _write_state,
)


def download(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    if (
        state.get("status") in {"downloaded", "prepared", "applied"}
        and state.get("source_schema_version") == SOURCE_SCHEMA_VERSION
    ):
        return state
    if state.get("source_schema_version") != SOURCE_SCHEMA_VERSION:
        previous_schema_version = state.get("source_schema_version")
        current = active_dataset_map(read_active_manifest(qdp_v2_root(workspace)))
        state = {
            "update_id": UPDATE_ID,
            "status": "pending",
            "start_date": START_DATE,
            "end_date": END_DATE,
            "source_schema_version": SOURCE_SCHEMA_VERSION,
            "upgraded_from_source_schema_version": previous_schema_version,
            "input_dataset_ids": {
                domain: dataset_id
                for domain, dataset_id in current.items()
                if domain
                in {
                    DataDomain.INCOME_STATEMENT_QUARTERLY,
                    DataDomain.BALANCE_SHEET_QUARTERLY,
                    DataDomain.CASH_FLOW_STATEMENT_QUARTERLY,
                }
            },
            "completed": {},
            "provider_future_rows_discarded": 0,
        }
        _write_state(workspace, state)
    client = _TushareClient(_resolve_tushare_token(workspace), workspace_root=workspace)
    periods = _report_periods()
    completed = dict(state.get("completed", {}) or {})
    if "input_dataset_ids" not in state:
        state["input_dataset_ids"] = {
            domain: dataset_id
            for domain, dataset_id in active_dataset_map(read_active_manifest(qdp_v2_root(workspace))).items()
            if domain
            in {
                DataDomain.INCOME_STATEMENT_QUARTERLY,
                DataDomain.BALANCE_SHEET_QUARTERLY,
                DataDomain.CASH_FLOW_STATEMENT_QUARTERLY,
            }
        }
    discarded_future_rows = int(state.get("provider_future_rows_discarded", 0) or 0)
    completed_calls = 0
    for spec in V2_STATEMENT_SPECS:
        spec_completed = dict(completed.get(spec.name, {}) or {})
        for period in periods:
            path = _raw_path(workspace, spec, period)
            current_fields = set(pq.read_schema(path).names) if path.is_file() else set()
            if not set(spec.fields).issubset(current_fields):
                raw = client.fetch(
                    spec.api_name,
                    params={"period": period},
                    fields=spec.fields,
                )
                scoped, future = _in_scope_provider_rows(raw)
                discarded_future_rows += future
                _write_parquet(scoped, path)
            spec_completed[period] = {
                "status": "completed",
                "path": str(path),
                "row_count": int(pq.ParquetFile(path).metadata.num_rows),
                "sha256": _sha256(path),
            }
            completed_calls += 1
            completed[spec.name] = spec_completed
            state.update(
                {
                    "update_id": UPDATE_ID,
                    "status": "downloading",
                    "start_date": START_DATE,
                    "end_date": END_DATE,
                    "maximum_report_period": MAX_REPORT_PERIOD,
                    "source_schema_version": SOURCE_SCHEMA_VERSION,
                    "period_count": len(periods),
                    "completed": completed,
                    "provider_future_rows_discarded": discarded_future_rows,
                    "provider_response_future_rows_not_persisted": True,
                }
            )
            _write_state(workspace, state)
            if completed_calls % 12 == 0:
                print(
                    json.dumps(
                        {
                            "completed_calls": completed_calls,
                            "total_calls": len(periods) * len(V2_STATEMENT_SPECS),
                        }
                    ),
                    flush=True,
                )
    state["status"] = "downloaded"
    state["source_schema_version"] = SOURCE_SCHEMA_VERSION
    _write_state(workspace, state)
    return state
