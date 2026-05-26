from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DAILY_RESEARCH_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = DAILY_RESEARCH_ROOT.parent
ACTIVE_MANIFEST_PATH = DAILY_RESEARCH_ROOT / "output" / "active_execution_strategy.json"
DEFAULT_FULLFIT_RUN_DIR = DAILY_RESEARCH_ROOT / "output" / "short_expert_policy_v5b_execalign_production_fullfit_20260421_r1"
DEFAULT_PRODUCTION_ROOT = DAILY_RESEARCH_ROOT / "output" / "short_expert_policy_v5b_execalign_production_default"
OLD_DAILY_RESEARCH_ROOTS = (
    r"H:\new_tdx64\PYPlugins\user\daily_research",
    "H:/new_tdx64/PYPlugins/user/daily_research",
)
ANCHOR_COPY_FILES = (
    "metrics.json",
    "deep_alpha_model.pt",
    "score_head_artifact.pkl",
    "risk_gate_artifact.pkl",
    "execution_alignment_artifact.pkl",
    "train_history.csv",
    "daily_live_score_panel.csv",
    "daily_live_target_weight_panel.csv",
    "portfolio_capped_daily_live_target_weight_panel.csv",
    "execution_aligned_daily_live_score_panel.csv",
    "execution_aligned_daily_live_target_weight_panel.csv",
    "live_latest_scores.csv",
    "execution_aligned_live_latest_scores.csv",
)


def _now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def file_sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_workspace_path_text(raw: Any, *, project_root: Path = DAILY_RESEARCH_ROOT) -> str:
    text = str(raw or "")
    if not text:
        return ""
    replacement = str(Path(project_root).resolve())
    normalized_replacement = replacement.replace("\\", "/")
    out = text
    for old_root in OLD_DAILY_RESEARCH_ROOTS:
        out = out.replace(old_root, replacement)
        out = out.replace(old_root.replace("\\", "/"), normalized_replacement)
    return out


def normalize_workspace_paths(payload: Any, *, project_root: Path = DAILY_RESEARCH_ROOT) -> Any:
    if isinstance(payload, dict):
        return {key: normalize_workspace_paths(value, project_root=project_root) for key, value in payload.items()}
    if isinstance(payload, list):
        return [normalize_workspace_paths(value, project_root=project_root) for value in payload]
    if isinstance(payload, str):
        return normalize_workspace_path_text(payload, project_root=project_root)
    return payload


def count_legacy_workspace_paths(payload: Any) -> int:
    if isinstance(payload, dict):
        return sum(count_legacy_workspace_paths(value) for value in payload.values())
    if isinstance(payload, list):
        return sum(count_legacy_workspace_paths(value) for value in payload)
    if isinstance(payload, str):
        text = payload.replace("/", "\\")
        return int(r"H:\new_tdx64\PYPlugins\user\daily_research" in text)
    return 0


def _date_text(raw: Any) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    try:
        parsed = pd.Timestamp(value)
    except Exception:
        return value[:10]
    if pd.isna(parsed):
        return value[:10]
    return str(parsed.date())


def panel_latest_date(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        frame = pd.read_csv(path, usecols=["date"])
    except Exception:
        return ""
    if frame.empty:
        return ""
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return ""
    return str(pd.Timestamp(dates.max()).date())


def panel_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return int(sum(1 for _ in handle) - 1)
    except Exception:
        return 0


def _positive_target_count(path: Path, *, latest_date: str) -> int:
    if not path.exists() or not latest_date:
        return 0
    try:
        frame = pd.read_csv(path)
    except Exception:
        return 0
    if frame.empty or "date" not in frame.columns or "target_weight" not in frame.columns:
        return 0
    dates = pd.to_datetime(frame["date"], errors="coerce")
    selected = frame.loc[dates.eq(pd.Timestamp(latest_date))]
    weights = pd.to_numeric(selected["target_weight"], errors="coerce").fillna(0.0)
    return int((weights > 0.0).sum())


def _sector_board_view_records(lake: Any) -> pd.DataFrame:
    frame = lake.list_datasets(dataset_kind="policy_sector_board_view")
    if frame is None or frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    if "status" in working.columns:
        status = working["status"].astype(str).str.lower()
        working = working.loc[~status.isin({"blocked", "failed", "error"})].copy()
    return working


def _latest_sector_board_view_id_for_dataset(lake: Any, lake_dataset_id: str) -> str:
    working = _sector_board_view_records(lake)
    if working.empty or "dataset_id" not in working.columns:
        return ""
    if "source" in working.columns:
        working = working.loc[working["source"].astype(str).eq(str(lake_dataset_id))].copy()
    if working.empty:
        return ""
    working["_created_at_sort"] = pd.to_datetime(working.get("created_at", ""), errors="coerce")
    working = working.sort_values(["_created_at_sort", "dataset_id"], ascending=[False, False])
    return str(working.iloc[0].get("dataset_id", "") or "")


def _read_sector_view_frame(paths: dict[str, Any], key: str) -> pd.DataFrame:
    path_text = normalize_workspace_path_text(paths.get(key, ""))
    path = Path(path_text) if path_text else Path("")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _dataset_symbols(lake: Any, lake_dataset_id: str) -> list[str]:
    metadata = lake.describe_dataset(str(lake_dataset_id))
    market_path = normalize_workspace_path_text(dict(metadata.get("content_paths", {}) or {}).get("bronze_market_data", ""))
    if not market_path or not Path(market_path).exists():
        raise ValueError(f"policy input bundle has no readable bronze_market_data: {lake_dataset_id}")
    market = pd.read_parquet(market_path, columns=["symbol"])
    symbols = sorted(str(item).strip().upper() for item in market["symbol"].dropna().astype(str).unique())
    return [item for item in symbols if item]


def _ensure_sector_board_view_for_dataset(*, data_lake_root: Path | str, lake_dataset_id: str) -> str:
    from daily_research.data_lake import ResearchDataLake

    lake = ResearchDataLake(data_lake_root)
    existing = _latest_sector_board_view_id_for_dataset(lake, lake_dataset_id)
    if existing:
        return existing

    records = _sector_board_view_records(lake)
    if records.empty or "dataset_id" not in records.columns:
        raise RuntimeError(f"no sector board view is available for lake dataset {lake_dataset_id}")
    records["_created_at_sort"] = pd.to_datetime(records.get("created_at", ""), errors="coerce")
    records = records.sort_values(["_created_at_sort", "dataset_id"], ascending=[False, False])
    source_view_id = str(records.iloc[0].get("dataset_id", "") or "")
    source_meta = lake.describe_dataset(source_view_id)
    source_paths = dict(source_meta.get("content_paths", {}) or {})
    industry = _read_sector_view_frame(source_paths, "industry_map")
    board = _read_sector_view_frame(source_paths, "board_membership")
    summary = _read_sector_view_frame(source_paths, "board_summary")
    if industry.empty or board.empty:
        raise RuntimeError(f"latest sector board view has no readable frames: {source_view_id}")

    symbols = set(_dataset_symbols(lake, lake_dataset_id))
    industry = industry.loc[industry["symbol"].astype(str).str.upper().isin(symbols)].copy()
    board = board.loc[board["symbol"].astype(str).str.upper().isin(symbols)].copy()
    if not summary.empty and {"board_kind", "board_name", "board_code"}.issubset(board.columns):
        keys = board[["board_kind", "board_name", "board_code"]].drop_duplicates()
        if {"board_kind", "board_name", "board_code"}.issubset(summary.columns):
            summary = summary.merge(keys, on=["board_kind", "board_name", "board_code"], how="inner")

    parameters = dict(source_meta.get("parameters", {}) or {})
    spec = {
        "source_market_dataset_id": str(lake_dataset_id),
        "view_kind": str(parameters.get("view_kind", "") or "latest_static_snapshot"),
        "view_name": str(parameters.get("view_name", "") or "sector_board_latest_static"),
        "snapshot_semantics": str(parameters.get("snapshot_semantics", "") or "latest_static_snapshot"),
        "as_of_date": str(parameters.get("as_of_date", "") or source_meta.get("created_at", "") or ""),
    }
    record = lake.save_sector_board_view(
        spec=spec,
        industry_map_frame=industry,
        board_membership_frame=board,
        board_summary_frame=summary,
        source_cache={
            "cloned_from_sector_board_view_id": source_view_id,
            "clone_reason": "production_signal_refresh_dataset_roll_forward",
        },
    )
    return str(record.dataset_id)


def audit_production_anchor(
    *,
    production_root: Path | str = DEFAULT_PRODUCTION_ROOT,
    fullfit_run_dir: Path | str = DEFAULT_FULLFIT_RUN_DIR,
    project_root: Path | str = DAILY_RESEARCH_ROOT,
) -> dict[str, Any]:
    production_root = Path(production_root)
    fullfit_run_dir = Path(fullfit_run_dir)
    project_root = Path(project_root)
    production_model = production_root / "deep_alpha_model.pt"
    fullfit_model = fullfit_run_dir / "deep_alpha_model.pt"
    production_hash = file_sha256(production_model)
    fullfit_hash = file_sha256(fullfit_model)
    production_metrics = read_json(production_root / "metrics.json")
    fullfit_metrics = read_json(fullfit_run_dir / "metrics.json")
    production_manifest = read_json(production_root / "production_retrain_manifest.json")
    model_hash_match = bool(production_hash and fullfit_hash and production_hash == fullfit_hash)
    metrics_train_end_match = _date_text(production_metrics.get("train_end")) == _date_text(fullfit_metrics.get("train_end"))
    old_path_count = count_legacy_workspace_paths(production_metrics) + count_legacy_workspace_paths(production_manifest)
    missing = [
        str(path)
        for path in (production_model, fullfit_model, production_root / "metrics.json", fullfit_run_dir / "metrics.json")
        if not path.exists()
    ]
    status = "missing" if missing else "ok"
    if not missing and (not model_hash_match or not metrics_train_end_match or old_path_count > 0):
        status = "stale_or_mismatched"
    return {
        "status": status,
        "production_root": str(production_root.resolve()),
        "fullfit_run_dir": str(fullfit_run_dir.resolve()),
        "project_root": str(project_root.resolve()),
        "model_hash_match": model_hash_match,
        "production_model_sha256": production_hash,
        "fullfit_model_sha256": fullfit_hash,
        "metrics_train_end_match": metrics_train_end_match,
        "production_train_end": _date_text(production_metrics.get("train_end")),
        "fullfit_train_end": _date_text(fullfit_metrics.get("train_end")),
        "old_path_count": int(old_path_count),
        "missing_paths": missing,
    }


def _build_production_manifest(
    *,
    production_root: Path,
    fullfit_run_dir: Path,
    active_manifest: dict[str, Any],
    fullfit_metrics: dict[str, Any],
    anchor_sync_manifest_path: Path,
) -> dict[str, Any]:
    latest_panel_date = panel_latest_date(production_root / "execution_aligned_daily_live_target_weight_panel.csv")
    return {
        "mode": "production_fullfit",
        "source": "existing_fullfit_anchor_sync",
        "source_fullfit_run_dir": str(fullfit_run_dir.resolve()),
        "source_formal_run_dir": normalize_workspace_path_text(active_manifest.get("source_formal_run_dir", ""), project_root=DAILY_RESEARCH_ROOT),
        "active_production_run_dir": str(fullfit_run_dir.resolve()),
        "production_root": str(production_root.resolve()),
        "active_execution_strategy_manifest": str(ACTIVE_MANIFEST_PATH.resolve()),
        "production_anchor_sync_manifest": str(anchor_sync_manifest_path.resolve()),
        "target_weight_semantics": str(active_manifest.get("target_weight_semantics", "research_raw_target_weight") or "research_raw_target_weight"),
        "target_weight_cap_mode": str(active_manifest.get("target_weight_cap_mode", "follow_research_raw_no_global_cap") or "follow_research_raw_no_global_cap"),
        "target_weight_cap_note": str(active_manifest.get("target_weight_cap_note", "") or ""),
        "score_panel_role": str(active_manifest.get("score_panel_role", "") or ""),
        "raw_live_target_weight_panel_csv": str((production_root / "daily_live_target_weight_panel.csv").resolve()),
        "raw_live_score_panel_csv": str((production_root / "daily_live_score_panel.csv").resolve()),
        "trade_plan_target_weight_panel_csv": str((production_root / "execution_aligned_daily_live_target_weight_panel.csv").resolve()),
        "trade_plan_score_panel_csv": str((production_root / "execution_aligned_daily_live_score_panel.csv").resolve()),
        "train_start_date": str(fullfit_metrics.get("train_start", fullfit_metrics.get("start_date", "")) or ""),
        "train_end_date": str(fullfit_metrics.get("valid_end", fullfit_metrics.get("train_end", "")) or ""),
        "model_train_end": str(fullfit_metrics.get("train_end", "") or ""),
        "launch_cutoff_date": latest_panel_date.replace("-", "") if latest_panel_date else "",
        "liquidity_pool": str(fullfit_metrics.get("liquidity_pool", "") or active_manifest.get("liquidity_pool", "") or ""),
        "rolling_liquidity_pool": str(fullfit_metrics.get("rolling_liquidity_pool", "") or active_manifest.get("liquidity_pool", "") or ""),
        "rolling_pool_rebalance_days": int(fullfit_metrics.get("rolling_pool_rebalance_days", 21) or 21),
        "rolling_pool_adv_window": int(fullfit_metrics.get("rolling_pool_adv_window", 20) or 20),
        "daily_pool_refresh_enabled": False,
        "data_source": "lake",
        "lake_dataset_id": str(active_manifest.get("lake_dataset_id", "") or active_manifest.get("source_market_dataset_id", "") or ""),
        "data_lake_root": str(active_manifest.get("data_lake_root", "") or (DAILY_RESEARCH_ROOT / "output" / "research_data_lake")),
        "retrain_frequency_policy": {
            "auto_retrain_enabled": False,
            "preferred_cadence": "explicit_user_submitted_only",
            "notes": "Production retrain is disabled from trade-plan and data refresh paths.",
        },
        "created_at": _now_text(),
    }


def sync_production_anchor_to_fullfit(
    *,
    active_manifest_path: Path | str = ACTIVE_MANIFEST_PATH,
    production_root: Path | str = DEFAULT_PRODUCTION_ROOT,
    fullfit_run_dir: Path | str = DEFAULT_FULLFIT_RUN_DIR,
    update_active_manifest: bool = True,
) -> dict[str, Any]:
    active_manifest_path = Path(active_manifest_path)
    production_root = Path(production_root)
    fullfit_run_dir = Path(fullfit_run_dir)
    before = audit_production_anchor(production_root=production_root, fullfit_run_dir=fullfit_run_dir)
    active_manifest = read_json(active_manifest_path)
    if not active_manifest:
        raise FileNotFoundError(f"active execution manifest missing: {active_manifest_path}")
    if not fullfit_run_dir.exists():
        raise FileNotFoundError(f"fullfit run dir missing: {fullfit_run_dir}")
    production_root.mkdir(parents=True, exist_ok=True)
    copied_files: list[str] = []
    skipped_files: list[str] = []
    for name in ANCHOR_COPY_FILES:
        source = fullfit_run_dir / name
        target = production_root / name
        if not source.exists():
            skipped_files.append(name)
            continue
        if source.suffix.lower() == ".json":
            payload = normalize_workspace_paths(read_json(source), project_root=DAILY_RESEARCH_ROOT)
            if name == "metrics.json":
                payload["production_anchor_source_run_dir"] = str(fullfit_run_dir.resolve())
                payload["production_anchor_synced_at"] = _now_text()
                payload["data_source"] = "lake"
                payload["lake_dataset_id"] = str(active_manifest.get("lake_dataset_id", "") or active_manifest.get("source_market_dataset_id", "") or "")
                payload["data_lake_root"] = str(active_manifest.get("data_lake_root", "") or (DAILY_RESEARCH_ROOT / "output" / "research_data_lake"))
            write_json_atomic(target, payload)
        else:
            shutil.copy2(source, target)
        copied_files.append(name)
    sync_manifest_path = production_root / "production_anchor_sync_manifest.json"
    fullfit_metrics = normalize_workspace_paths(read_json(fullfit_run_dir / "metrics.json"), project_root=DAILY_RESEARCH_ROOT)
    production_manifest = _build_production_manifest(
        production_root=production_root,
        fullfit_run_dir=fullfit_run_dir,
        active_manifest=active_manifest,
        fullfit_metrics=fullfit_metrics,
        anchor_sync_manifest_path=sync_manifest_path,
    )
    write_json_atomic(production_root / "production_retrain_manifest.json", production_manifest)
    touched_fields: list[str] = []
    if update_active_manifest:
        updates = {
            "production_root": str(production_root.resolve()),
            "trade_plan_refresh_run_dir": str(production_root.resolve()),
            "production_manifest_json": str((production_root / "production_retrain_manifest.json").resolve()),
            "trade_plan_target_weight_panel_csv": str((production_root / "execution_aligned_daily_live_target_weight_panel.csv").resolve()),
            "trade_plan_score_panel_csv": str((production_root / "execution_aligned_daily_live_score_panel.csv").resolve()),
            "research_candidate_target_weight_panel_csv": str((production_root / "daily_live_target_weight_panel.csv").resolve()),
            "research_candidate_score_panel_csv": str((production_root / "daily_live_score_panel.csv").resolve()),
            "production_anchor_source_run_dir": str(fullfit_run_dir.resolve()),
            "production_anchor_synced_at": _now_text(),
            "production_anchor_sync_manifest": str(sync_manifest_path.resolve()),
        }
        for key, value in updates.items():
            if active_manifest.get(key) != value:
                active_manifest[key] = value
                touched_fields.append(key)
        write_json_atomic(active_manifest_path, active_manifest)
    after = audit_production_anchor(production_root=production_root, fullfit_run_dir=fullfit_run_dir)
    result = {
        "status": "ok" if after.get("status") == "ok" else "degraded",
        "synced_at": _now_text(),
        "production_root": str(production_root.resolve()),
        "fullfit_run_dir": str(fullfit_run_dir.resolve()),
        "before": before,
        "after": after,
        "copied_files": copied_files,
        "skipped_files": skipped_files,
        "active_manifest_touched_fields": touched_fields,
        "manifest_path": str(sync_manifest_path.resolve()),
    }
    write_json_atomic(sync_manifest_path, result)
    return result


def signal_panel_summary(active_manifest: dict[str, Any], *, latest_completed_date: str) -> dict[str, Any]:
    target_text = str(active_manifest.get("trade_plan_target_weight_panel_csv", "") or "").strip()
    score_text = str(active_manifest.get("trade_plan_score_panel_csv", "") or "").strip()
    target_path = Path(target_text) if target_text else Path("__missing_trade_plan_target_weight_panel__")
    score_path = Path(score_text) if score_text else Path("__missing_trade_plan_score_panel__")
    required = _date_text(latest_completed_date)
    target_latest = panel_latest_date(target_path)
    score_latest = panel_latest_date(score_path)
    dates = [item for item in (target_latest, score_latest) if item]
    latest = min(dates) if len(dates) == 2 else (dates[0] if dates else "")
    covers = bool(required and target_latest and score_latest and pd.Timestamp(target_latest) >= pd.Timestamp(required) and pd.Timestamp(score_latest) >= pd.Timestamp(required))
    if not target_path.exists() or not score_path.exists():
        status = "missing"
    elif covers:
        status = "ok"
    else:
        status = "stale"
    return {
        "status": status,
        "required_date": required,
        "target_panel": str(target_path),
        "score_panel": str(score_path),
        "target_panel_exists": target_path.exists(),
        "score_panel_exists": score_path.exists(),
        "target_panel_latest_date": target_latest,
        "score_panel_latest_date": score_latest,
        "latest_date": latest,
        "target_panel_rows": panel_row_count(target_path),
        "score_panel_rows": panel_row_count(score_path),
        "target_position_count": _positive_target_count(target_path, latest_date=target_latest),
        "next_signal_action": "skip" if status == "ok" else "refresh",
    }


def refresh_production_live_panels(
    *,
    active_manifest_path: Path | str = ACTIVE_MANIFEST_PATH,
    as_of_date: str = "",
    sync_anchor: bool = True,
) -> dict[str, Any]:
    from daily_research.baseline.data_provider import get_latest_completed_trading_date
    from daily_research.deep_alpha.export_live_panels_from_run import refresh_live_panels_for_run

    active_manifest_path = Path(active_manifest_path)
    active_manifest = read_json(active_manifest_path)
    if not active_manifest:
        raise FileNotFoundError(f"active execution manifest missing: {active_manifest_path}")
    production_root = Path(str(active_manifest.get("production_root", "") or DEFAULT_PRODUCTION_ROOT)).resolve()
    anchor_result: dict[str, Any] = {}
    if sync_anchor:
        anchor_result = sync_production_anchor_to_fullfit(
            active_manifest_path=active_manifest_path,
            production_root=production_root,
            fullfit_run_dir=DEFAULT_FULLFIT_RUN_DIR,
            update_active_manifest=True,
        )
        active_manifest = read_json(active_manifest_path)
    required = _date_text(as_of_date or get_latest_completed_trading_date())
    lake_dataset_id = str(active_manifest.get("lake_dataset_id", "") or active_manifest.get("source_market_dataset_id", "") or "")
    if not lake_dataset_id:
        raise ValueError("active manifest has no lake_dataset_id/source_market_dataset_id for signal refresh")
    data_lake_root = str(active_manifest.get("data_lake_root", "") or (DAILY_RESEARCH_ROOT / "output" / "research_data_lake"))
    sector_board_view_id = str(active_manifest.get("sector_board_view_id", "") or "").strip()
    if not sector_board_view_id:
        sector_board_view_id = _ensure_sector_board_view_for_dataset(
            data_lake_root=data_lake_root,
            lake_dataset_id=lake_dataset_id,
        )
    export_summary = refresh_live_panels_for_run(
        production_root,
        latest_end_date=required.replace("-", ""),
        data_source="lake",
        lake_dataset_id=lake_dataset_id,
        data_lake_root=data_lake_root,
        production_manifest=active_manifest,
        sector_board_view_id=sector_board_view_id,
    )
    panel_status = signal_panel_summary(active_manifest, latest_completed_date=required)
    status = "ok" if panel_status.get("status") == "ok" else "failed"
    manifest_path = production_root / "live_panel_refresh_manifest.json"
    payload = {
        **export_summary,
        "status": status,
        "refreshed_at": _now_text(),
        "requested_as_of_date": required,
        "lake_dataset_id": lake_dataset_id,
        "data_lake_root": data_lake_root,
        "sector_board_view_id": sector_board_view_id,
        "production_root": str(production_root),
        "source_model_run": str(DEFAULT_FULLFIT_RUN_DIR.resolve()),
        "anchor_sync": anchor_result,
        "panel_status": panel_status,
    }
    write_json_atomic(manifest_path, payload)
    if status != "ok":
        raise RuntimeError(
            "production live signal refresh did not cover required date "
            f"{required}: target={panel_status.get('target_panel_latest_date', '')} "
            f"score={panel_status.get('score_panel_latest_date', '')}"
        )
    return {
        "status": status,
        "business_status": "ok",
        "artifact_status": "ok",
        "runner_status": "ok",
        "signal_refresh_manifest_path": str(manifest_path.resolve()),
        "signal_panel_latest_date": str(panel_status.get("latest_date", "")),
        "panel_latest_date": str(panel_status.get("latest_date", "")),
        "target_position_count": int(panel_status.get("target_position_count", 0) or 0),
        "artifact_paths": {
            "live_panel_refresh_manifest": str(manifest_path.resolve()),
            "target_panel": str(panel_status.get("target_panel", "")),
            "score_panel": str(panel_status.get("score_panel", "")),
            "production_anchor_sync_manifest": str(anchor_result.get("manifest_path", "")),
        },
        "evidence_paths": {
            "live_panel_refresh_manifest": str(manifest_path.resolve()),
            "target_panel": str(panel_status.get("target_panel", "")),
            "score_panel": str(panel_status.get("score_panel", "")),
            "production_anchor_sync_manifest": str(anchor_result.get("manifest_path", "")),
        },
        "signal_refresh": payload,
    }
