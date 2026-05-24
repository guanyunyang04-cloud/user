from __future__ import annotations

import sys
from pathlib import Path

EXECUTION_DEFAULT_ENHANCED_PROFILE = "up_low_breakout_v2"
EXECUTION_DEFAULT_STATE_ENSEMBLE_WEIGHTS = "trend_up_low_vol=ml:0.25,none:0.25,v2:0.50"
EXECUTION_DEFAULT_LGBM_N_ESTIMATORS = "520"


def has_arg(name: str) -> bool:
    for item in sys.argv[1:]:
        if item == name or item.startswith(name + "="):
            return True
    return False


def get_arg_value(name: str) -> str | None:
    items = sys.argv[1:]
    for idx, item in enumerate(items):
        if item == name:
            if idx + 1 >= len(items):
                raise ValueError(f"Argument {name} expects a value.")
            return items[idx + 1]
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return None


def inject_default_arg(name: str, value: str) -> None:
    if not has_arg(name):
        sys.argv.extend([name, value])


def inject_flag_arg(name: str) -> None:
    if not has_arg(name):
        sys.argv.append(name)


def consume_option_arg(name: str) -> str | None:
    items = sys.argv[1:]
    rewritten = [sys.argv[0]]
    captured: str | None = None
    idx = 0
    while idx < len(items):
        item = items[idx]
        if item == name:
            if idx + 1 >= len(items):
                raise ValueError(f"Argument {name} expects a value.")
            captured = items[idx + 1]
            idx += 2
            continue
        if item.startswith(name + "="):
            captured = item.split("=", 1)[1]
            idx += 1
            continue
        rewritten.append(item)
        idx += 1
    sys.argv = rewritten
    return captured


def consume_flag_arg(name: str) -> bool:
    items = sys.argv[1:]
    rewritten = [sys.argv[0]]
    found = False
    for item in items:
        if item == name:
            found = True
            continue
        rewritten.append(item)
    sys.argv = rewritten
    return found


def is_help_request() -> bool:
    return any(item in {"-h", "--help"} for item in sys.argv[1:])


def missing_runtime_dependency_error(exc: ModuleNotFoundError, *, command_hint: str) -> SystemExit:
    missing = str(getattr(exc, "name", "") or exc).strip()
    detail = (
        f"`daily_research` 运行时缺少依赖 `{missing}`。"
        "请先执行 "
        "`conda env update -f daily_research/environment.yml --prune` "
        "同步权威环境，再使用显式的 `yolos` python 重试。"
    )
    if command_hint:
        detail = f"{detail} 出错入口：`{command_hint}`。"
    return SystemExit(detail)


def bootstrap_execution_paths(entry_file: str) -> Path:
    exec_dir = Path(entry_file).resolve().parent
    baseline_dir = exec_dir.parent / "baseline"
    project_root = exec_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(baseline_dir) not in sys.path:
        sys.path.insert(0, str(baseline_dir))
    return exec_dir


def _resolve_requested_pool_size(*, pool_name: str = "", pool_size: int = 0) -> int:
    if int(pool_size or 0) > 0:
        return int(pool_size)
    if pool_name:
        from daily_research.execution.strategy_manifest import liquidity_pool_size_from_name

        resolved = liquidity_pool_size_from_name(pool_name)
        if resolved > 0:
            return resolved
    try:
        from daily_research.execution.strategy_manifest import (
            liquidity_pool_size_from_name,
            load_strategy_manifest,
            resolve_manifest_liquidity_pool_name,
        )

        manifest = load_strategy_manifest()
        resolved = liquidity_pool_size_from_name(resolve_manifest_liquidity_pool_name(manifest))
        if resolved > 0:
            return resolved
    except Exception:
        pass
    from daily_research.execution.liquidity_universe import DEFAULT_POOL_SIZE

    return int(DEFAULT_POOL_SIZE)


def ensure_default_pool_argument(*, pool_name: str = "", pool_size: int = 0, start_date: str = "") -> None:
    if has_arg("--stocks") or has_arg("--stocks-file") or is_help_request():
        return
    from daily_research.execution.liquidity_universe import ensure_default_pool_file, get_default_pool_file
    from daily_research.baseline.data_provider import find_universe_violations, load_cached_stock_name_map

    resolved_pool_size = _resolve_requested_pool_size(pool_name=pool_name, pool_size=pool_size)
    resolved_pool_name = f"liquid{resolved_pool_size}"
    data_source = str(get_arg_value("--data-source") or "lake").strip().lower()
    lake_dataset_id = str(get_arg_value("--lake-dataset-id") or "").strip()
    data_lake_root = str(get_arg_value("--data-lake-root") or "").strip()
    resolved_start_date = str(start_date or get_arg_value("--start-date") or "20240101").strip()
    if is_help_request():
        pool_file = get_default_pool_file(resolved_pool_size)
    else:
        pool_file = ensure_default_pool_file(
            pool_size=resolved_pool_size,
            start_date=resolved_start_date,
            data_source=data_source,
            lake_dataset_id=lake_dataset_id,
            data_lake_root=data_lake_root,
            refresh_cache=False,
        )
    if not pool_file.exists() and not is_help_request():
        raise FileNotFoundError(
            f"Default {resolved_pool_name} universe file not found after preflight: {pool_file}. "
            "Please run daily_research/execution/update_liquid_pool.py after close first."
        )
    if pool_file.exists() and not is_help_request():
        violations = _pool_file_violations(pool_file)
        if violations:
            pool_file = ensure_default_pool_file(
                pool_size=resolved_pool_size,
                start_date=resolved_start_date,
                data_source=data_source,
                lake_dataset_id=lake_dataset_id,
                data_lake_root=data_lake_root,
                refresh_cache=True,
            )
            violations = _pool_file_violations(pool_file)
            if violations:
                bad_examples = ", ".join(list(violations.keys())[:8])
                raise ValueError(
                    "Default execution pool contains stocks outside the allowed trade universe "
                    f"(main-board SH/SZ A-shares only, excluding ST). Re-run update_liquid_pool.py. Examples: {bad_examples}"
                )
    inject_default_arg("--stocks-file", str(pool_file))


def ensure_external_target_weight_universe_argument(
    *,
    target_weight_csv: str = "",
    stock_column: str = "stock",
    date_column: str = "date",
    output_name: str = "external_target_weight_latest",
) -> bool:
    if has_arg("--stocks") or has_arg("--stocks-file") or is_help_request():
        return False
    path_text = str(target_weight_csv or get_arg_value("--external-target-weight-csv") or "").strip()
    if not path_text:
        return False
    path = Path(path_text).expanduser()
    if not path.exists():
        return False

    import pandas as pd

    frame = pd.read_csv(path, usecols=lambda column: str(column).strip().lower() in {date_column.lower(), stock_column.lower()})
    columns = {str(column).strip().lower(): str(column) for column in frame.columns}
    if date_column.lower() not in columns or stock_column.lower() not in columns:
        return False
    dates = pd.to_datetime(frame[columns[date_column.lower()]], errors="coerce")
    latest_date = dates.max()
    if pd.isna(latest_date):
        return False
    latest = frame.loc[dates.eq(latest_date)].copy()
    stocks = sorted(
        {
            str(stock).strip().upper()
            for stock in latest[columns[stock_column.lower()]].dropna().astype(str)
            if str(stock).strip()
        }
    )
    if not stocks:
        return False
    from daily_research.execution.liquidity_universe import get_universe_dir

    universe_dir = get_universe_dir()
    universe_dir.mkdir(parents=True, exist_ok=True)
    universe_path = universe_dir / f"{output_name}.txt"
    universe_path.write_text("\n".join(stocks) + "\n", encoding="utf-8")
    inject_default_arg("--stocks-file", str(universe_path))
    return True


def _pool_file_violations(pool_file: Path) -> dict[str, str]:
    from daily_research.baseline.data_provider import find_universe_violations, load_cached_stock_name_map

    raw_stocks = [
        line.strip().upper()
        for line in pool_file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    stock_name_map = load_cached_stock_name_map()
    return find_universe_violations(
        raw_stocks,
        stock_name_map=stock_name_map if not stock_name_map.empty else None,
    )


def ensure_execution_strategy_defaults() -> None:
    # Promote the current execution default from ma60 to the validated ma50 baseline.
    inject_default_arg("--regime-ma-window", "50")
    inject_default_arg("--enhanced-profile", EXECUTION_DEFAULT_ENHANCED_PROFILE)
    inject_default_arg("--ensemble-state-weights", EXECUTION_DEFAULT_STATE_ENSEMBLE_WEIGHTS)
    inject_default_arg("--lgbm-n-estimators", EXECUTION_DEFAULT_LGBM_N_ESTIMATORS)


def ensure_text_file_from_example(target: Path, example: Path, default_text: str) -> None:
    if target.exists():
        return
    if example.exists():
        target.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        return
    target.write_text(default_text, encoding="utf-8")
