"""Machine-learning ranker for the all-limit-up event space."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.experiments.short_open_known_factor_rebuild import (
    audit_buy_feature_columns,
    feature_columns_for_profile,
)
from traditional_quant_research.experiments.two_day_kama_atr_breakout_analysis import (
    DEFAULT_MIN_AVAILABLE_MEMORY_GB,
    _assert_memory_available,
    available_memory_gb,
)


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/all_limitup_ml_strategy_research")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/brain/references/research_log/2026-06-07_all_limitup_ml_strategy_research.md")
DEFAULT_EVENT_PANEL_ROOT = Path("traditional_quant_research/output/experiments/short_open_known_factor_rebuild")
DEFAULT_PROFILE = "open_print_filter"
DEFAULT_TARGET_WINDOW = 1
DEFAULT_FEE_BPS = 30.0
DEFAULT_EVAL_YEARS = tuple(range(2019, 2027))
DEFAULT_MAX_TRAIN_YEARS = 5
DEFAULT_MIN_TRAIN_YEARS = 2
DEFAULT_MODEL_N_JOBS = 4
DEFAULT_BIG_LOSS_THRESHOLD_PCT = -5.0
DEFAULT_RISK_PENALTY_PCT = 5.0
DEFAULT_MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 160,
    "learning_rate": 0.04,
    "num_leaves": 15,
    "min_child_samples": 80,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": 42,
    "n_jobs": DEFAULT_MODEL_N_JOBS,
    "verbosity": -1,
}
ID_COLUMNS = (
    "date",
    "year",
    "code",
    "name_on_date",
    "industry",
    "entry_date",
    "entry_open",
    "entry_open_near_limit",
    "executable_entry",
    "open_below_kama_break_limitup",
    "close_cross_atr_upper",
    "entry_limit_up",
    "one_word_limit_like",
    "near_one_word_limit_like",
    "limit_up_run_ending_today",
    "board_stage",
)
POOL_NAMES = (
    "all_open_known",
    "not_near_limit_open",
    "executable_only",
    "kama_break_hard",
    "kama_atr_hard",
)


def run_all_limitup_ml_strategy_research(
    *,
    event_file: str | Path | None = None,
    eval_years: Sequence[int] = DEFAULT_EVAL_YEARS,
    profile: str = DEFAULT_PROFILE,
    target_window: int = DEFAULT_TARGET_WINDOW,
    fee_bps: float = DEFAULT_FEE_BPS,
    max_train_years: int = DEFAULT_MAX_TRAIN_YEARS,
    min_train_years: int = DEFAULT_MIN_TRAIN_YEARS,
    big_loss_threshold_pct: float = DEFAULT_BIG_LOSS_THRESHOLD_PCT,
    risk_penalty_pct: float = DEFAULT_RISK_PENALTY_PCT,
    model_n_jobs: int = DEFAULT_MODEL_N_JOBS,
    min_available_memory_gb: float = DEFAULT_MIN_AVAILABLE_MEMORY_GB,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
    model_params: Mapping[str, Any] | None = None,
    regressor_class: Any | None = None,
    classifier_class: Any | None = None,
) -> dict[str, Any]:
    selected_years = _parse_int_values(eval_years, "eval_years")
    if not selected_years:
        raise ValueError("eval_years must not be empty")
    if max_train_years <= 0 or min_train_years <= 0:
        raise ValueError("train years must be positive")
    if max_train_years < min_train_years:
        raise ValueError("max_train_years must be >= min_train_years")
    if target_window <= 0:
        raise ValueError("target_window must be positive")
    if model_n_jobs == 0:
        raise ValueError("model_n_jobs must be non-zero")

    run_id = f"all_limitup_ml_strategy_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = tuple(feature_columns_for_profile(profile))
    audit_buy_feature_columns(feature_columns)
    target_col = f"sell{int(target_window)}_close_ret_pct"
    source_event_file = Path(event_file) if event_file is not None else find_latest_event_panel()
    _assert_memory_available(min_available_memory_gb, context="before reading all-limit-up event panel")
    events = read_event_panel(
        source_event_file,
        feature_columns=feature_columns,
        target_col=target_col,
        fee_bps=fee_bps,
    )
    _assert_memory_available(min_available_memory_gb, context="after reading all-limit-up event panel")
    encoded_features = encode_feature_frame(events, feature_columns)
    plan = build_walk_forward_plan(
        events,
        eval_years=selected_years,
        max_train_years=max_train_years,
        min_train_years=min_train_years,
    )
    params = {**DEFAULT_MODEL_PARAMS, **dict(model_params or {})}
    params["n_jobs"] = int(model_n_jobs)
    predictions, importance, training_audit = build_walk_forward_predictions(
        events,
        encoded_features,
        plan,
        target_col="target_net_ret_pct",
        big_loss_threshold_pct=big_loss_threshold_pct,
        risk_penalty_pct=risk_penalty_pct,
        model_params=params,
        regressor_class=regressor_class,
        classifier_class=classifier_class,
    )
    del encoded_features
    gc.collect()
    _assert_memory_available(min_available_memory_gb, context="after ML walk-forward fit")

    portfolio, yearly, selected_trades = evaluate_prediction_pools(predictions)
    pool_summary = summarize_event_pools(events)
    feature_importance = summarize_feature_importance(importance)
    best = portfolio.iloc[0].to_dict() if not portfolio.empty else {}
    best_trades = (
        selected_trades.loc[selected_trades["strategy_id"].eq(best.get("strategy_id", ""))].copy()
        if not selected_trades.empty and best
        else pd.DataFrame()
    )
    summary = build_summary(
        run_id=run_id,
        run_dir=run_dir,
        source_event_file=source_event_file,
        events=events,
        predictions=predictions,
        plan=plan,
        portfolio=portfolio,
        feature_columns=feature_columns,
        profile=profile,
        target_window=target_window,
        fee_bps=fee_bps,
        max_train_years=max_train_years,
        min_train_years=min_train_years,
        big_loss_threshold_pct=big_loss_threshold_pct,
        risk_penalty_pct=risk_penalty_pct,
        model_params=params,
        min_available_memory_gb=min_available_memory_gb,
    )
    markdown = render_markdown(summary, pool_summary, portfolio, yearly, feature_importance, best_trades)

    pool_summary.to_csv(run_dir / "event_pool_summary.csv", index=False, encoding="utf-8-sig")
    plan.to_csv(run_dir / "walk_forward_plan.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(run_dir / "ml_predictions.csv", index=False, encoding="utf-8-sig")
    training_audit.to_csv(run_dir / "ml_training_audit.csv", index=False, encoding="utf-8-sig")
    feature_importance.to_csv(run_dir / "ml_feature_importance.csv", index=False, encoding="utf-8-sig")
    portfolio.to_csv(run_dir / "portfolio_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(run_dir / "portfolio_yearly.csv", index=False, encoding="utf-8-sig")
    selected_trades.to_csv(run_dir / "selected_trades.csv", index=False, encoding="utf-8-sig")
    best_trades.to_csv(run_dir / "best_strategy_trades.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    log_path: Path | None = None
    if write_research_log:
        log_path = Path(research_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(markdown, encoding="utf-8")

    return {**summary, "run_dir": str(run_dir), "research_log": str(log_path) if log_path is not None else None}


def find_latest_event_panel(root: str | Path = DEFAULT_EVENT_PANEL_ROOT) -> Path:
    candidates = sorted(
        Path(root).glob("*/event_feature_panel.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no event_feature_panel.csv found under {root}")
    return candidates[0]


def read_event_panel(
    event_file: str | Path,
    *,
    feature_columns: Sequence[str],
    target_col: str,
    fee_bps: float,
) -> pd.DataFrame:
    path = Path(event_file)
    if not path.exists():
        raise FileNotFoundError(path)
    required = set(ID_COLUMNS).union(feature_columns).union({target_col})
    header = pd.read_csv(path, nrows=0)
    missing = sorted(required.difference(header.columns))
    if missing:
        raise ValueError(f"event panel missing columns: {missing}")
    events = pd.read_csv(path, usecols=sorted(required), low_memory=False)
    events["date"] = pd.to_datetime(events["date"])
    events["entry_date"] = pd.to_datetime(events["entry_date"])
    events["year"] = pd.to_numeric(events["year"], errors="coerce").astype("Int64")
    events["target_raw_ret_pct"] = pd.to_numeric(events[target_col], errors="coerce")
    events["target_net_ret_pct"] = events["target_raw_ret_pct"] - float(fee_bps) / 100.0
    events["exit_date"] = infer_exit_dates(events)
    events = reduce_event_panel_memory(events)
    return events.sort_values(["date", "code"]).reset_index(drop=True)


def infer_exit_dates(events: pd.DataFrame) -> pd.Series:
    if events.empty:
        return pd.Series(pd.NaT, index=events.index)
    calendar = pd.Index(sorted(pd.concat([events["date"], events["entry_date"]]).dropna().unique()))
    next_date: dict[pd.Timestamp, pd.Timestamp] = {}
    for pos, date in enumerate(calendar):
        if pos + 1 < len(calendar):
            next_date[pd.Timestamp(date)] = pd.Timestamp(calendar[pos + 1])
    fallback = pd.to_datetime(events["entry_date"]) + pd.offsets.BDay(1)
    mapped = pd.to_datetime(events["entry_date"]).map(next_date)
    return pd.to_datetime(mapped).fillna(fallback)


def reduce_event_panel_memory(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    for column in frame.columns:
        if column in {"date", "entry_date", "exit_date"}:
            continue
        if pd.api.types.is_bool_dtype(frame[column]):
            frame[column] = frame[column].astype("bool")
        elif pd.api.types.is_integer_dtype(frame[column]):
            frame[column] = pd.to_numeric(frame[column], downcast="integer")
        elif pd.api.types.is_float_dtype(frame[column]):
            frame[column] = pd.to_numeric(frame[column], downcast="float")
    return frame


def encode_feature_frame(events: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    features = events.loc[:, list(feature_columns)].copy()
    for column in features.columns:
        if pd.api.types.is_bool_dtype(features[column]):
            features[column] = features[column].astype("int8")
        elif features[column].dtype == object:
            lowered = features[column].astype(str).str.lower()
            if set(lowered.dropna().unique()).issubset({"true", "false", "nan", "none"}):
                features[column] = lowered.eq("true").astype("int8")
    categorical = [column for column in features.columns if features[column].dtype == object or str(features[column].dtype) == "category"]
    if categorical:
        features = pd.get_dummies(features, columns=categorical, dummy_na=True, dtype=np.int8)
    for column in features.columns:
        features[column] = pd.to_numeric(features[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if pd.api.types.is_float_dtype(features[column]):
            features[column] = features[column].astype("float32")
    return features


def build_walk_forward_plan(
    events: pd.DataFrame,
    *,
    eval_years: Sequence[int],
    max_train_years: int,
    min_train_years: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target = pd.to_numeric(events["target_net_ret_pct"], errors="coerce")
    for eval_year in _parse_int_values(eval_years, "eval_years"):
        train_start_year = int(eval_year) - int(max_train_years)
        train_mask = events["year"].ge(train_start_year) & events["year"].lt(int(eval_year)) & target.notna()
        eval_mask = events["year"].eq(int(eval_year)) & target.notna()
        train_years = sorted(events.loc[train_mask, "year"].dropna().astype(int).unique().tolist())
        status = "ready" if len(train_years) >= int(min_train_years) and int(train_mask.sum()) > 0 and int(eval_mask.sum()) > 0 else "skipped"
        rows.append(
            {
                "eval_year": int(eval_year),
                "train_start_year": int(train_start_year),
                "train_end_year": int(eval_year) - 1,
                "train_years": ",".join(str(year) for year in train_years),
                "train_year_count": int(len(train_years)),
                "train_row_count": int(train_mask.sum()),
                "eval_row_count": int(eval_mask.sum()),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def build_walk_forward_predictions(
    events: pd.DataFrame,
    encoded_features: pd.DataFrame,
    plan: pd.DataFrame,
    *,
    target_col: str,
    big_loss_threshold_pct: float,
    risk_penalty_pct: float,
    model_params: Mapping[str, Any],
    regressor_class: Any | None = None,
    classifier_class: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regressor = regressor_class if regressor_class is not None else _load_lgbm_regressor()
    classifier = classifier_class if classifier_class is not None else _load_lgbm_classifier()
    target = pd.to_numeric(events[target_col], errors="coerce")
    prediction_frames: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    feature_names = encoded_features.columns.tolist()
    for _, row in plan.iterrows():
        eval_year = int(row["eval_year"])
        status = str(row["status"])
        if status != "ready":
            audit_rows.append({**row.to_dict(), "model_status": status, "target_mean": np.nan, "target_std": np.nan})
            continue
        train_mask = events["year"].ge(int(row["train_start_year"])) & events["year"].le(int(row["train_end_year"])) & target.notna()
        eval_mask = events["year"].eq(eval_year) & target.notna()
        train_x = encoded_features.loc[train_mask].copy()
        eval_x = encoded_features.loc[eval_mask].copy()
        train_y = target.loc[train_mask].astype(float)
        medians = train_x.median(numeric_only=True)
        train_x = train_x.fillna(medians).fillna(0.0)
        eval_x = eval_x.fillna(medians).fillna(0.0)
        reg_model = regressor(**dict(model_params))
        reg_model.fit(train_x, train_y)
        pred_ret = np.asarray(reg_model.predict(eval_x), dtype=float)
        big_loss_prob = np.full(len(eval_x), np.nan, dtype=float)
        train_loss = train_y.le(float(big_loss_threshold_pct)).astype(int)
        if train_loss.nunique() > 1:
            clf_model = classifier(**dict(model_params))
            clf_model.fit(train_x, train_loss)
            if hasattr(clf_model, "predict_proba"):
                big_loss_prob = np.asarray(clf_model.predict_proba(eval_x)[:, 1], dtype=float)
            else:
                big_loss_prob = np.asarray(clf_model.predict(eval_x), dtype=float)
        else:
            big_loss_prob[:] = float(train_loss.mean())
        score = pred_ret - float(risk_penalty_pct) * big_loss_prob
        prediction = events.loc[eval_mask, list(ID_COLUMNS)].copy()
        prediction["eval_year"] = eval_year
        prediction["exit_date"] = events.loc[eval_mask, "exit_date"].to_numpy()
        prediction["target_net_ret_pct"] = target.loc[eval_mask].to_numpy(dtype=float)
        prediction["predicted_ret_pct"] = pred_ret
        prediction["predicted_big_loss_prob"] = big_loss_prob
        prediction["ml_score"] = score
        prediction["risk_penalty_pct"] = float(risk_penalty_pct)
        prediction["big_loss_threshold_pct"] = float(big_loss_threshold_pct)
        prediction_frames.append(prediction)
        importances = getattr(reg_model, "feature_importances_", np.zeros(len(feature_names), dtype=float))
        for feature, importance in zip(feature_names, importances):
            importance_rows.append({"eval_year": eval_year, "feature": feature, "importance": float(importance)})
        audit_rows.append(
            {
                **row.to_dict(),
                "model_status": "ready",
                "target_mean": float(train_y.mean()),
                "target_std": float(train_y.std(ddof=0)),
            }
        )
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    importance = pd.DataFrame(importance_rows)
    audit = pd.DataFrame(audit_rows)
    return predictions, importance, audit


def build_pool_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    index = frame.index
    all_mask = pd.Series(True, index=index)
    near_open = _truthy(frame.get("entry_open_near_limit", pd.Series(False, index=index)))
    executable = _truthy(frame.get("executable_entry", pd.Series(True, index=index)))
    kama_break = _truthy(frame.get("open_below_kama_break_limitup", pd.Series(False, index=index)))
    atr_break = _truthy(frame.get("close_cross_atr_upper", pd.Series(False, index=index)))
    return {
        "all_open_known": all_mask,
        "not_near_limit_open": ~near_open,
        "executable_only": executable,
        "kama_break_hard": kama_break,
        "kama_atr_hard": kama_break & atr_break,
    }


def evaluate_prediction_pools(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    masks = build_pool_masks(predictions)
    summary_rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    for pool_name in POOL_NAMES:
        mask = masks.get(pool_name, pd.Series(False, index=predictions.index))
        for max_positions in (1, 2):
            selected = schedule_top_positions(predictions.loc[mask].copy(), pool_name=pool_name, max_positions=max_positions)
            summary_rows.append(summarize_portfolio(selected, pool_name=pool_name, max_positions=max_positions))
            if not selected.empty:
                strategy_id = f"{pool_name}__lgbm_risk_rank__pos{int(max_positions)}"
                selected = selected.assign(strategy_id=strategy_id, pool_name=pool_name, max_positions=int(max_positions))
                selected_frames.append(selected)
                yearly_frames.append(summarize_portfolio_yearly(selected, strategy_id=strategy_id))
    portfolio = pd.DataFrame(summary_rows)
    portfolio = portfolio.sort_values(
        ["mean_period_net_ret_pct", "positive_year_rate", "min_year_period_ret_pct", "period_count"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    yearly = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    selected_trades = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    return portfolio, yearly, selected_trades


def schedule_top_positions(frame: pd.DataFrame, *, pool_name: str, max_positions: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    required = ["entry_date", "exit_date", "ml_score", "target_net_ret_pct"]
    clean = frame.dropna(subset=required).copy()
    if clean.empty:
        return clean
    clean["entry_date"] = pd.to_datetime(clean["entry_date"])
    clean["exit_date"] = pd.to_datetime(clean["exit_date"])
    clean = clean.sort_values(["entry_date", "ml_score"], ascending=[True, False])
    open_exit_dates: list[pd.Timestamp] = []
    selected_indices: list[Any] = []
    for entry_date, group in clean.groupby("entry_date", sort=True):
        entry_ts = pd.Timestamp(entry_date)
        open_exit_dates = [date for date in open_exit_dates if date >= entry_ts]
        slots = int(max_positions) - len(open_exit_dates)
        if slots <= 0:
            continue
        take = group.head(slots)
        selected_indices.extend(take.index.tolist())
        open_exit_dates.extend(pd.to_datetime(take["exit_date"]).tolist())
    selected = clean.loc[selected_indices].copy()
    selected["pool_name"] = pool_name
    selected["max_positions"] = int(max_positions)
    return selected.reset_index(drop=True)


def summarize_portfolio(selected: pd.DataFrame, *, pool_name: str, max_positions: int) -> dict[str, Any]:
    strategy_id = f"{pool_name}__lgbm_risk_rank__pos{int(max_positions)}"
    base = {"strategy_id": strategy_id, "pool_name": pool_name, "max_positions": int(max_positions)}
    if selected.empty:
        return {
            **base,
            "trade_count": 0,
            "period_count": 0,
            "mean_trade_net_ret_pct": np.nan,
            "median_trade_net_ret_pct": np.nan,
            "trade_win_rate": np.nan,
            "mean_period_net_ret_pct": np.nan,
            "median_period_net_ret_pct": np.nan,
            "period_win_rate": np.nan,
            "positive_year_rate": np.nan,
            "min_year_period_ret_pct": np.nan,
            "p10_trade_net_ret_pct": np.nan,
            "p90_trade_net_ret_pct": np.nan,
            "worst_trade_pct": np.nan,
            "near_limit_open_rate": np.nan,
            "executable_rate": np.nan,
        }
    returns = pd.to_numeric(selected["target_net_ret_pct"], errors="coerce").dropna()
    period_returns = selected.groupby("entry_date")["target_net_ret_pct"].mean().dropna()
    yearly = selected.groupby(pd.to_datetime(selected["entry_date"]).dt.year)["target_net_ret_pct"].mean()
    return {
        **base,
        "trade_count": int(len(selected)),
        "period_count": int(period_returns.size),
        "mean_trade_net_ret_pct": float(returns.mean()) if not returns.empty else np.nan,
        "median_trade_net_ret_pct": float(returns.median()) if not returns.empty else np.nan,
        "trade_win_rate": float((returns > 0).mean()) if not returns.empty else np.nan,
        "mean_period_net_ret_pct": float(period_returns.mean()) if not period_returns.empty else np.nan,
        "median_period_net_ret_pct": float(period_returns.median()) if not period_returns.empty else np.nan,
        "period_win_rate": float((period_returns > 0).mean()) if not period_returns.empty else np.nan,
        "positive_year_rate": float((yearly > 0).mean()) if not yearly.empty else np.nan,
        "min_year_period_ret_pct": float(yearly.min()) if not yearly.empty else np.nan,
        "p10_trade_net_ret_pct": float(returns.quantile(0.10)) if not returns.empty else np.nan,
        "p90_trade_net_ret_pct": float(returns.quantile(0.90)) if not returns.empty else np.nan,
        "worst_trade_pct": float(returns.min()) if not returns.empty else np.nan,
        "near_limit_open_rate": float(_truthy(selected.get("entry_open_near_limit", pd.Series(False, index=selected.index))).mean()),
        "executable_rate": float(_truthy(selected.get("executable_entry", pd.Series(True, index=selected.index))).mean()),
    }


def summarize_portfolio_yearly(selected: pd.DataFrame, *, strategy_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, subset in selected.groupby(pd.to_datetime(selected["entry_date"]).dt.year):
        returns = pd.to_numeric(subset["target_net_ret_pct"], errors="coerce").dropna()
        period_returns = subset.groupby("entry_date")["target_net_ret_pct"].mean().dropna()
        rows.append(
            {
                "strategy_id": strategy_id,
                "year": int(year),
                "trade_count": int(len(subset)),
                "period_count": int(period_returns.size),
                "mean_trade_net_ret_pct": float(returns.mean()) if not returns.empty else np.nan,
                "mean_period_net_ret_pct": float(period_returns.mean()) if not period_returns.empty else np.nan,
                "median_period_net_ret_pct": float(period_returns.median()) if not period_returns.empty else np.nan,
                "period_win_rate": float((period_returns > 0).mean()) if not period_returns.empty else np.nan,
                "min_trade_pct": float(returns.min()) if not returns.empty else np.nan,
                "max_trade_pct": float(returns.max()) if not returns.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_event_pools(events: pd.DataFrame) -> pd.DataFrame:
    masks = build_pool_masks(events)
    rows: list[dict[str, Any]] = []
    for name in POOL_NAMES:
        subset = events.loc[masks[name]].copy()
        returns = pd.to_numeric(subset.get("target_net_ret_pct"), errors="coerce").dropna()
        rows.append(
            {
                "pool_name": name,
                "event_count": int(len(subset)),
                "mean_net_ret_pct": float(returns.mean()) if not returns.empty else np.nan,
                "median_net_ret_pct": float(returns.median()) if not returns.empty else np.nan,
                "win_rate": float((returns > 0).mean()) if not returns.empty else np.nan,
                "near_limit_open_rate": float(_truthy(subset.get("entry_open_near_limit", pd.Series(False, index=subset.index))).mean()) if not subset.empty else np.nan,
                "executable_rate": float(_truthy(subset.get("executable_entry", pd.Series(True, index=subset.index))).mean()) if not subset.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_feature_importance(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return pd.DataFrame(columns=["feature", "mean_importance", "eval_year_count"])
    return (
        importance.groupby("feature", as_index=False)
        .agg(mean_importance=("importance", "mean"), eval_year_count=("eval_year", "nunique"))
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )


def build_summary(
    *,
    run_id: str,
    run_dir: Path,
    source_event_file: Path,
    events: pd.DataFrame,
    predictions: pd.DataFrame,
    plan: pd.DataFrame,
    portfolio: pd.DataFrame,
    feature_columns: Sequence[str],
    profile: str,
    target_window: int,
    fee_bps: float,
    max_train_years: int,
    min_train_years: int,
    big_loss_threshold_pct: float,
    risk_penalty_pct: float,
    model_params: Mapping[str, Any],
    min_available_memory_gb: float,
) -> dict[str, Any]:
    practical = (
        portfolio.loc[
            portfolio["near_limit_open_rate"].le(0.01)
            & portfolio["executable_rate"].ge(0.95)
            & portfolio["period_count"].ge(30)
        ].copy()
        if not portfolio.empty
        else pd.DataFrame()
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "source_event_file": str(source_event_file),
        "profile": profile,
        "target_window": int(target_window),
        "fee_bps": float(fee_bps),
        "event_count": int(len(events)),
        "prediction_count": int(len(predictions)),
        "eval_years": sorted(plan["eval_year"].astype(int).tolist()) if not plan.empty else [],
        "ready_eval_years": sorted(plan.loc[plan["status"].eq("ready"), "eval_year"].astype(int).tolist()) if not plan.empty else [],
        "feature_column_count": int(len(feature_columns)),
        "feature_leakage_audit": "passed",
        "model_family": "lightgbm.LGBMRegressor + LightGBM big-loss classifier",
        "max_train_years": int(max_train_years),
        "min_train_years": int(min_train_years),
        "big_loss_threshold_pct": float(big_loss_threshold_pct),
        "risk_penalty_pct": float(risk_penalty_pct),
        "model_params": dict(model_params),
        "best_strategy": portfolio.iloc[0].to_dict() if not portfolio.empty else {},
        "best_practical_strategy": practical.iloc[0].to_dict() if not practical.empty else {},
        "decision": "all_limitup_ml_diagnostic",
        "strategy_candidate_count": 0,
        "min_available_memory_gb": float(min_available_memory_gb),
        "available_memory_gb_at_summary": available_memory_gb(),
    }


def render_markdown(
    summary: dict[str, Any],
    pool_summary: pd.DataFrame,
    portfolio: pd.DataFrame,
    yearly: pd.DataFrame,
    feature_importance: pd.DataFrame,
    best_trades: pd.DataFrame,
) -> str:
    best_id = str(summary.get("best_strategy", {}).get("strategy_id", ""))
    practical_id = str(summary.get("best_practical_strategy", {}).get("strategy_id", ""))
    best_yearly = yearly.loc[yearly["strategy_id"].eq(best_id)].copy() if not yearly.empty and best_id else pd.DataFrame()
    practical = portfolio.loc[portfolio["near_limit_open_rate"].le(0.01) & portfolio["executable_rate"].ge(0.95)].copy() if not portfolio.empty else pd.DataFrame()
    practical_yearly = yearly.loc[yearly["strategy_id"].eq(practical_id)].copy() if not yearly.empty and practical_id else pd.DataFrame()
    lines = [
        "# All-Limit-Up ML Strategy Research",
        "",
        f"- run_id: `{summary.get('run_id')}`",
        f"- source_event_file: `{summary.get('source_event_file')}`",
        f"- event_count: `{summary.get('event_count')}`",
        f"- prediction_count: `{summary.get('prediction_count')}`",
        f"- profile: `{summary.get('profile')}`",
        f"- feature_column_count: `{summary.get('feature_column_count')}`",
        f"- fee_bps: `{summary.get('fee_bps')}`",
        f"- best_strategy_id: `{best_id}`",
        f"- best_practical_strategy_id: `{practical_id}`",
        f"- decision: `{summary.get('decision')}`",
        f"- strategy_candidate_count: `{summary.get('strategy_candidate_count')}`",
        "",
        "## Event Pools",
        "",
        _markdown_table(pool_summary),
        "",
        "## Portfolio Ranking",
        "",
        _markdown_table(
            portfolio.head(20),
            [
                "strategy_id",
                "trade_count",
                "period_count",
                "mean_period_net_ret_pct",
                "median_period_net_ret_pct",
                "period_win_rate",
                "positive_year_rate",
                "min_year_period_ret_pct",
                "mean_trade_net_ret_pct",
                "p10_trade_net_ret_pct",
                "p90_trade_net_ret_pct",
                "near_limit_open_rate",
                "executable_rate",
            ],
        ),
        "",
        "## Practical Ranking",
        "",
        _markdown_table(
            practical.head(12),
            [
                "strategy_id",
                "trade_count",
                "period_count",
                "mean_period_net_ret_pct",
                "median_period_net_ret_pct",
                "period_win_rate",
                "positive_year_rate",
                "min_year_period_ret_pct",
                "p10_trade_net_ret_pct",
                "p90_trade_net_ret_pct",
                "near_limit_open_rate",
                "executable_rate",
            ],
        ),
        "",
        "## Practical Best Yearly",
        "",
        _markdown_table(practical_yearly),
        "",
        "## Best Strategy Yearly",
        "",
        _markdown_table(best_yearly),
        "",
        "## Top Feature Importance",
        "",
        _markdown_table(feature_importance.head(25)),
        "",
        "## Best Trades Sample",
        "",
        _markdown_table(
            best_trades.head(30),
            [
                "date",
                "entry_date",
                "exit_date",
                "code",
                "name_on_date",
                "pool_name",
                "ml_score",
                "predicted_ret_pct",
                "predicted_big_loss_prob",
                "target_net_ret_pct",
                "entry_open_near_limit",
                "executable_entry",
                "open_below_kama_break_limitup",
                "close_cross_atr_upper",
            ],
        ),
        "",
        "## Interpretation Boundary",
        "",
        "- This is a diagnostic ML ranker over all limit-up events; it is not an execution recommendation.",
        "- KAMA/ATR appears as ordinary features and as comparison pools, not as the event boundary.",
        "- The open-known profile includes next-open fields, so it is valid only after the Day+1 open print is known.",
        "- Near-limit and one-word paths must be separately haircut before any promotion decision.",
    ]
    return "\n".join(lines) + "\n"


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _load_lgbm_regressor() -> Any:
    try:
        from lightgbm import LGBMRegressor
    except Exception as exc:  # pragma: no cover - dependency is environment-specific
        raise RuntimeError("lightgbm is required for all_limitup_ml_strategy_research") from exc
    return LGBMRegressor


def _load_lgbm_classifier() -> Any:
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:  # pragma: no cover - dependency is environment-specific
        raise RuntimeError("lightgbm is required for all_limitup_ml_strategy_research") from exc
    return LGBMClassifier


def _parse_int_values(values: Sequence[int] | str, name: str) -> tuple[int, ...]:
    if isinstance(values, str):
        raw = [item.strip() for item in values.split(",") if item.strip()]
    else:
        raw = list(values)
    parsed = tuple(int(item) for item in raw)
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    return parsed


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    if frame is None or frame.empty:
        return "_No rows._"
    output = frame.copy()
    if columns is not None:
        output = output[[column for column in columns if column in output.columns]]
    for column in output.columns:
        if pd.api.types.is_numeric_dtype(output[column]):
            output[column] = output[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.6g}")
    return output.to_markdown(index=False)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-file", default=None)
    parser.add_argument("--eval-years", default=",".join(str(year) for year in DEFAULT_EVAL_YEARS))
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--target-window", type=int, default=DEFAULT_TARGET_WINDOW)
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS)
    parser.add_argument("--max-train-years", type=int, default=DEFAULT_MAX_TRAIN_YEARS)
    parser.add_argument("--min-train-years", type=int, default=DEFAULT_MIN_TRAIN_YEARS)
    parser.add_argument("--big-loss-threshold-pct", type=float, default=DEFAULT_BIG_LOSS_THRESHOLD_PCT)
    parser.add_argument("--risk-penalty-pct", type=float, default=DEFAULT_RISK_PENALTY_PCT)
    parser.add_argument("--model-n-jobs", type=int, default=DEFAULT_MODEL_N_JOBS)
    parser.add_argument("--min-available-memory-gb", type=float, default=DEFAULT_MIN_AVAILABLE_MEMORY_GB)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", default=str(DEFAULT_RESEARCH_LOG))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_all_limitup_ml_strategy_research(
        event_file=args.event_file,
        eval_years=_parse_int_values(args.eval_years, "eval_years"),
        profile=args.profile,
        target_window=args.target_window,
        fee_bps=args.fee_bps,
        max_train_years=args.max_train_years,
        min_train_years=args.min_train_years,
        big_loss_threshold_pct=args.big_loss_threshold_pct,
        risk_penalty_pct=args.risk_penalty_pct,
        model_n_jobs=args.model_n_jobs,
        min_available_memory_gb=args.min_available_memory_gb,
        output_dir=args.output_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
