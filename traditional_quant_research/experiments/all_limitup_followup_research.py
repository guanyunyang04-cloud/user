"""Follow-up diagnostics for the all-limit-up ML event strategy."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.experiments.all_limitup_ml_strategy_research import (
    DEFAULT_BIG_LOSS_THRESHOLD_PCT,
    DEFAULT_EVAL_YEARS,
    DEFAULT_FEE_BPS,
    DEFAULT_MAX_TRAIN_YEARS,
    DEFAULT_MIN_TRAIN_YEARS,
    DEFAULT_MODEL_PARAMS,
    DEFAULT_PROFILE,
    DEFAULT_RISK_PENALTY_PCT,
    DEFAULT_TARGET_WINDOW,
    ID_COLUMNS,
    build_pool_masks,
    build_walk_forward_plan,
    build_walk_forward_predictions,
    encode_feature_frame,
    find_latest_event_panel,
    infer_exit_dates,
    read_event_panel,
    schedule_top_positions,
    summarize_portfolio,
    summarize_portfolio_yearly,
)
from traditional_quant_research.experiments.short_open_known_factor_rebuild import (
    audit_buy_feature_columns,
    feature_columns_for_profile,
)
from traditional_quant_research.experiments.two_day_kama_atr_breakout_analysis import (
    DEFAULT_MIN_AVAILABLE_MEMORY_GB,
    _assert_memory_available,
    available_memory_gb,
)


DEFAULT_ML_RUN_ROOT = Path("traditional_quant_research/output/experiments/all_limitup_ml_strategy_research")
DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/all_limitup_followup_research")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/brain/references/research_log/2026-06-07_all_limitup_followup_research.md")
LABEL_WINDOWS = (1, 3, 5, 10, 20)
PRACTICAL_POOLS = ("executable_only", "not_near_limit_open", "all_open_known")
PRACTICAL_STRATEGY_ID = "executable_only__lgbm_risk_rank__pos1"
DEFAULT_ABLATION_MODEL_PARAMS: dict[str, Any] = {
    **DEFAULT_MODEL_PARAMS,
    "n_estimators": 96,
    "learning_rate": 0.045,
    "n_jobs": 2,
}
STRESS_SCENARIOS: tuple[dict[str, Any], ...] = (
    {"scenario": "reported_30bps"},
    {"scenario": "strict_fill", "block_non_executable": True, "block_near_limit": True},
    {"scenario": "fee_100bps", "extra_fee_bps": 70.0},
    {
        "scenario": "gap_haircut",
        "block_near_limit": True,
        "gap_haircuts": ((7.0, 100.0), (5.0, 50.0)),
    },
    {
        "scenario": "liquidity_haircut",
        "block_near_limit": True,
        "low_liquidity_haircut_bps": 75.0,
        "liquidity_quantile": 0.20,
    },
    {
        "scenario": "tail_x1_5",
        "tail_quantile": 0.10,
        "tail_loss_multiplier": 1.50,
    },
    {
        "scenario": "combined_harsh",
        "block_non_executable": True,
        "block_near_limit": True,
        "extra_fee_bps": 70.0,
        "gap_haircuts": ((7.0, 100.0), (5.0, 50.0)),
        "low_liquidity_haircut_bps": 75.0,
        "liquidity_quantile": 0.20,
        "tail_quantile": 0.10,
        "tail_loss_multiplier": 1.25,
    },
)


def run_all_limitup_followup_research(
    *,
    ml_run_dir: str | Path | None = None,
    event_file: str | Path | None = None,
    profile: str = DEFAULT_PROFILE,
    target_window: int = DEFAULT_TARGET_WINDOW,
    fee_bps: float = DEFAULT_FEE_BPS,
    eval_years: Sequence[int] = DEFAULT_EVAL_YEARS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
    min_available_memory_gb: float = DEFAULT_MIN_AVAILABLE_MEMORY_GB,
    run_feature_ablation_models: bool = True,
    max_train_years: int = DEFAULT_MAX_TRAIN_YEARS,
    min_train_years: int = DEFAULT_MIN_TRAIN_YEARS,
    big_loss_threshold_pct: float = DEFAULT_BIG_LOSS_THRESHOLD_PCT,
    risk_penalty_pct: float = DEFAULT_RISK_PENALTY_PCT,
    ablation_model_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run low-memory follow-up diagnostics from an existing all-limit-up ML run."""

    run_id = f"all_limitup_followup_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    source_ml_run_dir = Path(ml_run_dir) if ml_run_dir is not None else latest_run_dir(DEFAULT_ML_RUN_ROOT)
    source_summary = read_source_summary(source_ml_run_dir)
    source_event_file = Path(event_file) if event_file is not None else source_event_path(source_summary)
    if not source_event_file.exists():
        source_event_file = find_latest_event_panel()

    feature_columns = tuple(feature_columns_for_profile(profile))
    audit_buy_feature_columns(feature_columns)
    _assert_memory_available(min_available_memory_gb, context="before reading follow-up inputs")
    predictions = read_predictions(source_ml_run_dir / "ml_predictions.csv")
    event_panel = read_followup_event_panel(
        source_event_file,
        feature_columns=feature_columns,
        label_windows=LABEL_WINDOWS,
        fee_bps=fee_bps,
    )
    enriched = enrich_predictions(predictions, event_panel, feature_columns=feature_columns)
    _assert_memory_available(min_available_memory_gb, context="after reading follow-up inputs")

    event_type_summary = summarize_event_facets(enriched)
    label_path_summary = summarize_label_paths(enriched, label_windows=LABEL_WINDOWS)
    stress_summary, stress_yearly = run_stress_grid(enriched)
    failure_trades, failure_summary = build_failure_attribution(enriched)
    feature_ablation_note = "skipped"
    if run_feature_ablation_models:
        try:
            feature_ablation = run_feature_family_ablation(
                event_panel,
                feature_columns=feature_columns,
                eval_years=eval_years,
                max_train_years=max_train_years,
                min_train_years=min_train_years,
                big_loss_threshold_pct=big_loss_threshold_pct,
                risk_penalty_pct=risk_penalty_pct,
                model_params={**DEFAULT_ABLATION_MODEL_PARAMS, **dict(ablation_model_params or {})},
                min_available_memory_gb=min_available_memory_gb,
            )
            feature_ablation_note = "light_model_retrain"
        except RuntimeError as exc:
            feature_ablation = build_feature_family_proxy_ablation(
                source_ml_run_dir,
                enriched,
                feature_columns=feature_columns,
            )
            feature_ablation_note = f"importance_proxy_after_runtime_error: {exc}"
    else:
        feature_ablation = build_feature_family_proxy_ablation(
            source_ml_run_dir,
            enriched,
            feature_columns=feature_columns,
        )
        feature_ablation_note = "importance_proxy_requested"
    summary = build_summary(
        run_id=run_id,
        run_dir=run_dir,
        source_ml_run_dir=source_ml_run_dir,
        source_event_file=source_event_file,
        source_summary=source_summary,
        event_panel=event_panel,
        predictions=predictions,
        enriched=enriched,
        stress_summary=stress_summary,
        feature_ablation=feature_ablation,
        failure_summary=failure_summary,
        profile=profile,
        fee_bps=fee_bps,
        target_window=target_window,
        run_feature_ablation_models=run_feature_ablation_models,
        feature_ablation_note=feature_ablation_note,
        min_available_memory_gb=min_available_memory_gb,
    )
    markdown = render_markdown(
        summary,
        event_type_summary=event_type_summary,
        label_path_summary=label_path_summary,
        stress_summary=stress_summary,
        stress_yearly=stress_yearly,
        feature_ablation=feature_ablation,
        failure_summary=failure_summary,
        failure_trades=failure_trades,
    )

    event_type_summary.to_csv(run_dir / "event_type_summary.csv", index=False, encoding="utf-8-sig")
    label_path_summary.to_csv(run_dir / "label_path_summary.csv", index=False, encoding="utf-8-sig")
    stress_summary.to_csv(run_dir / "strategy_stress_summary.csv", index=False, encoding="utf-8-sig")
    stress_yearly.to_csv(run_dir / "strategy_stress_yearly.csv", index=False, encoding="utf-8-sig")
    failure_summary.to_csv(run_dir / "failure_reason_summary.csv", index=False, encoding="utf-8-sig")
    failure_trades.to_csv(run_dir / "failure_trades.csv", index=False, encoding="utf-8-sig")
    feature_ablation.to_csv(run_dir / "feature_family_ablation.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    log_path: Path | None = None
    if write_research_log:
        log_path = Path(research_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(markdown, encoding="utf-8")

    del event_panel, enriched
    gc.collect()
    _assert_memory_available(min_available_memory_gb, context="after writing follow-up outputs")
    return {**summary, "run_dir": str(run_dir), "research_log": str(log_path) if log_path else None}


def latest_run_dir(root: str | Path) -> Path:
    candidates = sorted(
        [path for path in Path(root).glob("*") if path.is_dir() and (path / "summary.json").exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no all-limit-up ML run with summary.json found under {root}")
    return candidates[0]


def read_source_summary(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def source_event_path(summary: Mapping[str, Any]) -> Path:
    raw = str(summary.get("source_event_file") or "")
    return Path(raw) if raw else find_latest_event_panel()


def read_predictions(path: str | Path) -> pd.DataFrame:
    required = [
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
        "eval_year",
        "exit_date",
        "target_net_ret_pct",
        "predicted_ret_pct",
        "predicted_big_loss_prob",
        "ml_score",
    ]
    frame = pd.read_csv(path, usecols=lambda column: column in required, low_memory=False)
    for column in ("date", "entry_date", "exit_date"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column])
    return reduce_memory(frame)


def read_followup_event_panel(
    event_file: str | Path,
    *,
    feature_columns: Sequence[str],
    label_windows: Sequence[int],
    fee_bps: float,
) -> pd.DataFrame:
    path = Path(event_file)
    if not path.exists():
        raise FileNotFoundError(path)
    labels = label_columns(label_windows)
    required = set(ID_COLUMNS).union(feature_columns).union(labels).union({"next_open_gap_pct", "next_gap_bucket"})
    header = pd.read_csv(path, nrows=0)
    missing = sorted(required.difference(header.columns))
    if missing:
        raise ValueError(f"event panel missing follow-up columns: {missing}")
    frame = pd.read_csv(path, usecols=sorted(required), low_memory=False)
    for column in ("date", "entry_date"):
        frame[column] = pd.to_datetime(frame[column])
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["exit_date"] = infer_exit_dates(frame)
    derived: dict[str, pd.Series] = {}
    for window in label_windows:
        close_col = f"sell{int(window)}_close_ret_pct"
        derived[f"sell{int(window)}_close_net_ret_pct"] = pd.to_numeric(frame[close_col], errors="coerce") - float(fee_bps) / 100.0
        high_col = f"sell{int(window)}_max_high_pct"
        low_col = f"sell{int(window)}_min_low_pct"
        if high_col in frame.columns:
            derived[f"sell{int(window)}_mfe_net_pct"] = pd.to_numeric(frame[high_col], errors="coerce") - float(fee_bps) / 100.0
        if low_col in frame.columns:
            derived[f"sell{int(window)}_mae_net_pct"] = pd.to_numeric(frame[low_col], errors="coerce") - float(fee_bps) / 100.0
    derived["target_net_ret_pct"] = derived[f"sell{DEFAULT_TARGET_WINDOW}_close_net_ret_pct"]
    frame = pd.concat([frame, pd.DataFrame(derived, index=frame.index)], axis=1)
    return reduce_memory(frame.sort_values(["date", "code"]).reset_index(drop=True))


def label_columns(label_windows: Sequence[int]) -> set[str]:
    columns: set[str] = set()
    for window in label_windows:
        columns.update(
            {
                f"sell{int(window)}_close_ret_pct",
                f"sell{int(window)}_max_high_pct",
                f"sell{int(window)}_min_low_pct",
            }
        )
    return columns


def enrich_predictions(predictions: pd.DataFrame, event_panel: pd.DataFrame, *, feature_columns: Sequence[str]) -> pd.DataFrame:
    keys = ["date", "code"]
    extra_columns = [
        column
        for column in event_panel.columns
        if column in set(keys).union(feature_columns).union(label_columns(LABEL_WINDOWS))
        or column.endswith("_net_ret_pct")
        or column.endswith("_mfe_net_pct")
        or column.endswith("_mae_net_pct")
        or column in {"next_open_gap_pct", "next_gap_bucket", "signal_amount_log10", "market_breadth_5d", "market_limitup_rate"}
    ]
    extra = event_panel.loc[:, sorted(set(extra_columns))].copy()
    merged = predictions.merge(extra, on=keys, how="left", suffixes=("", "_event"))
    if "target_net_ret_pct_event" in merged.columns:
        merged = merged.drop(columns=["target_net_ret_pct_event"])
    add_selection_flags(merged)
    return reduce_memory(merged)


def add_selection_flags(frame: pd.DataFrame) -> None:
    frame["selected_executable_pos1"] = False
    frame["selected_executable_pos2"] = False
    masks = build_pool_masks(frame)
    for max_positions in (1, 2):
        selected = schedule_top_positions(
            frame.loc[masks["executable_only"]].copy(),
            pool_name="executable_only",
            max_positions=max_positions,
        )
        if selected.empty:
            continue
        selected_keys = set(zip(pd.to_datetime(selected["date"]), selected["code"].astype(str)))
        flag = frame.apply(lambda row: (pd.Timestamp(row["date"]), str(row["code"])) in selected_keys, axis=1)
        frame[f"selected_executable_pos{max_positions}"] = flag.to_numpy(dtype=bool)


def summarize_event_facets(frame: pd.DataFrame) -> pd.DataFrame:
    facets = assign_event_facets(frame)
    rows: list[dict[str, Any]] = []
    for facet in facets.columns:
        labels = facets[facet].reset_index(drop=True)
        base = frame.reset_index(drop=True)
        for level, positions in labels.groupby(labels, dropna=False).groups.items():
            subset = base.loc[list(positions)].copy()
            returns = pd.to_numeric(subset["target_net_ret_pct"], errors="coerce").dropna()
            rows.append(
                {
                    "facet": facet,
                    "level": str(level),
                    "event_count": int(len(subset)),
                    "mean_net_ret_pct": _mean(returns),
                    "median_net_ret_pct": _median(returns),
                    "win_rate": _rate(returns > 0),
                    "big_loss_rate": _rate(returns <= -5.0),
                    "near_limit_open_rate": _rate(_truthy_col(subset, "entry_open_near_limit")),
                    "executable_rate": _rate(_truthy_col(subset, "executable_entry")),
                    "selected_top1_rate": _rate(_truthy_col(subset, "selected_executable_pos1")),
                    "mean_ml_score": _mean(pd.to_numeric(subset.get("ml_score"), errors="coerce")),
                }
            )
    output = pd.DataFrame(rows)
    return output.sort_values(["facet", "mean_net_ret_pct", "event_count"], ascending=[True, False, False]).reset_index(drop=True)


def assign_event_facets(frame: pd.DataFrame) -> pd.DataFrame:
    index = frame.index
    facets = pd.DataFrame(index=index)
    run = pd.to_numeric(frame.get("limit_up_run_ending_today"), errors="coerce")
    facets["board_run_bucket"] = pd.cut(
        run,
        bins=[-np.inf, 1, 2, 3, np.inf],
        labels=["run_1", "run_2", "run_3", "run_4plus"],
    ).astype("object").fillna("unknown")
    facets["board_stage"] = frame.get("board_stage", pd.Series("unknown", index=index)).fillna("unknown").astype(str)
    one_word = _truthy_col(frame, "one_word_limit_like")
    near_one_word = _truthy_col(frame, "near_one_word_limit_like")
    facets["one_word_type"] = np.select(
        [one_word, near_one_word],
        ["one_word", "near_one_word"],
        default="regular_board",
    )
    kama = _truthy_col(frame, "open_below_kama_break_limitup")
    atr = _truthy_col(frame, "close_cross_atr_upper")
    facets["kama_atr_type"] = np.select(
        [kama & atr, kama & ~atr, ~kama & atr],
        ["kama_atr", "kama_only", "atr_only"],
        default="neither",
    )
    if "next_gap_bucket" in frame.columns:
        facets["next_gap_bucket"] = frame["next_gap_bucket"].fillna("unknown").astype(str)
    else:
        gap = pd.to_numeric(frame.get("next_open_gap_pct"), errors="coerce")
        facets["next_gap_bucket"] = pd.cut(
            gap,
            bins=[-np.inf, -3, 0, 3, 6, 9, np.inf],
            labels=["gap_le_m3", "gap_m3_0", "gap_0_3", "gap_3_6", "gap_6_9", "near_limit"],
        ).astype("object").fillna("unknown")
    facets["market_heat_bucket"] = quantile_bucket(frame.get("market_limitup_rate"), labels=("market_cold", "market_mid", "market_hot"))
    facets["market_breadth_bucket"] = quantile_bucket(frame.get("market_breadth_5d"), labels=("breadth_weak", "breadth_mid", "breadth_strong"))
    facets["amount_bucket"] = quantile_bucket(frame.get("signal_amount_log10"), labels=("amount_low", "amount_mid", "amount_high"))
    facets["volatility_bucket"] = quantile_bucket(frame.get("volatility_20_pct"), labels=("vol_low", "vol_mid", "vol_high"))
    return facets


def quantile_bucket(values: pd.Series | None, *, labels: Sequence[str]) -> pd.Series:
    if values is None:
        return pd.Series(["unknown"] * 0)
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series("unknown", index=numeric.index, dtype="object")
    clean = numeric.dropna()
    if clean.empty:
        return output
    ranks = clean.rank(method="first", pct=True)
    bins = np.linspace(0, 1, len(labels) + 1)
    for pos, label in enumerate(labels):
        lower = bins[pos]
        upper = bins[pos + 1]
        mask = ranks.gt(lower) & ranks.le(upper)
        if pos == 0:
            mask = ranks.ge(lower) & ranks.le(upper)
        output.loc[mask.index[mask]] = label
    return output


def summarize_label_paths(frame: pd.DataFrame, *, label_windows: Sequence[int]) -> pd.DataFrame:
    masks = build_named_masks(frame)
    rows: list[dict[str, Any]] = []
    for group_name, mask in masks.items():
        subset = frame.loc[mask].copy()
        for window in label_windows:
            ret = pd.to_numeric(subset.get(f"sell{int(window)}_close_net_ret_pct"), errors="coerce").dropna()
            mfe = pd.to_numeric(subset.get(f"sell{int(window)}_mfe_net_pct"), errors="coerce").dropna()
            mae = pd.to_numeric(subset.get(f"sell{int(window)}_mae_net_pct"), errors="coerce").dropna()
            rows.append(
                {
                    "group_name": group_name,
                    "horizon": int(window),
                    "event_count": int(len(subset)),
                    "mean_close_net_ret_pct": _mean(ret),
                    "median_close_net_ret_pct": _median(ret),
                    "win_rate": _rate(ret > 0),
                    "p10_close_net_ret_pct": _quantile(ret, 0.10),
                    "p90_close_net_ret_pct": _quantile(ret, 0.90),
                    "big_loss_rate": _rate(ret <= -5.0),
                    "big_win_rate": _rate(ret >= 5.0),
                    "mean_mfe_net_pct": _mean(mfe),
                    "mean_mae_net_pct": _mean(mae),
                    "mae_le_m5_rate": _rate(mae <= -5.0),
                }
            )
    return pd.DataFrame(rows).sort_values(["group_name", "horizon"]).reset_index(drop=True)


def build_named_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    masks = build_pool_masks(frame)
    named = {name: mask for name, mask in masks.items()}
    named["selected_executable_pos1"] = _truthy_col(frame, "selected_executable_pos1")
    named["selected_executable_pos2"] = _truthy_col(frame, "selected_executable_pos2")
    named["failed_selected_pos1"] = named["selected_executable_pos1"] & pd.to_numeric(frame["target_net_ret_pct"], errors="coerce").le(-5.0)
    return named


def run_stress_grid(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    masks = build_pool_masks(frame)
    summary_rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []
    for scenario in STRESS_SCENARIOS:
        for pool_name in PRACTICAL_POOLS:
            base = frame.loc[masks[pool_name]].copy()
            stressed, stress_meta = apply_stress_adjustments(base, scenario)
            for max_positions in (1, 2):
                selected = schedule_top_positions(stressed, pool_name=pool_name, max_positions=max_positions)
                summary = summarize_portfolio(selected, pool_name=pool_name, max_positions=max_positions)
                strategy_id = f"{pool_name}__{scenario['scenario']}__pos{int(max_positions)}"
                summary.update(stress_meta)
                summary["strategy_id"] = strategy_id
                summary["scenario"] = scenario["scenario"]
                summary_rows.append(summary)
                if not selected.empty:
                    yearly = summarize_portfolio_yearly(selected, strategy_id=strategy_id)
                    yearly["scenario"] = scenario["scenario"]
                    yearly["pool_name"] = pool_name
                    yearly["max_positions"] = int(max_positions)
                    yearly_frames.append(yearly)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["pool_name", "max_positions", "mean_period_net_ret_pct"],
        ascending=[True, True, False],
        na_position="last",
    )
    yearly = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    return summary.reset_index(drop=True), yearly


def apply_stress_adjustments(frame: pd.DataFrame, scenario: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = frame.copy()
    candidate_count = int(len(work))
    tradable = pd.Series(True, index=work.index)
    if scenario.get("block_non_executable"):
        tradable &= _truthy_col(work, "executable_entry")
    if scenario.get("block_near_limit"):
        tradable &= ~_truthy_col(work, "entry_open_near_limit")
    adjusted = pd.to_numeric(work["target_net_ret_pct"], errors="coerce").astype(float)
    adjusted -= float(scenario.get("extra_fee_bps", 0.0)) / 100.0
    gap = pd.to_numeric(work.get("next_open_gap_pct"), errors="coerce")
    total_haircut = pd.Series(0.0, index=work.index)
    for threshold, bps in scenario.get("gap_haircuts", ()):
        total_haircut = total_haircut.mask(gap.ge(float(threshold)) & total_haircut.eq(0.0), float(bps) / 100.0)
    if scenario.get("low_liquidity_haircut_bps"):
        amount = pd.to_numeric(work.get("signal_amount_log10"), errors="coerce")
        cutoff = amount.quantile(float(scenario.get("liquidity_quantile", 0.20))) if amount.notna().any() else np.nan
        low_liquidity = amount.le(cutoff) if pd.notna(cutoff) else pd.Series(False, index=work.index)
        if "liquid_amount_ok" in work.columns:
            low_liquidity |= ~_truthy_col(work, "liquid_amount_ok")
        total_haircut += low_liquidity.astype(float) * (float(scenario["low_liquidity_haircut_bps"]) / 100.0)
    adjusted -= total_haircut
    if scenario.get("tail_loss_multiplier"):
        cutoff = adjusted.quantile(float(scenario.get("tail_quantile", 0.10)))
        tail = adjusted.le(cutoff) & adjusted.lt(0)
        adjusted = adjusted.mask(tail, adjusted * float(scenario["tail_loss_multiplier"]))
    work["stress_net_ret_pct"] = adjusted
    work["target_net_ret_pct"] = adjusted
    work = work.loc[tradable].copy()
    return work, {
        "scenario": str(scenario.get("scenario", "")),
        "candidate_count_before_stress": candidate_count,
        "candidate_count_after_stress": int(len(work)),
        "blocked_by_stress_count": int(candidate_count - len(work)),
        "extra_fee_bps": float(scenario.get("extra_fee_bps", 0.0)),
    }


def build_failure_attribution(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = schedule_top_positions(
        frame.loc[build_pool_masks(frame)["executable_only"]].copy(),
        pool_name="executable_only",
        max_positions=1,
    )
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()
    failures = selected.loc[pd.to_numeric(selected["target_net_ret_pct"], errors="coerce").le(-5.0)].copy()
    if failures.empty:
        return failures, pd.DataFrame()
    failures = attribute_failure_reasons(failures, reference=frame)
    rows: list[dict[str, Any]] = []
    reason_cols = [column for column in failures.columns if column.startswith("reason_")]
    for reason_col in reason_cols:
        subset = failures.loc[_truthy_col(failures, reason_col)].copy()
        if subset.empty:
            continue
        returns = pd.to_numeric(subset["target_net_ret_pct"], errors="coerce").dropna()
        rows.append(
            {
                "reason": reason_col.replace("reason_", ""),
                "failed_trade_count": int(len(subset)),
                "share_of_failures": float(len(subset) / len(failures)),
                "mean_loss_pct": _mean(returns),
                "median_loss_pct": _median(returns),
                "mean_predicted_ret_pct": _mean(pd.to_numeric(subset.get("predicted_ret_pct"), errors="coerce")),
                "mean_big_loss_prob": _mean(pd.to_numeric(subset.get("predicted_big_loss_prob"), errors="coerce")),
                "years": ",".join(str(year) for year in sorted(pd.to_datetime(subset["entry_date"]).dt.year.unique())),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["failed_trade_count", "mean_loss_pct"], ascending=[False, True]).reset_index(drop=True)
    failures["reason_list"] = failures[reason_cols].apply(
        lambda row: "|".join(column.replace("reason_", "") for column, active in row.items() if bool(active)),
        axis=1,
    )
    return failures.sort_values("target_net_ret_pct").reset_index(drop=True), summary


def attribute_failure_reasons(failures: pd.DataFrame, *, reference: pd.DataFrame) -> pd.DataFrame:
    output = failures.copy()
    q = reference_quantiles(reference)
    output["reason_weak_market"] = pd.to_numeric(output.get("market_breadth_5d"), errors="coerce").le(q.get("market_breadth_5d_q25", np.nan)) | pd.to_numeric(
        output.get("market_limitup_rate"), errors="coerce"
    ).le(q.get("market_limitup_rate_q25", np.nan))
    output["reason_high_open_gap"] = pd.to_numeric(output.get("next_open_gap_pct"), errors="coerce").ge(5.0)
    amount = pd.to_numeric(output.get("signal_amount_log10"), errors="coerce")
    output["reason_low_liquidity"] = amount.le(q.get("signal_amount_log10_q25", np.nan))
    if "liquid_amount_ok" in output.columns:
        output["reason_low_liquidity"] |= ~_truthy_col(output, "liquid_amount_ok")
    output["reason_high_volatility"] = pd.to_numeric(output.get("volatility_20_pct"), errors="coerce").ge(q.get("volatility_20_pct_q75", np.nan))
    output["reason_overextended"] = pd.to_numeric(output.get("signal_ret20_before_pct"), errors="coerce").ge(q.get("signal_ret20_before_pct_q75", np.nan)) | pd.to_numeric(
        output.get("price_position_60d"), errors="coerce"
    ).ge(q.get("price_position_60d_q75", np.nan))
    output["reason_weak_industry"] = pd.to_numeric(output.get("industry_ret5_mean"), errors="coerce").le(q.get("industry_ret5_mean_q25", np.nan)) | pd.to_numeric(
        output.get("industry_limitup_rate"), errors="coerce"
    ).le(q.get("industry_limitup_rate_q25", np.nan))
    output["reason_advanced_board"] = pd.to_numeric(output.get("limit_up_run_ending_today"), errors="coerce").ge(3.0)
    output["reason_no_kama_atr_confirmation"] = ~(_truthy_col(output, "open_below_kama_break_limitup") & _truthy_col(output, "close_cross_atr_upper"))
    output["reason_model_overconfidence"] = pd.to_numeric(output.get("predicted_ret_pct"), errors="coerce").ge(q.get("predicted_ret_pct_q75", np.nan))
    reason_cols = [column for column in output.columns if column.startswith("reason_")]
    output["reason_unclassified"] = ~output[reason_cols].any(axis=1)
    return output


def reference_quantiles(reference: pd.DataFrame) -> dict[str, float]:
    quantiles: dict[str, float] = {}
    for column in (
        "market_breadth_5d",
        "market_limitup_rate",
        "signal_amount_log10",
        "volatility_20_pct",
        "signal_ret20_before_pct",
        "price_position_60d",
        "industry_ret5_mean",
        "industry_limitup_rate",
        "predicted_ret_pct",
    ):
        values = pd.to_numeric(reference.get(column), errors="coerce")
        quantiles[f"{column}_q25"] = float(values.quantile(0.25)) if values.notna().any() else np.nan
        quantiles[f"{column}_q75"] = float(values.quantile(0.75)) if values.notna().any() else np.nan
    return quantiles


def run_feature_family_ablation(
    events: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    eval_years: Sequence[int],
    max_train_years: int,
    min_train_years: int,
    big_loss_threshold_pct: float,
    risk_penalty_pct: float,
    model_params: Mapping[str, Any],
    min_available_memory_gb: float,
) -> pd.DataFrame:
    families = feature_family_specs(feature_columns)
    specs = {
        "market_industry_only": families["market_industry"],
        "open_print_only": families["open_print"],
        "signal_liquidity_only": families["signal_liquidity"],
        "trend_risk_only": families["trend_risk"],
        "no_open_print": tuple(column for column in feature_columns if column not in set(families["open_print"])),
    }
    plan = build_walk_forward_plan(
        events,
        eval_years=eval_years,
        max_train_years=max_train_years,
        min_train_years=min_train_years,
    )
    rows: list[dict[str, Any]] = []
    for model_name, columns in specs.items():
        columns = tuple(column for column in columns if column in events.columns)
        if not columns:
            continue
        _assert_memory_available(min_available_memory_gb, context=f"before feature ablation {model_name}")
        encoded = encode_feature_frame(events, columns)
        predictions, _, audit = build_walk_forward_predictions(
            events,
            encoded,
            plan,
            target_col="target_net_ret_pct",
            big_loss_threshold_pct=big_loss_threshold_pct,
            risk_penalty_pct=risk_penalty_pct,
            model_params=model_params,
        )
        del encoded
        gc.collect()
        masks = build_pool_masks(predictions)
        for pool_name in ("executable_only", "not_near_limit_open"):
            for max_positions in (1, 2):
                selected = schedule_top_positions(predictions.loc[masks[pool_name]].copy(), pool_name=pool_name, max_positions=max_positions)
                summary = summarize_portfolio(selected, pool_name=pool_name, max_positions=max_positions)
                rows.append(
                    {
                        "model_name": model_name,
                        "feature_count": int(len(columns)),
                        "ready_eval_year_count": int(audit.loc[audit["model_status"].eq("ready"), "eval_year"].nunique()) if not audit.empty else 0,
                        **summary,
                    }
                )
        del predictions
        gc.collect()
    return pd.DataFrame(rows).sort_values(
        ["pool_name", "max_positions", "mean_period_net_ret_pct"],
        ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)


def build_feature_family_proxy_ablation(
    ml_run_dir: str | Path,
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Build a lightweight feature-family diagnostic from saved importance and rank IC."""

    families = feature_family_specs(feature_columns)
    importance_path = Path(ml_run_dir) / "ml_feature_importance.csv"
    importance = pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame(columns=["feature", "mean_importance"])
    if "importance" in importance.columns and "mean_importance" not in importance.columns:
        importance = (
            importance.groupby("feature", as_index=False)
            .agg(mean_importance=("importance", "mean"), eval_year_count=("eval_year", "nunique"))
        )
    total_importance = float(pd.to_numeric(importance.get("mean_importance"), errors="coerce").fillna(0.0).sum()) if not importance.empty else 0.0
    target = pd.to_numeric(frame.get("target_net_ret_pct"), errors="coerce")
    rows: list[dict[str, Any]] = []
    for family_name, columns in families.items():
        family_importance = importance.loc[importance["feature"].isin(columns)].copy() if not importance.empty else pd.DataFrame()
        family_total = float(pd.to_numeric(family_importance.get("mean_importance"), errors="coerce").fillna(0.0).sum()) if not family_importance.empty else 0.0
        correlations: list[float] = []
        for column in columns:
            if column not in frame.columns:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().sum() < 30 or values.nunique(dropna=True) < 2:
                continue
            corr = rank_pearson_corr(values, target)
            if pd.notna(corr):
                correlations.append(float(abs(corr)))
        strongest = ""
        if not family_importance.empty:
            strongest = str(
                family_importance.sort_values("mean_importance", ascending=False)["feature"].iloc[0]
            )
        rows.append(
            {
                "model_name": f"{family_name}_importance_proxy",
                "pool_name": "importance_proxy",
                "max_positions": 0,
                "feature_count": int(len(columns)),
                "total_importance": family_total,
                "importance_share": family_total / total_importance if total_importance > 0 else np.nan,
                "mean_abs_spearman_to_target": float(np.mean(correlations)) if correlations else np.nan,
                "strongest_feature": strongest,
                "trade_count": np.nan,
                "mean_period_net_ret_pct": np.nan,
                "median_period_net_ret_pct": np.nan,
                "period_win_rate": np.nan,
                "positive_year_rate": np.nan,
                "min_year_period_ret_pct": np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["importance_share", "mean_abs_spearman_to_target"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def feature_family_specs(feature_columns: Sequence[str]) -> dict[str, tuple[str, ...]]:
    families: dict[str, list[str]] = {
        "market_industry": [],
        "open_print": [],
        "signal_liquidity": [],
        "trend_risk": [],
    }
    open_names = {
        "next_open_gap_pct",
        "next_gap_bucket",
        "entry_open_near_limit",
        "entry_open_normal",
        "entry_open_above_kama",
        "entry_open_above_ma5",
        "entry_open_above_ma10",
        "entry_open_low_flat_small_high",
    }
    liquidity_names = {
        "turn",
        "turn_x60",
        "price",
        "price_low_bucket",
        "price_high_bucket",
        "liquid_amount_ok",
        "signal_amount_log10",
        "signal_amount_x20",
        "signal_turn_x20",
    }
    for column in feature_columns:
        if column.startswith("market_") or column.startswith("industry_"):
            families["market_industry"].append(column)
        elif column in open_names or column.startswith("entry_open") or column.startswith("next_"):
            families["open_print"].append(column)
        elif column.startswith("signal_") or column in liquidity_names:
            families["signal_liquidity"].append(column)
        else:
            families["trend_risk"].append(column)
    return {name: tuple(columns) for name, columns in families.items()}


def rank_pearson_corr(left: pd.Series, right: pd.Series) -> float:
    pair = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"), "right": pd.to_numeric(right, errors="coerce")}).dropna()
    if len(pair) < 2:
        return np.nan
    if pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return np.nan
    return float(pair["left"].rank(method="average").corr(pair["right"].rank(method="average")))


def build_summary(
    *,
    run_id: str,
    run_dir: Path,
    source_ml_run_dir: Path,
    source_event_file: Path,
    source_summary: Mapping[str, Any],
    event_panel: pd.DataFrame,
    predictions: pd.DataFrame,
    enriched: pd.DataFrame,
    stress_summary: pd.DataFrame,
    feature_ablation: pd.DataFrame,
    failure_summary: pd.DataFrame,
    profile: str,
    fee_bps: float,
    target_window: int,
    run_feature_ablation_models: bool,
    feature_ablation_note: str,
    min_available_memory_gb: float,
) -> dict[str, Any]:
    practical_stress = stress_summary.loc[
        stress_summary["pool_name"].eq("executable_only") & stress_summary["max_positions"].eq(1)
    ].copy()
    harsh = practical_stress.loc[practical_stress["scenario"].eq("combined_harsh")].iloc[0].to_dict() if not practical_stress.loc[practical_stress["scenario"].eq("combined_harsh")].empty else {}
    reported = practical_stress.loc[practical_stress["scenario"].eq("reported_30bps")].iloc[0].to_dict() if not practical_stress.loc[practical_stress["scenario"].eq("reported_30bps")].empty else {}
    best_ablation = feature_ablation.iloc[0].to_dict() if not feature_ablation.empty else {}
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "source_ml_run_dir": str(source_ml_run_dir),
        "source_event_file": str(source_event_file),
        "source_ml_run_id": source_summary.get("run_id"),
        "profile": profile,
        "fee_bps": float(fee_bps),
        "target_window": int(target_window),
        "event_count": int(len(event_panel)),
        "prediction_count": int(len(predictions)),
        "enriched_prediction_count": int(len(enriched)),
        "run_feature_ablation_models": bool(run_feature_ablation_models),
        "feature_ablation_note": feature_ablation_note,
        "reported_practical_top1": reported,
        "combined_harsh_practical_top1": harsh,
        "best_feature_family_ablation": best_ablation,
        "top_failure_reason": failure_summary.iloc[0].to_dict() if not failure_summary.empty else {},
        "decision": "all_limitup_followup_diagnostic",
        "strategy_candidate_count": 0,
        "min_available_memory_gb": float(min_available_memory_gb),
        "available_memory_gb_at_summary": available_memory_gb(),
    }


def render_markdown(
    summary: Mapping[str, Any],
    *,
    event_type_summary: pd.DataFrame,
    label_path_summary: pd.DataFrame,
    stress_summary: pd.DataFrame,
    stress_yearly: pd.DataFrame,
    feature_ablation: pd.DataFrame,
    failure_summary: pd.DataFrame,
    failure_trades: pd.DataFrame,
) -> str:
    reported = summary.get("reported_practical_top1", {}) or {}
    harsh = summary.get("combined_harsh_practical_top1", {}) or {}
    practical_stress = stress_summary.loc[
        stress_summary["pool_name"].eq("executable_only") & stress_summary["max_positions"].eq(1)
    ].copy()
    practical_yearly = stress_yearly.loc[
        stress_yearly["strategy_id"].astype(str).str.contains("executable_only__reported_30bps__pos1", regex=False)
    ].copy() if not stress_yearly.empty else pd.DataFrame()
    selected_paths = label_path_summary.loc[
        label_path_summary["group_name"].isin(["selected_executable_pos1", "selected_executable_pos2", "executable_only"])
    ].copy()
    useful_facets = event_type_summary.loc[
        event_type_summary["event_count"].ge(100)
        & event_type_summary["facet"].isin(["board_run_bucket", "one_word_type", "kama_atr_type", "next_gap_bucket", "market_heat_bucket", "amount_bucket"])
    ].copy()
    failure_columns = [
        "date",
        "entry_date",
        "code",
        "name_on_date",
        "target_net_ret_pct",
        "predicted_ret_pct",
        "predicted_big_loss_prob",
        "next_open_gap_pct",
        "market_breadth_5d",
        "signal_amount_log10",
        "reason_list",
    ]
    lines = [
        "# All-Limit-Up Follow-Up Research",
        "",
        f"- run_id: `{summary.get('run_id')}`",
        f"- source_ml_run_dir: `{summary.get('source_ml_run_dir')}`",
        f"- source_event_file: `{summary.get('source_event_file')}`",
        f"- event_count: `{summary.get('event_count')}`",
        f"- prediction_count: `{summary.get('prediction_count')}`",
        f"- reported_executable_top1_mean_pct: `{reported.get('mean_period_net_ret_pct')}`",
        f"- combined_harsh_executable_top1_mean_pct: `{harsh.get('mean_period_net_ret_pct')}`",
        f"- feature_ablation_note: `{summary.get('feature_ablation_note')}`",
        f"- decision: `{summary.get('decision')}`",
        f"- strategy_candidate_count: `{summary.get('strategy_candidate_count')}`",
        "",
        "## Label And Path Summary",
        "",
        _markdown_table(
            selected_paths,
            [
                "group_name",
                "horizon",
                "event_count",
                "mean_close_net_ret_pct",
                "median_close_net_ret_pct",
                "win_rate",
                "big_loss_rate",
                "mean_mfe_net_pct",
                "mean_mae_net_pct",
                "mae_le_m5_rate",
            ],
        ),
        "",
        "## Event Facet Lift",
        "",
        _markdown_table(
            useful_facets.head(40),
            [
                "facet",
                "level",
                "event_count",
                "mean_net_ret_pct",
                "median_net_ret_pct",
                "win_rate",
                "big_loss_rate",
                "near_limit_open_rate",
                "executable_rate",
                "selected_top1_rate",
                "mean_ml_score",
            ],
        ),
        "",
        "## Practical Top1 Stress",
        "",
        _markdown_table(
            practical_stress,
            [
                "scenario",
                "trade_count",
                "period_count",
                "mean_period_net_ret_pct",
                "median_period_net_ret_pct",
                "period_win_rate",
                "positive_year_rate",
                "min_year_period_ret_pct",
                "p10_trade_net_ret_pct",
                "p90_trade_net_ret_pct",
                "blocked_by_stress_count",
            ],
        ),
        "",
        "## Reported Practical Yearly",
        "",
        _markdown_table(practical_yearly),
        "",
        "## Feature Family Ablation",
        "",
        _markdown_table(
            feature_ablation,
            [
                "model_name",
                "pool_name",
                "max_positions",
                "feature_count",
                "importance_share",
                "mean_abs_spearman_to_target",
                "strongest_feature",
                "trade_count",
                "mean_period_net_ret_pct",
                "median_period_net_ret_pct",
                "period_win_rate",
                "positive_year_rate",
                "min_year_period_ret_pct",
            ],
        ),
        "",
        "## Failure Reasons",
        "",
        _markdown_table(failure_summary),
        "",
        "## Worst Selected Failures",
        "",
        _markdown_table(failure_trades.head(30), failure_columns),
        "",
        "## Interpretation Boundary",
        "",
        "- This is a follow-up diagnostic report, not an execution recommendation.",
        "- Five-minute bars are still needed for stronger fill, slippage, and intraday path evidence.",
        "- Feature-family ablation uses a lighter model grid than the source full ML run, so compare ranks and robustness rather than exact headline returns.",
        "- The open-known profile remains valid only after the next open print is known.",
    ]
    return "\n".join(lines) + "\n"


def reduce_memory(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_integer_dtype(output[column]):
            output[column] = pd.to_numeric(output[column], downcast="integer")
        elif pd.api.types.is_float_dtype(output[column]):
            output[column] = pd.to_numeric(output[column], downcast="float")
    return output


def _truthy_col(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def _median(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def _quantile(values: pd.Series, q: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.quantile(q)) if not values.empty else np.nan


def _rate(values: pd.Series) -> float:
    clean = pd.Series(values).dropna()
    return float(clean.mean()) if not clean.empty else np.nan


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def parse_int_values(values: Sequence[int] | str) -> tuple[int, ...]:
    if isinstance(values, str):
        return tuple(int(item.strip()) for item in values.split(",") if item.strip())
    return tuple(int(item) for item in values)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-run-dir", default=None)
    parser.add_argument("--event-file", default=None)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--target-window", type=int, default=DEFAULT_TARGET_WINDOW)
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS)
    parser.add_argument("--eval-years", default=",".join(str(year) for year in DEFAULT_EVAL_YEARS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", default=str(DEFAULT_RESEARCH_LOG))
    parser.add_argument("--min-available-memory-gb", type=float, default=DEFAULT_MIN_AVAILABLE_MEMORY_GB)
    parser.add_argument("--skip-feature-ablation-models", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_all_limitup_followup_research(
        ml_run_dir=args.ml_run_dir,
        event_file=args.event_file,
        profile=args.profile,
        target_window=args.target_window,
        fee_bps=args.fee_bps,
        eval_years=parse_int_values(args.eval_years),
        output_dir=args.output_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
        min_available_memory_gb=args.min_available_memory_gb,
        run_feature_ablation_models=not args.skip_feature_ablation_models,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
