from __future__ import annotations

from typing import Any


DEFAULT_RESEARCH_OBJECTIVE_MODE = "execution_first"
DEFAULT_EXECUTION_ALIGNMENT_MODE = "train_eval_auto"
DEFAULT_EXECUTION_ALIGNMENT_OBJECTIVE = "robust_composite"
DEFAULT_EXECUTION_ALIGNMENT_TRANSACTION_COST_BPS = 3.0
DEFAULT_EXECUTION_ALIGNMENT_SLIPPAGE_BPS = 7.0
DEFAULT_EXECUTION_ALIGNMENT_SELL_TAX_BPS = 10.0
DEFAULT_CHECKPOINT_SELECTION_OBJECTIVE = "primary_annual_return"

RAW_HOLDOUT_LABEL = "holdout_backtest"
EXECUTION_HOLDOUT_LABEL = "execution_aligned_holdout_backtest"
PRIMARY_HOLDOUT_LABEL = "primary_research_backtest"
RAW_MONTHLY_LABEL = "monthly_backtest_summary"
EXECUTION_MONTHLY_LABEL = "execution_aligned_monthly_backtest_summary"
PRIMARY_MONTHLY_LABEL = "primary_research_monthly_summary"
PRIMARY_MONTHLY_DIAGNOSTICS_LABEL = "primary_research_monthly_diagnostics"


def normalize_research_objective_mode(raw: Any) -> str:
    normalized = str(raw or "").strip().lower()
    if normalized in {"", "default"}:
        return DEFAULT_RESEARCH_OBJECTIVE_MODE
    if normalized not in {"execution_first", "raw_holdout"}:
        raise ValueError(
            f"Unsupported research objective mode: {raw}. "
            "Expected one of: execution_first, raw_holdout."
        )
    return normalized


def resolve_primary_backtest_label(metrics: dict[str, Any], research_objective_mode: str = "") -> str:
    explicit = str(metrics.get("primary_research_backtest_label", "") or "").strip()
    if explicit:
        return explicit
    mode = normalize_research_objective_mode(research_objective_mode or metrics.get("research_objective_mode", ""))
    execution_metrics = metrics.get(EXECUTION_HOLDOUT_LABEL)
    if mode == "execution_first" and isinstance(execution_metrics, dict) and execution_metrics:
        return EXECUTION_HOLDOUT_LABEL
    return RAW_HOLDOUT_LABEL


def resolve_primary_backtest(metrics: dict[str, Any], research_objective_mode: str = "") -> tuple[str, dict[str, Any]]:
    explicit_payload = metrics.get(PRIMARY_HOLDOUT_LABEL)
    explicit_label = str(metrics.get("primary_research_backtest_label", "") or "").strip()
    if explicit_label and isinstance(explicit_payload, dict) and explicit_payload:
        return explicit_label, dict(explicit_payload)

    label = resolve_primary_backtest_label(metrics, research_objective_mode=research_objective_mode)
    payload = metrics.get(label)
    if isinstance(payload, dict):
        return label, dict(payload)
    return label, {}


def resolve_primary_panel_mode(metrics: dict[str, Any], research_objective_mode: str = "") -> str:
    label = resolve_primary_backtest_label(metrics, research_objective_mode=research_objective_mode)
    execution_profile = str(metrics.get("execution_alignment_profile", "") or "").strip()
    if label == EXECUTION_HOLDOUT_LABEL and execution_profile:
        return "execution_aligned"
    return "raw"


def resolve_primary_monthly_summary_label(metrics: dict[str, Any], research_objective_mode: str = "") -> str:
    label = resolve_primary_backtest_label(metrics, research_objective_mode=research_objective_mode)
    if label == EXECUTION_HOLDOUT_LABEL:
        return EXECUTION_MONTHLY_LABEL
    return RAW_MONTHLY_LABEL


def resolve_primary_monthly_diagnostics_label(metrics: dict[str, Any], research_objective_mode: str = "") -> str:
    label = resolve_primary_monthly_summary_label(metrics, research_objective_mode=research_objective_mode)
    if label == EXECUTION_MONTHLY_LABEL:
        return "execution_aligned_monthly_backtest_diagnostics"
    return "monthly_backtest_diagnostics"


def resolve_checkpoint_metric_name(objective: str) -> str:
    normalized = str(objective or "").strip().lower()
    if normalized == "valid_loss":
        return "valid_loss"
    if normalized == "primary_annual_return":
        return "primary_annual_return"
    if normalized == "primary_excess_annual_return":
        return "primary_excess_annual_return"
    if normalized == "primary_excess_sharpe":
        return "primary_excess_sharpe"
    raise ValueError(
        f"Unsupported checkpoint selection objective: {objective}. "
        "Expected one of: valid_loss, primary_annual_return, primary_excess_annual_return, primary_excess_sharpe."
    )


def resolve_checkpoint_metric_value(primary_backtest: dict[str, Any], objective: str) -> tuple[str, float]:
    metric_name = resolve_checkpoint_metric_name(objective)
    if metric_name == "valid_loss":
        return metric_name, float("nan")
    key = {
        "primary_annual_return": "annual_return",
        "primary_excess_annual_return": "excess_annual_return",
        "primary_excess_sharpe": "excess_sharpe",
    }[metric_name]
    try:
        value = float(primary_backtest.get(key, float("nan")))
    except Exception:
        value = float("nan")
    return metric_name, value
