from __future__ import annotations

from typing import Any


DEFAULT_RESEARCH_OBJECTIVE_MODE = "execution_first"
DEFAULT_EXECUTION_ALIGNMENT_MODE = "train_eval_auto"
DEFAULT_EXECUTION_ALIGNMENT_OBJECTIVE = "robust_composite"
DEFAULT_EXECUTION_ALIGNMENT_TRANSACTION_COST_BPS = 3.0
DEFAULT_EXECUTION_ALIGNMENT_SLIPPAGE_BPS = 7.0
DEFAULT_EXECUTION_ALIGNMENT_SELL_TAX_BPS = 10.0
DEFAULT_CHECKPOINT_SELECTION_OBJECTIVE = "primary_annual_return"
CHECKPOINT_SELECTION_OBJECTIVES = (
    "valid_loss",
    "primary_annual_return",
    "primary_excess_annual_return",
    "primary_excess_sharpe",
    "primary_monthly_positive_ratio",
    "primary_monthly_median_return",
    "primary_monthly_robust_score",
)

RAW_HOLDOUT_LABEL = "holdout_backtest"
EXECUTION_HOLDOUT_LABEL = "execution_aligned_holdout_backtest"
PRIMARY_HOLDOUT_LABEL = "primary_research_backtest"
RAW_MONTHLY_LABEL = "monthly_backtest_summary"
EXECUTION_MONTHLY_LABEL = "execution_aligned_monthly_backtest_summary"
PRIMARY_MONTHLY_LABEL = "primary_research_monthly_summary"
PRIMARY_MONTHLY_DIAGNOSTICS_LABEL = "primary_research_monthly_diagnostics"


def summarize_primary_monthly_objectives(primary_monthly_diagnostics: dict[str, Any] | None) -> dict[str, float]:
    diagnostics = primary_monthly_diagnostics if isinstance(primary_monthly_diagnostics, dict) else {}
    positive_ratio = float(diagnostics.get("positive_month_ratio", 0.0) or 0.0)
    median_monthly_return = float(diagnostics.get("median_monthly_return", 0.0) or 0.0)
    mean_monthly_return = float(diagnostics.get("mean_monthly_return", 0.0) or 0.0)
    worst_monthly_return = float(diagnostics.get("worst_monthly_return", 0.0) or 0.0)
    top3_positive_share = float(diagnostics.get("top3_positive_month_share", 0.0) or 0.0)
    longest_negative_streak = int(diagnostics.get("longest_negative_streak", 0) or 0)

    downside_penalty = max(-worst_monthly_return, 0.0)
    concentration_penalty = max(top3_positive_share - 0.60, 0.0)
    streak_penalty = max(longest_negative_streak - 2, 0)
    monthly_robust_score = (
        mean_monthly_return
        + median_monthly_return
        + 0.05 * (positive_ratio - 0.50)
        - 0.35 * downside_penalty
        - 0.05 * concentration_penalty
        - 0.01 * float(streak_penalty)
    )
    return {
        "primary_monthly_positive_ratio": positive_ratio,
        "primary_monthly_median_return": median_monthly_return,
        "primary_monthly_mean_return": mean_monthly_return,
        "primary_monthly_worst_return": worst_monthly_return,
        "primary_monthly_top3_positive_share": top3_positive_share,
        "primary_monthly_longest_negative_streak": float(longest_negative_streak),
        "primary_monthly_robust_score": float(monthly_robust_score),
    }


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
    if normalized in CHECKPOINT_SELECTION_OBJECTIVES:
        return normalized
    raise ValueError(
        f"Unsupported checkpoint selection objective: {objective}. "
        f"Expected one of: {', '.join(CHECKPOINT_SELECTION_OBJECTIVES)}."
    )


def resolve_checkpoint_metric_value(
    primary_backtest: dict[str, Any],
    objective: str,
    *,
    primary_monthly_diagnostics: dict[str, Any] | None = None,
) -> tuple[str, float]:
    metric_name = resolve_checkpoint_metric_name(objective)
    if metric_name == "valid_loss":
        return metric_name, float("nan")
    if metric_name.startswith("primary_monthly_"):
        monthly_objectives = summarize_primary_monthly_objectives(primary_monthly_diagnostics)
        try:
            value = float(monthly_objectives.get(metric_name, float("nan")))
        except Exception:
            value = float("nan")
        return metric_name, value
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
