from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.baseline.data_provider import get_latest_completed_trading_date
from daily_research.execution.update_default_candidate_production import (
    DEFAULT_ACTIVE_EXECUTION_STRATEGY_MANIFEST,
    _activate_strategy,
    _sync_production_root,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
UPDATE_PRODUCTION_SCRIPT = PROJECT_ROOT / "daily_research" / "execution" / "update_default_candidate_production.py"
EXECUTION_POLICY_AUDIT_SCRIPT = PROJECT_ROOT / "daily_research" / "deep_alpha" / "run_execution_policy_audit.py"
EXECUTION_H2H_SCRIPT = PROJECT_ROOT / "daily_research" / "tools" / "execution_candidate_multiwindow_h2h.py"
RUN_TRADE_PLAN_SCRIPT = PROJECT_ROOT / "daily_research" / "execution" / "run_trade_plan.py"
DEFAULT_PRODUCTION_ROOT = OUTPUT_ROOT / "deep_alpha_short_alpha_execalign_production_default"
DEFAULT_PRODUCTION_MANIFEST = DEFAULT_PRODUCTION_ROOT / "production_retrain_manifest.json"
DEFAULT_EXECUTION_POLICY_PROFILE = "regoff_k1_5d_ensemble_native_anchor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fresh short_alpha production full-fit retrain under the current profit-max execution policy, "
            "replay it against the current production root, and promote it only if the fresh production root wins."
        )
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--production-manifest", default=str(DEFAULT_PRODUCTION_MANIFEST))
    parser.add_argument("--production-root", default=str(DEFAULT_PRODUCTION_ROOT))
    parser.add_argument("--strategy-manifest-path", default=str(DEFAULT_ACTIVE_EXECUTION_STRATEGY_MANIFEST))
    parser.add_argument("--source-run-dir", default="")
    parser.add_argument("--execution-policy-profile", default=DEFAULT_EXECUTION_POLICY_PROFILE)
    parser.add_argument("--end-date", default="")
    parser.add_argument("--root-tag", default="short_alpha_profitmax_production_refresh_20260405_r1")
    parser.add_argument("--activate-best", dest="activate_best", action="store_true")
    parser.add_argument("--no-activate-best", dest="activate_best", action="store_false")
    parser.set_defaults(activate_best=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))


def _resolve_source_run_dir(args: argparse.Namespace, production_manifest: dict[str, Any]) -> Path:
    explicit = str(args.source_run_dir or "").strip()
    if explicit:
        return Path(explicit).resolve()
    raw = str(production_manifest.get("source_formal_run_dir", "") or "").strip()
    if not raw:
        raise ValueError("source_formal_run_dir is missing from the production manifest.")
    return Path(raw).resolve()


def _resolve_bridge_start(strategy_manifest: dict[str, Any], source_run_dir: Path) -> str:
    explicit = str(strategy_manifest.get("backtest_start_date", "") or "").strip()
    if explicit and len(explicit) == 8 and explicit.isdigit():
        return pd.Timestamp(explicit).strftime("%Y-%m-%d")
    tail = source_run_dir.name.rsplit("_", 2)
    if len(tail) >= 3 and tail[-2].isdigit():
        return pd.Timestamp(tail[-2]).strftime("%Y-%m-%d")
    return "2025-03-18"


def _resolve_single_profile_run(audit_root: Path, execution_policy_profile: str) -> Path:
    profiles_root = audit_root / "profiles"
    runs = [path for path in sorted(profiles_root.iterdir()) if path.is_dir()]
    if not runs:
        raise RuntimeError(f"No replay runs were found under {profiles_root}")
    requested = str(execution_policy_profile).strip()
    matching = [path for path in runs if path.name.startswith(f"{requested}_")]
    if len(matching) == 1:
        return matching[0].resolve()
    if len(runs) == 1:
        return runs[0].resolve()
    raise RuntimeError(
        f"Expected a unique replay run for {requested} under {profiles_root}, found {len(matching)} matches and {len(runs)} total runs."
    )


def _run_policy_audit(
    *,
    python_executable: str,
    run_dir: Path,
    execution_policy_profile: str,
    experiment_tag: str,
) -> tuple[Path, Path]:
    command = [
        str(python_executable),
        str(EXECUTION_POLICY_AUDIT_SCRIPT),
        "--run-dir",
        str(run_dir),
        "--panel-scope",
        "live",
        "--profile-set",
        str(execution_policy_profile),
        "--selection-objective",
        "excess_annual_return",
        "--experiment-tag",
        experiment_tag,
    ]
    _run_command(command)
    audit_root = (OUTPUT_ROOT / experiment_tag).resolve()
    return audit_root, _resolve_single_profile_run(audit_root, execution_policy_profile)


def _run_h2h(
    *,
    python_executable: str,
    review_run_dir: Path,
    current_run_dir: Path,
    bridge_start: str,
    output_dir: Path,
) -> Path:
    command = [
        str(python_executable),
        str(EXECUTION_H2H_SCRIPT),
        "--run-a",
        str(review_run_dir),
        "--label-a",
        "review_production",
        "--run-b",
        str(current_run_dir),
        "--label-b",
        "current_production",
        "--bridge-start",
        str(bridge_start),
        "--output-dir",
        str(output_dir),
    ]
    _run_command(command)
    return output_dir / "head2head_summary.csv"


def _evaluate_h2h(head2head_csv: Path) -> dict[str, Any]:
    frame = pd.read_csv(head2head_csv)
    if frame.empty:
        raise RuntimeError(f"head2head summary is empty: {head2head_csv}")
    full_row = frame.loc[frame["window_label"] == "full_available"]
    if full_row.empty:
        raise RuntimeError("full_available row is missing from head2head summary.")
    full_row = full_row.iloc[0]
    window_type_column = next(
        (column for column in frame.columns if column.startswith("window_type_")),
        "",
    )
    if not window_type_column:
        raise RuntimeError("window_type column is missing from head2head summary.")
    named = frame.loc[frame[window_type_column] == "named_window"].copy()
    annual_wins = int((named["winner_excess_annual_return"] == "review_production").sum())
    annual_losses = int((named["winner_excess_annual_return"] == "current_production").sum())
    sharpe_wins = int((named["winner_excess_sharpe"] == "review_production").sum())
    sharpe_losses = int((named["winner_excess_sharpe"] == "current_production").sum())
    should_promote = bool(
        float(full_row["excess_annual_return_delta"]) > 0.0
        and float(full_row["excess_sharpe_delta"]) > 0.0
        and annual_wins >= annual_losses
        and sharpe_wins >= sharpe_losses
    )
    return {
        "full_excess_annual_return_delta": float(full_row["excess_annual_return_delta"]),
        "full_excess_sharpe_delta": float(full_row["excess_sharpe_delta"]),
        "named_window_count": int(len(named)),
        "named_window_annual_wins": annual_wins,
        "named_window_annual_losses": annual_losses,
        "named_window_sharpe_wins": sharpe_wins,
        "named_window_sharpe_losses": sharpe_losses,
        "should_promote": should_promote,
    }


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except Exception:
        return "n/a"


def _num(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "n/a"


def _promote_review_root(
    *,
    review_manifest: dict[str, Any],
    production_root: Path,
    strategy_manifest_path: Path,
    current_strategy_manifest: dict[str, Any],
    activate_best: bool,
) -> tuple[Path | None, dict[str, Any]]:
    review_run_dir = Path(review_manifest["active_production_run_dir"]).resolve()
    source_run_dir = Path(review_manifest["source_formal_run_dir"]).resolve()
    _sync_production_root(
        run_dir=review_run_dir,
        production_root=production_root,
        source_run_dir=source_run_dir,
        latest_completed_date=str(review_manifest.get("launch_cutoff_date", "")),
        latest_trainable_date=str(review_manifest.get("train_end_date", "")),
        internal_monitor_start_date=str(review_manifest.get("internal_monitor_start_date", "")),
        internal_monitor_days=int(review_manifest.get("internal_monitor_days", 3) or 3),
        train_start_date=str(review_manifest.get("train_start_date", "20210101")),
        strategy_manifest_path=strategy_manifest_path,
        activate_strategy=bool(activate_best),
    )
    if not activate_best:
        return None, {}
    strategy_name = str(current_strategy_manifest.get("strategy_name", "") or "").strip()
    strategy_panel_mode = str(current_strategy_manifest.get("panel_mode", "raw") or "raw").strip().lower()
    manifest_path, payload = _activate_strategy(
        source_run_dir=source_run_dir,
        production_root=production_root,
        strategy_manifest_path=strategy_manifest_path,
        strategy_name=strategy_name,
        strategy_panel_mode=strategy_panel_mode,
    )
    return manifest_path, payload


def _write_summary(
    *,
    output_dir: Path,
    source_run_dir: Path,
    current_audit_root: Path,
    review_audit_root: Path,
    h2h_output_dir: Path,
    evaluation: dict[str, Any],
    review_manifest: dict[str, Any],
    promoted: bool,
    active_manifest_path: Path | None,
    active_payload: dict[str, Any],
) -> None:
    summary_payload = {
        "source_run_dir": str(source_run_dir),
        "current_audit_root": str(current_audit_root),
        "review_audit_root": str(review_audit_root),
        "h2h_output_dir": str(h2h_output_dir),
        "review_manifest": review_manifest,
        "evaluation": evaluation,
        "promoted": bool(promoted),
        "active_execution_strategy_manifest": "" if active_manifest_path is None else str(active_manifest_path),
        "active_execution_candidate_label": str(active_payload.get("candidate_label", "") or ""),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Short Alpha Profit-Max Production Refresh",
        "",
        f"- source_run_dir: `{source_run_dir}`",
        f"- review_run_dir: `{review_manifest.get('active_production_run_dir', '')}`",
        f"- full-period excess annual delta: `{_pct(evaluation.get('full_excess_annual_return_delta'))}`",
        f"- full-period excess Sharpe delta: `{_num(evaluation.get('full_excess_sharpe_delta'))}`",
        f"- named-window annual wins/losses: `{evaluation.get('named_window_annual_wins', 0)}/{evaluation.get('named_window_annual_losses', 0)}`",
        f"- named-window Sharpe wins/losses: `{evaluation.get('named_window_sharpe_wins', 0)}/{evaluation.get('named_window_sharpe_losses', 0)}`",
        f"- promoted: `{bool(promoted)}`",
    ]
    if active_manifest_path is not None:
        lines.extend(
            [
                "",
                "## Activated Strategy",
                f"- active_execution_strategy_manifest: `{active_manifest_path}`",
                f"- candidate_label: `{active_payload.get('candidate_label', '')}`",
                f"- execution_policy_label: `{active_payload.get('execution_policy_label', '')}`",
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    production_manifest = _load_json(Path(args.production_manifest).resolve())
    if not production_manifest:
        raise FileNotFoundError(f"Invalid production manifest: {args.production_manifest}")
    current_strategy_manifest = _load_json(Path(args.strategy_manifest_path).resolve())
    source_run_dir = _resolve_source_run_dir(args, production_manifest)
    latest_completed_date = pd.Timestamp(
        args.end_date or production_manifest.get("launch_cutoff_date") or get_latest_completed_trading_date()
    ).strftime("%Y%m%d")

    output_dir = OUTPUT_ROOT / str(args.root_tag).strip()
    output_dir.mkdir(parents=True, exist_ok=True)
    review_root = output_dir / "review_production_root"
    review_strategy_manifest = output_dir / "review_active_execution_strategy.json"

    review_manifest_path = review_root / "production_retrain_manifest.json"
    if not review_manifest_path.exists():
        refresh_cmd = [
            str(args.python_executable),
            str(UPDATE_PRODUCTION_SCRIPT),
            "--source-run-dir",
            str(source_run_dir),
            "--production-root",
            str(review_root),
            "--strategy-manifest-path",
            str(review_strategy_manifest),
            "--experiment-tag",
            f"{args.root_tag}/runs/review_profitmax_fullfit",
            "--end-date",
            latest_completed_date,
            "--research-objective-mode",
            "execution_first",
            "--execution-alignment-mode",
            "profile",
            "--execution-alignment-profile",
            str(args.execution_policy_profile),
            "--no-activate-strategy",
        ]
        _run_command(refresh_cmd)

    review_manifest = _load_json(review_manifest_path)
    if not review_manifest:
        raise FileNotFoundError(f"Review production manifest was not created under {review_root}")

    current_audit_root, current_replay_run = _run_policy_audit(
        python_executable=str(args.python_executable),
        run_dir=Path(args.production_root).resolve(),
        execution_policy_profile=str(args.execution_policy_profile),
        experiment_tag=f"{args.root_tag}/current_policy_audit",
    )
    review_audit_root, review_replay_run = _run_policy_audit(
        python_executable=str(args.python_executable),
        run_dir=review_root,
        execution_policy_profile=str(args.execution_policy_profile),
        experiment_tag=f"{args.root_tag}/review_policy_audit",
    )

    bridge_start = _resolve_bridge_start(current_strategy_manifest, source_run_dir)
    h2h_output_dir = output_dir / "recent_h2h_review_vs_current"
    head2head_csv = _run_h2h(
        python_executable=str(args.python_executable),
        review_run_dir=review_replay_run,
        current_run_dir=current_replay_run,
        bridge_start=bridge_start,
        output_dir=h2h_output_dir,
    )
    evaluation = _evaluate_h2h(head2head_csv)

    promoted = False
    active_manifest_path: Path | None = None
    active_payload: dict[str, Any] = {}
    if bool(evaluation["should_promote"]):
        active_manifest_path, active_payload = _promote_review_root(
            review_manifest=review_manifest,
            production_root=Path(args.production_root).resolve(),
            strategy_manifest_path=Path(args.strategy_manifest_path).resolve(),
            current_strategy_manifest=current_strategy_manifest,
            activate_best=bool(args.activate_best),
        )
        promoted = True
        _run_command([str(args.python_executable), str(RUN_TRADE_PLAN_SCRIPT)])

    _write_summary(
        output_dir=output_dir,
        source_run_dir=source_run_dir,
        current_audit_root=current_audit_root,
        review_audit_root=review_audit_root,
        h2h_output_dir=h2h_output_dir,
        evaluation=evaluation,
        review_manifest=review_manifest,
        promoted=promoted,
        active_manifest_path=active_manifest_path,
        active_payload=active_payload,
    )
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
