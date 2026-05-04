from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DATE = "2026-05-02"
DEFAULT_OUTPUT_DIR = Path("daily_research/output/continuous_policy/analysis/cycle_audits")

BRAIN_EVIDENCE_FILES = (
    Path("daily_research/brain/state_center.md"),
    Path("daily_research/brain/continuous_policy_design_contract.md"),
    Path("daily_research/brain/knowledge_center.md"),
    Path("daily_research/brain/operations_center.md"),
    Path("daily_research/brain/episodic_memory.md"),
)

CODE_EVIDENCE_FILES = (
    Path("daily_research/continuous_policy/run_self_optimizing_study.py"),
    Path("daily_research/continuous_policy/portfolio_simulator.py"),
    Path("daily_research/continuous_policy/allocation_optimizer.py"),
    Path("daily_research/continuous_policy/model_seq_v3.py"),
)

STUDY_TAGS = (
    "cp_v3_portfolio_daily_allocation_breadth_r34_evidence_confirm_20260430",
    "cp_v3_portfolio_daily_unified_allocation_r35_postfix4_bounded_confirm_20260430",
    "cp_v3_portfolio_daily_risk_aware_unified_allocation_r36c_source_distribution_gate_screening64_20260501",
    "cp_v3_portfolio_daily_decision_focused_allocation_r37b_screening64_20260501",
    "self_opt_study_r38_source_hard_negative_regret_20260501",
    "self_opt_study_r39_allocation_objective_consolidation_execblend_20260502",
)

PROBLEM_TERMS = {
    "source_sell_wrong": (
        "source_positive_forward_sell_share",
        "source_strong_positive_forward_sell_count",
        "strong false sell",
        "false-sell",
        "source hard-negative",
    ),
    "receiver_not_executable": (
        "receiver_unrealized_deploy_share",
        "receiver_exec_guard",
        "semantic_no_headroom",
        "authorized_add_no_weight_change",
        "deploy_intent_unrealized",
    ),
    "cash_dead_or_overdefensive": (
        "cash_reserve_rate",
        "cash_timing_quality_1d",
        "cash dead",
        "dead branch",
        "cash-relief",
    ),
    "economic_quality_unstable": (
        "receiver_minus_source_forward_excess_5d",
        "monthly_return_mean",
        "monthly_consistency",
        "max_drawdown",
        "drawdown",
    ),
    "old_translation_path": (
        "direct_action",
        "order_translation",
        "simulator guard",
        "translation guard",
        "action_budget_split_v1",
    ),
}

MAINLINE_PHASES = (
    {
        "phase": "r1-r4",
        "theme": "alpha prior / result-value-budget split",
        "conclusion": "单票动作与组合预算混在一起，无法闭合资金从哪里来、去哪里。",
    },
    {
        "phase": "r5-r11b",
        "theme": "lifecycle arbitration / sell attribution / sell-source contract",
        "conclusion": "卖出不是单票看跌，而是组合资金来源选择；hidden funding sell 与 held-side release 反复污染旧仓。",
    },
    {
        "phase": "r12-r18",
        "theme": "release / translation / deploy / direct-action repair",
        "conclusion": "动作语义更清楚，但订单翻译与资金部署仍会改写策略意图。",
    },
    {
        "phase": "r19-r23",
        "theme": "portfolio daily ranking / cash-aware / source-receiver execution",
        "conclusion": "执行合同能被拉正，但执行干净不等于收益、回撤、月度质量稳定。",
    },
    {
        "phase": "r24-r30",
        "theme": "allocation teacher / economic release / brake-relief",
        "conclusion": "局部放宽与局部刹车反复把系统推向卖错 source 或 source dormant / cash 过高。",
    },
    {
        "phase": "r31-r34",
        "theme": "receiver semantic closure / source clean-pass / breadth",
        "conclusion": "receiver 旁路和 source 强势误卖能被压住，但 breadth、spread、收益质量仍不稳定。",
    },
    {
        "phase": "r35-r39",
        "theme": "unified allocation / risk-aware / decision-focused / objective consolidation",
        "conclusion": "方向转向正确，但代码仍接在旧 action-head + simulator-guard 框架内，形成半新半旧路径。",
    },
)

RENAMED_OLD_SOLUTIONS = (
    "把 r35-r39 命名为 unified / risk-aware / decision-focused / objective consolidation，但搜索配置仍连续使用 action_budget_split_v1、budget_v3、split_v2、v15 simulator calibration 与 portfolio_daily_ranking_v2_gated。",
    "把 source 问题从 release quality 改名为 economic release、forward proxy、hard-negative regret，本质仍是用局部惩罚调 source gate。",
    "把 cash 问题从 cash-aware 改名为 cash relief、cash deployment target、cash timing target，本质仍是在旧 score/guard 后修现金分支。",
    "把 receiver 问题从 deploy/add repair 改名为 receiver executable subset、receiver breadth，本质仍是在旧 direct action path 后修可买性。",
)

LOCAL_PATCHES_WITHOUT_PROGRESS = (
    "单边加强 source 防错会得到 positive-forward sell = 0，但系统退化为 source_target_count 过低和 cash_reserve_rate 过高。",
    "单边恢复 source 或 receiver breadth 会重新暴露 source positive distribution、receiver-source spread 或 no-headroom add。",
    "只把 v2 gate 后验指标加入 scoring，不能解决训练阶段 credit assignment；screening 与 confirm 会继续分裂。",
    "让 simulator guard 继续 OR/覆盖模型意图，会把主策略变成翻译器补漏洞，而不是学习资金分配。",
)

VALIDATED_CONCLUSIONS = (
    "continuous_policy 必须保持 research / shadow_only，未过 full gate 与 stable confirm 前不能 promotion / live。",
    "receiver_unrealized_deploy_share = 0、source_positive_forward_sell_share = 0 只是必要条件，不是成功条件。",
    "source、receiver、cash 不能作为独立局部头分别修补；它们是同一个组合日资金分配问题。",
    "r31 receiver executable subset、r33 source clean-pass、r20-r23 execution contracts 是有效安全边界，应冻结为最后防线。",
    "foundation model 只能增强状态表征，不能替代可审计的 allocation decision layer。",
)

ROUTES_TO_RETIRE = (
    "废弃 r40 = 再新增一个 source/cash/reduce 局部 penalty 的路线。",
    "废弃把 simulator guard 当主策略逻辑的路线。",
    "废弃以单项 gate 清零、短窗 smoke 高分或 insufficient evidence confirm 作为阶段成功的路线。",
    "废弃继续在 action head 主导下包装 unified allocation 名称的路线。",
    "废弃为恢复收益而粗暴放宽 source clean-pass 的路线。",
)


@dataclass(frozen=True)
class StudySnapshot:
    study_tag: str
    objective_profile: str
    search_profile: str
    budget_semantics: str
    budget_calibration: str
    phase: str
    training_evidence_status: str
    promotion_status: str
    passed_check_count: float | None
    total_check_count: float | None
    failed_checks: list[str]
    annual_return: float | None
    sharpe: float | None
    max_drawdown: float | None
    monthly_return_mean: float | None
    cash_timing_quality_1d: float | None
    cash_reserve_rate: float | None
    source_target_count: float | None
    receiver_target_count: float | None
    source_positive_forward_sell_share: float | None
    receiver_minus_source_forward_excess_5d: float | None


def _workspace_path(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else WORKSPACE_ROOT / p


def _read_text(path: Path) -> str:
    full = _workspace_path(path)
    if not full.exists():
        return ""
    return full.read_text(encoding="utf-8-sig", errors="replace")


def _read_json(path: Path) -> dict[str, Any]:
    full = _workspace_path(path)
    if not full.exists():
        return {}
    data = json.loads(full.read_text(encoding="utf-8-sig"))
    return data if isinstance(data, dict) else {}


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _term_counts() -> dict[str, int]:
    corpus = "\n".join(_read_text(path) for path in BRAIN_EVIDENCE_FILES + CODE_EVIDENCE_FILES)
    lower = corpus.lower()
    counts: dict[str, int] = {}
    for cluster, terms in PROBLEM_TERMS.items():
        counts[cluster] = sum(lower.count(term.lower()) for term in terms)
    return counts


def _profile_continuity() -> dict[str, Any]:
    study_code = _read_text(Path("daily_research/continuous_policy/run_self_optimizing_study.py"))
    profile_names = sorted(set(re.findall(r"split_heads_portfolio_daily_[a-z0-9_]+_r([0-9]+)", study_code)))
    r34_r39_block = study_code[
        study_code.find('"split_heads_portfolio_daily_allocation_breadth_r34"') :
        study_code.find("def _normalize_date_text")
    ]
    r40_start = study_code.find('"split_heads_portfolio_daily_end_to_end_allocation_layer_r40"')
    r40_block = study_code[r40_start : r40_start + 2200] if r40_start >= 0 else ""
    profile_keys = sorted(
        set(re.findall(r'"(split_heads_portfolio_daily_[^"]+_r(?:3[4-9]))"', r34_r39_block))
    )
    objective_map_count = len(re.findall(r'portfolio_daily_ranking_v2_gated', r34_r39_block))
    return {
        "profile_r_count": len(profile_names),
        "r34_r39_profiles": profile_keys,
        "r34_r39_profile_count": len(profile_keys),
        "r34_r39_v2_gated_mentions": objective_map_count,
        "r34_r39_action_budget_split_v1_mentions": r34_r39_block.count("action_budget_split_v1"),
        "r34_r39_v15_calibration_mentions": r34_r39_block.count(
            "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15"
        ),
        "r34_r39_split_v2_mentions": r34_r39_block.count("split_v2"),
        "r40_end_to_end_profile_exists": bool(r40_block),
        "r40_allocation_layer_v1_mentions": r40_block.count("allocation_layer_v1")
        + r40_block.count("BUDGET_SEMANTICS_ALLOCATION_LAYER"),
        "r40_end_to_end_calibration_mentions": r40_block.count("end_to_end_allocation_layer_v1")
        + r40_block.count("BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER"),
        "r40_action_budget_split_v1_mentions": r40_block.count("action_budget_split_v1"),
        "r40_v15_calibration_mentions": r40_block.count(
            "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15"
        ),
        "r40_exits_old_action_budget_path": bool(
            r40_block
            and (
                "end_to_end_allocation_layer_v1" in r40_block
                or "BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER" in r40_block
            )
            and (
                "allocation_layer_v1" in r40_block
                or "BUDGET_SEMANTICS_ALLOCATION_LAYER" in r40_block
            )
            and "action_budget_split_v1" not in r40_block
            and "cash_constraint_portfolio_daily_ranking_receiver_exec_guard_v15" not in r40_block
        ),
    }


def _code_path_status() -> dict[str, Any]:
    simulator = _read_text(Path("daily_research/continuous_policy/portfolio_simulator.py"))
    optimizer = _read_text(Path("daily_research/continuous_policy/allocation_optimizer.py"))
    model = _read_text(Path("daily_research/continuous_policy/model_seq_v3.py"))
    return {
        "portfolio_simulator_direct_action_mentions": simulator.count("direct_action"),
        "portfolio_simulator_guard_mentions": simulator.lower().count("guard"),
        "portfolio_simulator_unified_candidate_mentions": simulator.count("portfolio_daily_unified"),
        "simulator_still_or_joins_unified_receiver_candidate": bool(
            re.search(r"direct_action_(?:open|add)_signal[\s\S]{0,1200}portfolio_daily_unified_receiver_candidate", simulator)
        ),
        "semidifferentiable_optimizer_exists": "solve_semidifferentiable_allocation" in optimizer,
        "optimizer_is_sequential_sell_then_buy": "sell_budget" in optimizer and "buy_budget" in optimizer,
        "allocation_final_objective_in_model": "portfolio_daily_allocation_final_objective" in model,
        "action_loss_still_configured": '"action_total"' in model,
    }


def _select_trial(summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    for phase_name in ("confirmatory_trials", "screening_trials", "trials"):
        trials = summary.get(phase_name)
        if isinstance(trials, list) and trials:
            completed = [trial for trial in trials if isinstance(trial, dict) and trial.get("status") == "completed"]
            trial = completed[0] if completed else trials[0]
            return phase_name.replace("_trials", ""), trial if isinstance(trial, dict) else {}
    return "", {}


def _metric(metrics: dict[str, Any], *names: str) -> float | None:
    for name in names:
        if name in metrics:
            return _to_float(metrics.get(name))
    return None


def _study_snapshots() -> list[StudySnapshot]:
    snapshots: list[StudySnapshot] = []
    for tag in STUDY_TAGS:
        path = Path("daily_research/output/continuous_policy/studies") / tag / "study_summary.json"
        summary = _read_json(path)
        if not summary:
            continue
        phase, trial = _select_trial(summary)
        metrics = trial.get("primary_metrics") if isinstance(trial.get("primary_metrics"), dict) else {}
        if not metrics and isinstance(trial.get("metrics"), dict):
            metrics = trial["metrics"]
        config = trial.get("trial_config") if isinstance(trial.get("trial_config"), dict) else {}
        snapshots.append(
            StudySnapshot(
                study_tag=tag,
                objective_profile=str(summary.get("objective_profile", "")),
                search_profile=str(summary.get("search_profile", "")),
                budget_semantics=str(summary.get("budget_semantics", config.get("budget_semantics", ""))),
                budget_calibration=str(summary.get("budget_calibration", config.get("budget_calibration", ""))),
                phase=phase,
                training_evidence_status=str(metrics.get("training_evidence_status", "")),
                promotion_status=str(trial.get("promotion_status", "")),
                passed_check_count=_to_float(trial.get("passed_check_count")),
                total_check_count=_to_float(trial.get("total_check_count")),
                failed_checks=[str(item) for item in trial.get("failed_checks", []) if str(item)],
                annual_return=_metric(metrics, "annual_return"),
                sharpe=_metric(metrics, "sharpe"),
                max_drawdown=_metric(metrics, "max_drawdown"),
                monthly_return_mean=_metric(metrics, "monthly_return_mean"),
                cash_timing_quality_1d=_metric(metrics, "cash_timing_quality_1d"),
                cash_reserve_rate=_metric(metrics, "portfolio_daily_cash_reserve_rate", "cash_reserve_rate"),
                source_target_count=_metric(metrics, "portfolio_daily_source_target_count", "source_target_count"),
                receiver_target_count=_metric(metrics, "portfolio_daily_receiver_target_count", "receiver_target_count"),
                source_positive_forward_sell_share=_metric(
                    metrics,
                    "portfolio_daily_source_positive_forward_sell_share",
                    "source_positive_forward_sell_share",
                ),
                receiver_minus_source_forward_excess_5d=_metric(
                    metrics,
                    "portfolio_daily_receiver_minus_source_forward_excess_5d",
                    "receiver_minus_source_forward_excess_5d",
                ),
            )
        )
    return snapshots


def build_audit() -> dict[str, Any]:
    term_counts = _term_counts()
    profile_status = _profile_continuity()
    code_status = _code_path_status()
    snapshots = _study_snapshots()
    cycle_detected = (
        profile_status["r34_r39_v2_gated_mentions"] >= 6
        and profile_status["r34_r39_action_budget_split_v1_mentions"] >= 6
        and code_status["portfolio_simulator_direct_action_mentions"] > 100
        and code_status["simulator_still_or_joins_unified_receiver_candidate"]
    )
    return {
        "audit_date": AUDIT_DATE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cycle_detected": cycle_detected,
        "verdict": (
            "存在重复循环；应停止 r39 后继续追加局部 penalty/guard 的原路线。"
            if cycle_detected
            else "未发现足够证据证明存在结构性重复循环。"
        ),
        "evidence_files": [path.as_posix() for path in BRAIN_EVIDENCE_FILES + CODE_EVIDENCE_FILES],
        "term_counts": term_counts,
        "profile_continuity": profile_status,
        "code_path_status": code_status,
        "study_snapshots": [asdict(snapshot) for snapshot in snapshots],
        "mainline_phases": MAINLINE_PHASES,
        "repeated_problem_clusters": [
            "source 卖错：正 forward source 被卖、强势误卖、source opportunity cost 被低估。",
            "receiver 买不了或买不宽：no-headroom add、receiver unrealized deploy、flat open breadth 接不进主路径。",
            "cash 死分支或过度保守：cash timing 弱、cash reserve 过高、收益和 exposure 利用不足。",
            "收益/回撤/月度质量不过线：语义干净后仍可能 annual return 弱、drawdown 失控或 monthly consistency 不稳。",
            "confirm 不稳定：screening 高分不等于 fresh confirm / stable confirm 能过。",
            "旧翻译路径残留：action head、direct action signal、translation guard 与 simulator guard 仍在主决策链中。",
        ],
        "renamed_old_solutions": RENAMED_OLD_SOLUTIONS,
        "local_patches_without_progress": LOCAL_PATCHES_WITHOUT_PROGRESS,
        "validated_conclusions": VALIDATED_CONCLUSIONS,
        "routes_to_retire": ROUTES_TO_RETIRE,
        "true_bottleneck": (
            "credit assignment 尚未在同一个可优化 allocation objective / allocation layer 内闭合。"
            "当前仍是半新半旧路径：训练目标逐步加入 source/receiver/cash 统一信号，"
            "但执行和验证仍大量经过 action head、direct-action translation、v15 simulator guard 与后验 v2 gate。"
            "真正瓶颈不是再找一个局部 penalty，而是让模型和 optimizer 一次性决定今天资金从谁释放、给谁、留多少现金、承担多少成本和回撤。"
        ),
        "priority_execution_plan": [
            "P0：冻结 r20-r23、r31、r33 的安全合同，只作为最后边界；新研究不得再以局部 guard 清零作为成功。",
            "P1：建立循环审计与主线停环规则，任何 r40+ 新 profile 若仍沿用 action_budget_split_v1 + v15 calibration + portfolio_daily_ranking_v2_gated，必须先解释架构差异。",
            "P2：把下一主线命名为 end-to-end allocation layer，不再命名为局部 r39 修复；目标是替换旧主决策链，而不是继续包装旧 action path。",
            "P3：重构执行路径，让 solve_semidifferentiable_allocation 或其升级版成为 source/receiver/cash 的主 allocation 层，simulator guard 只做安全裁剪。",
            "P4：训练目标直接优化 allocation_final_objective、trade quality、cash deployment、risk-adjusted return、drawdown control、monthly quality，并用 confirm/stability 验证。",
        ],
    }


def _fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def render_markdown(audit: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Continuous Policy 主线循环审计 {audit['audit_date']}")
    lines.append("")
    lines.append(f"## 总判定")
    lines.append(f"- {audit['verdict']}")
    lines.append(f"- 当前应终止“r39 后继续追加局部 penalty / guard”的惯性路线，转向 end-to-end allocation layer。")
    lines.append("")
    lines.append("## 证据摘要")
    profile = audit["profile_continuity"]
    code = audit["code_path_status"]
    lines.append(f"- r34-r39 profile 数量：{profile['r34_r39_profile_count']}；其中 `portfolio_daily_ranking_v2_gated` 提及 {profile['r34_r39_v2_gated_mentions']} 次。")
    lines.append(f"- r34-r39 仍提及 `action_budget_split_v1` {profile['r34_r39_action_budget_split_v1_mentions']} 次，仍提及 v15 simulator calibration {profile['r34_r39_v15_calibration_mentions']} 次。")
    lines.append(f"- `portfolio_simulator.py` 中 `direct_action` 提及 {code['portfolio_simulator_direct_action_mentions']} 次，`guard` 提及 {code['portfolio_simulator_guard_mentions']} 次。")
    lines.append(f"- 半可微 optimizer 已存在：{code['semidifferentiable_optimizer_exists']}；但当前实现仍是顺序 sell-budget 再 buy-budget：{code['optimizer_is_sequential_sell_then_buy']}。")
    lines.append("")
    lines.append("## 1. 反复讨论的问题")
    for item in audit["repeated_problem_clusters"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 2. 旧方案换皮")
    for item in audit["renamed_old_solutions"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 3. 没有实质推进的局部修补")
    for item in audit["local_patches_without_progress"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 4. 已验证、不应重复论证的结论")
    for item in audit["validated_conclusions"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 5. 应废弃路线")
    for item in audit["routes_to_retire"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 6. 真正瓶颈")
    lines.append(f"- {audit['true_bottleneck']}")
    lines.append("")
    lines.append("## 主线阶段复盘")
    for phase in audit["mainline_phases"]:
        lines.append(f"- `{phase['phase']}`：{phase['theme']}。结论：{phase['conclusion']}")
    lines.append("")
    lines.append("## r34-r39 研究快照")
    lines.append("| study | phase | evidence | gate | annual | sharpe | cash | source | receiver | pos_src | spread | failed |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in audit["study_snapshots"]:
        gate = ""
        if row["passed_check_count"] is not None and row["total_check_count"] is not None:
            gate = f"{int(row['passed_check_count'])}/{int(row['total_check_count'])}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["study_tag"],
                    row["phase"],
                    row["training_evidence_status"],
                    gate,
                    _fmt_float(row["annual_return"]),
                    _fmt_float(row["sharpe"]),
                    _fmt_float(row["cash_reserve_rate"]),
                    _fmt_float(row["source_target_count"]),
                    _fmt_float(row["receiver_target_count"]),
                    _fmt_float(row["source_positive_forward_sell_share"]),
                    _fmt_float(row["receiver_minus_source_forward_excess_5d"]),
                    ",".join(row["failed_checks"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## 收敛后的优先级行动方案")
    for item in audit["priority_execution_plan"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 本审计结论")
    lines.append("- 下一主线不应再以“最小改动”追加局部惩罚。最有效路径是把旧 action translation 主路径降级为兼容层，把 source/receiver/cash 统一资金分配层提升为主路径。")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit continuous_policy research loops and emit a stop-cycle report.")
    parser.add_argument("--write", action="store_true", help="write JSON and Markdown audit artifacts")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="output directory relative to workspace")
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    args = parser.parse_args()

    audit = build_audit()
    markdown = render_markdown(audit)
    if args.write:
        output_dir = _workspace_path(Path(args.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"continuous_policy_cycle_audit_{AUDIT_DATE.replace('-', '')}"
        (output_dir / f"{stem}.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")
    if args.json:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
