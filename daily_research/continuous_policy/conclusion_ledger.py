from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.continuous_policy.runtime import (
    CONTINUOUS_POLICY_ROOT,
    LATEST_PROTOCOL_SUMMARY_PATH,
    now_iso,
    read_json,
    timestamp_tag,
    update_latest_summary,
    write_json,
)


ANALYSIS_ROOT = CONTINUOUS_POLICY_ROOT / "analysis" / "conclusion_ledgers"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh the Daily Research conclusion ledger.")
    parser.add_argument("--protocol-summary", default=str(LATEST_PROTOCOL_SUMMARY_PATH))
    parser.add_argument("--tag", default="")
    return parser


def _entry(
    *,
    conclusion_id: str,
    title: str,
    statement: str,
    confidence: str,
    evidence: list[str],
    revisit_trigger: str = "",
) -> dict[str, Any]:
    return {
        "id": conclusion_id,
        "title": title,
        "statement": statement,
        "confidence": confidence,
        "evidence": evidence,
        "revisit_trigger": revisit_trigger,
    }


def _safe_float(mapping: dict[str, Any], key: str) -> float:
    return float(mapping.get(key, 0.0) or 0.0)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol_summary_path = Path(str(args.protocol_summary or "")).expanduser().resolve()
    protocol_summary = read_json(protocol_summary_path)
    promotion_gate = dict(protocol_summary.get("promotion_gate", {}) or {})
    evaluation = dict(protocol_summary.get("evaluation", {}) or {})
    continuity = dict(evaluation.get("continuity_metrics", {}) or {})
    shadow = dict(protocol_summary.get("shadow", {}) or {})
    shadow_continuity = dict(shadow.get("continuity_metrics", {}) or {})
    train = dict(protocol_summary.get("train", {}) or {})
    training_evidence = dict(protocol_summary.get("training_evidence", {}) or {})
    latest_label_preset = str(protocol_summary.get("label_preset", "") or train.get("label_preset", "") or "")
    latest_backend = str(protocol_summary.get("trainer_backend", "") or train.get("trainer_backend", "") or "")
    failed_checks = {str(item) for item in (promotion_gate.get("failed_checks", []) or [])}
    hold_share = _safe_float(continuity, "hold_share")
    reduce_success_rate = _safe_float(continuity, "reduce_success_rate_5d")
    cash_timing_quality = _safe_float(continuity, "cash_timing_quality_1d")
    shadow_immediate_reversal = _safe_float(shadow_continuity, "immediate_reversal_rate_3d")

    local_bottleneck_parts: list[str] = []
    if hold_share < 0.12:
        local_bottleneck_parts.append(f"hold_share={hold_share:.4f}")
    if reduce_success_rate < 0.45:
        local_bottleneck_parts.append(f"reduce_success_rate_5d={reduce_success_rate:.4f}")
    if cash_timing_quality < 0.02:
        local_bottleneck_parts.append(f"cash_timing_quality_1d={cash_timing_quality:.4f}")
    if "shadow_reversal" in failed_checks or shadow_immediate_reversal > 0.35:
        local_bottleneck_parts.append(f"shadow_immediate_reversal_rate_3d={shadow_immediate_reversal:.4f}")
    if local_bottleneck_parts:
        behavior_bottleneck_statement = "当前 continuous_policy 的局部瓶颈已收口为：" + "、".join(local_bottleneck_parts)
        if hold_share >= 0.12:
            behavior_bottleneck_statement += f"；其中 hold_share 已提升到 {hold_share:.4f}，不再是第一瓶颈。"
        else:
            behavior_bottleneck_statement += "。"
    else:
        behavior_bottleneck_statement = "当前 continuous_policy 的主要局部瓶颈已明显缓解，需要进入新一轮 formal / shadow 证据再继续判定。"

    stable = [
        _entry(
            conclusion_id="live_default_still_frozen",
            title="Live 默认链路继续冻结",
            statement="live 默认执行仍应维持在 strongest-model / active manifest 主链路，不能因 continuous_policy 的单轮结果静默切换。",
            confidence="high",
            evidence=[
                "daily_research/output/active_execution_strategy.json",
                "daily_research/brain/state_center.md",
            ],
        ),
        _entry(
            conclusion_id="strongest_model_current_anchor",
            title="Strongest-Model 当前锚点稳定",
            statement="formal / recent / promotable 当前仍统一锚定 short_expert_monthly_v1，这一结论仍然稳定。",
            confidence="high",
            evidence=[
                "daily_research/brain/state_center.md",
                "daily_research/brain/identity_layer.md",
            ],
        ),
        _entry(
            conclusion_id="continuous_policy_contract_split_is_correct",
            title="Continuous Policy 合同拆分合理",
            statement="prototype_gbdt_v1 继续承担 shadow / teacher / ablation；formal_torch_v2 / formal_torch_seq_v3 / formal_torch_hier_v4 才承担 promotable formal candidate，这个治理拆分仍然合理。",
            confidence="high",
            evidence=[
                "daily_research/continuous_policy/training_contracts.py",
                "daily_research/brain/knowledge_center.md",
            ],
        ),
        _entry(
            conclusion_id="current_continuous_policy_shadow_only",
            title="Continuous Policy 仍是 Shadow Only",
            statement=f"latest protocol 的 promotion gate 仍为 {str(promotion_gate.get('status', 'unknown') or 'unknown')}，因此 continuous_policy 仍不能进入 live promotion。",
            confidence="high",
            evidence=[
                str(protocol_summary_path),
                "daily_research/brain/state_center.md",
            ],
        ),
    ]

    stage_local = [
        _entry(
            conclusion_id="latest_cp_protocol_stage_local",
            title="最新 Continuous Policy 结论是阶段性结论",
            statement=f"最新 protocol 使用 {latest_backend or 'unknown backend'} + {latest_label_preset or 'unknown preset'} 得到的结果，只代表当前状态表达、标签、decoder 与训练合同组合下的局部表现，不是跨架构终局结论。",
            confidence="medium",
            evidence=[str(protocol_summary_path)],
            revisit_trigger="任何状态表达、decoder profile、训练后端或时序模型发生变化后，都必须重新判定。",
        ),
        _entry(
            conclusion_id="policy_v5d_v5e_first_round_rejection",
            title="Policy v5d / v5e 是当前设计下的阶段性否定",
            statement="policy_v5d / policy_v5e 的否决应继续理解为当前 successor 设计下的阶段性失败，而不是整条方法族永久无效。",
            confidence="medium",
            evidence=[
                "daily_research/brain/episodic_memory.md",
                "daily_research/brain/state_center.md",
            ],
            revisit_trigger="如果未来切换到更强模型、不同状态表达或更自由动作空间，允许重新验证。",
        ),
        _entry(
            conclusion_id="current_behavior_bottlenecks_are_local",
            title="当前行为瓶颈是局部瓶颈",
            statement=behavior_bottleneck_statement,
            confidence="high",
            evidence=[str(protocol_summary_path)],
            revisit_trigger="只要 teacher / state / decoder / stronger model branch 任一升级，就必须重新量化。",
        ),
    ]
    if str(training_evidence.get("status", "") or "") != "sufficient":
        stage_local.append(
            _entry(
                conclusion_id="current_training_evidence_is_not_enough",
                title="当前训练证据仍不足",
                statement=(
                    "当前 protocol 的训练证据仍不足："
                    f"train_day_count={int(training_evidence.get('train_day_count', 0) or 0)}、"
                    f"teacher_action_rows={int(training_evidence.get('teacher_action_rows', 0) or 0)}、"
                    f"best_epoch={int(training_evidence.get('best_epoch', 0) or 0)}/completed_epochs={int(training_evidence.get('completed_epochs', 0) or 0)}。"
                    "在继续给出更强结论前，应先按 recommended_actions 做 strict resume 或扩训练窗口。"
                ),
                confidence="high",
                evidence=[str(protocol_summary_path)],
                revisit_trigger="当 training_evidence.status 变为 sufficient 后，才能把不足证据从阶段结论里移除。",
            )
        )

    strong_model_recheck = [
        _entry(
            conclusion_id="stronger_backbone_not_globally_rejected",
            title="更强 Backbone 不是全局禁区",
            statement="更强时序 backbone、Mamba、TSFM 或更自由的序列模型不再是全局否定对象；是否值得做，取决于信息增益和当前瓶颈是否持续存在。",
            confidence="high",
            evidence=[
                "daily_research/brain/identity_layer.md",
                "daily_research/brain/knowledge_center.md",
            ],
            revisit_trigger="当当前主线在 reduce / cash / early exit 上继续受限时，强模型复核应立即进入主计划。",
        ),
        _entry(
            conclusion_id="weak_model_failures_need_recheck_on_stronger_models",
            title="弱模型下失败的设定需要强模型复核",
            statement="凡是在弱模型、旧状态表达或旧动作空间下被淘汰的设定，都不能再被视为永久淘汰；要按信息增益优先级进入强模型复核池。",
            confidence="high",
            evidence=[
                "daily_research/brain/identity_layer.md",
                "daily_research/brain/operations_center.md",
            ],
            revisit_trigger="当连续策略或 successor 进入新训练后端时，应优先复核高价值旧设定。",
        ),
    ]

    next_actions = [
        "先用 behavior audit 明确当前 reduce / cash / early exit 的 teacher-vs-model 差距。",
        "先修当前主线的状态、标签、decoder 与组合预算头，再决定哪些旧结论需要在更强模型上复核。",
        "只有当当前主线继续明显卡住时，才继续扩大更强时序模型或更自由 backbone。",
    ]
    if str(training_evidence.get("status", "") or "") != "sufficient":
        next_actions = list(training_evidence.get("recommended_actions", []) or []) + next_actions

    payload = {
        "run_tag": str(args.tag or timestamp_tag("conclusion_ledger")),
        "generated_at": now_iso(),
        "source_protocol_summary": str(protocol_summary_path),
        "stable_conclusions": stable,
        "stage_local_conclusions": stage_local,
        "strong_model_recheck_conclusions": strong_model_recheck,
        "next_actions": next_actions,
    }
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = ANALYSIS_ROOT / f"{payload['run_tag']}.json"
    write_json(output_path, payload)
    payload["output_path"] = str(output_path.resolve())
    update_latest_summary("conclusion_ledger", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
