# Daily Research 状态中枢

快照日期：`2026-06-27`

## 当前对象

### `daily_research_project`
- 定义：当前正式生产研究与执行主线分脑。
- 当前阶段：执行端冻结，研究端已转向 QDP 数据基底上的收盘后短线选股。
- active 执行物化真源仍是 `daily_research/output/active_execution_strategy.json`；该对象只在执行、paper/live、active/default、trade plan 或 broker 任务中激活。

### `execution_freeze`
- 当前状态：`frozen_skeleton_only / awaiting_research_rebuild`。
- 语义：保留骨架、只读诊断、候选评估和手动流程说明；不恢复 active、不生成正式交易计划、不接 paper/live/broker，除非用户显式授权恢复或重建。
- 旧 live/default 历史标签：`short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`，仅作为历史 active 事实。

### `qdp_data_substrate`
- 定义：`daily_research` 当前唯一正式数据消费对象。
- 归属：QDP 负责 provider、ingest、lake catalog、canonical bundle、policy input loader、pool/sector-board view、memmap 和 training pack。
- 当前事实：QDP lake 已迁到 `quant_data_platform/data/lake`，`daily_research/output/research_data_lake` 已删除；本分脑只消费 QDP lake / manifest / memmap / training pack。
- 当前 replacement training pack：`quant_data_platform/data/memmap/training_pack/tradeable_mainboard_style_structural_alpha_v2_label_v2_backfilled_intraday_v2_training_pack_20260626_01/qdp_training_pack_manifest.json`。
- 当前 active replacement memmap：`tradeable_mainboard_style_structural_alpha_v2_label_v2_backfilled_intraday_v2_2010_2026_20260626_01`，`qdp validate-memmap` status `ok`，feature_count `307`，symbol_count `3025`。

### `provider_boundary`
- `daily_research` 不直连任何在线 provider。
- `mootdx_online`、`BaoStock`、`CNInfo` 等只作为 QDP 上游 ingest / raw archive / canonical 治理对象存在。
- 研究、训练和 diagnostics 只读取 QDP explicit lake dataset id、manifest、memmap 或 training pack。

### `shortline_after_close_research`
- 当前研究目标：收盘后短线选股。
- 工作流：D 收盘后更新 QDP，D+1 只有入场条件满足时买入，T+1 约束下最早 D+2 卖出。
- 机制方向：强势启动/延续、强势回踩低吸、行业扩散补涨。
- 已完成事实：
  - Stage 0 固定 D+1 open 诊断跑通，但 validation-selected same-candidate test 未通过。
  - 全历史 307 特征上涨潜力画像显示，固定 next-open baseline 下稳定关系主体偏 anti-overheat / anti-chase，而不是简单追涨。
  - raw/未标准化工程特征诊断确认 `turn`、`ret_10d`、`ret_20d`、行业相对涨幅、20d 高点距离和 local drawdown 等低值方向更稳。
  - `shortline_raw_condition_matrix_v1` 显示 `pullback_intraday_recovery` 是当前最强条件族，`market_confirmed_anti_overheat` 和 `industry_not_collapsing_not_overheat` 是次级上下文过滤。
- 下一步：把 pullback repair、market context、industry context 作为监督式 shortline scorer 或更窄条件网格的先验；不进入 promotion、candidate matrix、score-backtest bridge、active/default/live。

### `alpha_v2_history`
- `path_policy / alpha_v2` 复杂模型线保留为历史研究和可复用基础设施。
- 当前 anchor 仍是 `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01`，但它不是 multi-seed、candidate matrix、execution review 或 active/default 依据。
- 2026-06-24 后，优先级已从旧 path20/alpha_v2 复杂模型切到短线机制验证和 scorer。

## 当前优先级
1. 基于当前 QDP replacement pack 和 raw/engineering feature 证据，构建 shortline upside / entry scorer baseline。
2. 用 score decile、top-decile 期望、validation-selected same-candidate test 和成本后表现验证短线 scorer。
3. 需要新增数据时，先回到 QDP provider/canonical 对象，不在 `daily_research` 中临时在线取数。
4. 执行端继续冻结；任何 active/default/live/paper/broker 任务都单独激活 `daily_research_active_artifact`。

## 当前边界
- `formal`、`recent`、`promotion`、`live` 是不同对象，不混写。
- smoke、dry-run、short-window、interrupted、insufficient evidence、failed trial、realtime tail label 只能作为对应等级证据。
- completed run 不等于 completed model-quality evidence。
- `latest_*` 不是无条件真源；若 latest study/protocol/audit/ledger 不同源，使用 explicit run tag / protocol tag / dataset id。
- PowerShell 中文显示异常时，先用显式 UTF-8 复读，不直接判定文档损坏。

## 证据入口
- QDP replacement data base activation：`daily_research/brain/references/qdp_alpha_v2_replacement_data_base_activation_20260626.md`
- QDP data coverage audit：`daily_research/brain/references/qdp_alpha_v2_data_coverage_audit_and_backfill_plan_20260624.md`
- Shortline plan：`daily_research/brain/references/shortline_after_close_research_plan_20260624.md`
- Stage 0 fixed next-open diagnostic：`daily_research/brain/references/shortline_stage0_fixed_next_open_diagnostic_20260624.md`
- Shortline feature profile：`daily_research/brain/references/shortline_upside_feature_profile_20260624.md`
- Shortline feature semantics：`daily_research/brain/references/shortline_upside_feature_semantics_20260624.md`
- Shortline raw diagnostic：`daily_research/brain/references/shortline_raw_upside_diagnostic_20260624.md`
- Shortline raw condition matrix：`daily_research/brain/references/shortline_raw_condition_matrix_20260624.md`
- Machine evidence index：`daily_research/brain/references/evidence_registry.json`

## 归档入口
- 历史归档：`daily_research/brain/references/state_center_archive_20260510.md`
- 早期历史：`daily_research/brain/references/state_center_history_raw_20260424.md`
- 早期 evidence index：`daily_research/brain/references/state_center_evidence_index_20260424.md`
