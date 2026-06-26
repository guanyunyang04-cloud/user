# Continuous Policy 方法库对象
快照日期：`2026-06-27`

本文件保留 continuous_policy 的长期方法论和历史索引。它不是当前 active/default 合同；当前 `daily_research` 研究指针见 `state_center.md`。

## object `continuous_policy_history`
`type`: archived_research_method_library
`definition`: 以日为单位进行连续决策的交易执行模型研究线，关注 `source / receiver / cash allocation`、组合资金流和持仓上下文。
`current_status`: research / shadow lineage；不承担 live/default、promotion 或 execution unfreeze。
`material_evidence`: `daily_research/brain/references/evidence_registry.json` and dated rXX references.
`activation`: 用户明确询问 continuous_policy、r-number、source/receiver/cash、DFL-PG、portfolio_set v5 或历史执行模型时激活。

## Object Semantics
### object `allocation_objective`
`purpose`: 学习当前组合状态下的资金接收、资金释放、现金保留、gross / turnover / cost 取舍。
`distinction`: 个股生命周期动作和组合预算语义分层；卖出不是简单看跌，而是释放资金给更优机会或风险状态。
`lesson`: 个股未来上涨不等于今天该加仓；正 forward 持仓也可能因机会成本被减仓。

### object `teacher_or_guard`
`purpose`: warm start、auxiliary prior、heuristic scaffold、最后安全裁剪和诊断。
`invariant`: teacher 或 simulator guard 不替代主策略目标，也不把短窗 smoke 变成正式 verdict。

### object `cashflow_contract`
`current_name`: `portfolio_cashflow_decision_v1`
`purpose`: 统一 prediction、simulator、release trace 与 continuity metrics 的 source/receiver/cash 语义。
`history`: r68 使 source/receiver/cash translation closure 首次可用；后续 r69-r74 主要围绕 behavior quality、value arbitration、lake features 和 evidence sufficiency。

### object `evidence_contract`
`levels`: protocol smoke、study summary、behavior evidence、promotion discussion.
`invariant`: completed protocol、completed study、model-quality evidence 和 execution authority 是不同层级。
`truth_sources`: protocol summary、study summary、registry、dataset catalog、reference。

## Historical Entries
- `r31-r39`: receiver/source/cash contract 与 allocation objective baseline。
- `r40-r48`: end-to-end allocation、convex/OPE/solver 方向；未过 stable confirm。
- `r49-r52`: capital-flow closure、native allocation vector、validation closure 研究链。
- `r53-r55`: cash-funded allocator、semantic budget、cash timing release controller。
- `r56-r61`: release-first allocator、core-v4、profile binding、decision-focused wiring。
- `r62-r64`: DuckDB + Parquet data lake 与 full-universe strict Gold。
- `r65`: portfolio-set v5 architecture upgrade。
- `r67`: paper-driven DFL-PG v1 replacement for portfolio-set v5。
- `r68`: cashflow decision v1 translation closure。
- `r69`: value arbitration behavior-quality mechanism。
- `r70`: version-boundary and oracle-feasibility repair。
- `r71`: multi-stage regret behavior-quality mechanism。
- `r72`: generic data lake evaluator for evaluate/shadow/export。
- `r73`: lake-native decision feature utilization and r71 source/receiver collapse repair。
- `r74`: lake-native behavior-quality feature contract and V5 behavior objective。

## Pure Functions
- `activate_continuous_policy(task)`: returns true only for explicit continuous_policy / rXX / source-receiver-cash tasks.
- `classify_cp_evidence(run)`: separates smoke, protocol-level evidence, study-level evidence, behavior evidence and promotion discussion.
- `map_cp_lesson_to_current_research(lesson)`: returns reusable prior, not execution authority.

## Procedure Entry
### procedure `inspect_continuous_policy_history`
`input`: r-number, run tag, mechanism question, or comparison request.
`steps`: query evidence registry；open explicit rXX reference or summary；classify evidence；return reusable lesson and current relevance.
`side_effects`: none unless user asks to update a reference.

## Current Relevance
- Useful as method library for portfolio allocation, cash/source/receiver semantics, evidence grading and run-status discipline.
- Not the current shortline scorer baseline.
- Not a live/default or active artifact authority.
- Full details remain in dated references and the machine registry.
